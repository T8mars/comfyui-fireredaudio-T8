"""Comfy-Kitchen ConvRot INT8 serialization and model loading.

The checkpoint keeps ordinary parameters under their upstream names.  Selected
Linear weights are stored as INT8 at ``<module>.weight`` with a companion
``<module>.weight_scale`` tensor.  The loader rebuilds Comfy-Kitchen's tensor
subclass before any inference code can observe the weight.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .configuration_fireredaudio import FireRedAudioConfig
from .modeling_fireredaudio import FireRedAudioForCausalLM
from .quantization import read_quantization_metadata


logger = logging.getLogger(__name__)
FORMAT = "comfy-kitchen-convrot-int8"


def configure_convrot_backend() -> None:
    """Select a backend order that is reliable in the bundled Windows runtime."""

    try:
        import comfy_kitchen
    except ImportError as exc:
        raise RuntimeError(
            "ConvRot INT8 模型需要 comfy-kitchen==0.2.31"
        ) from exc

    # Some comfy-kitchen Windows wheels bundle a CUDA extension compiled against
    # a newer toolkit than the installed driver.  Triton uses PyTorch's working
    # CUDA runtime and is therefore the safe first choice in the portable build.
    if os.name == "nt":
        comfy_kitchen.set_backend_priority(["triton", "eager"])
    else:
        comfy_kitchen.set_backend_priority(["cuda", "triton", "eager"])


def make_convrot_weight(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int,
    original_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Rebuild one serialized Comfy-Kitchen ConvRot tensor subclass."""

    configure_convrot_backend()
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

    params = TensorWiseINT8Layout.Params(
        scale=scale,
        orig_dtype=original_dtype,
        orig_shape=tuple(qdata.shape),
        is_weight=True,
        convrot=True,
        convrot_groupsize=int(group_size),
    )
    return QuantizedTensor(qdata, "TensorWiseINT8Layout", params)


def _assign_state_tensor(
    model: FireRedAudioForCausalLM,
    name: str,
    tensor: torch.Tensor,
) -> None:
    module_name, tensor_name = name.rsplit(".", 1)
    module = model.get_submodule(module_name)
    if tensor_name in module._parameters:
        module._parameters[tensor_name] = torch.nn.Parameter(
            tensor, requires_grad=False
        )
        return
    if tensor_name in module._buffers:
        module._buffers[tensor_name] = tensor
        return
    raise KeyError(f"模型结构中不存在状态张量：{name}")


def _checkpoint_files(model_dir: Path) -> list[Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"无效的 safetensors 索引：{index_path}")
        return [model_dir / item for item in sorted(set(weight_map.values()))]
    single = model_dir / "model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"缺少 ConvRot safetensors 权重：{model_dir}")


def _initialize_nonpersistent_buffers(model: FireRedAudioForCausalLM) -> None:
    """Recreate buffers omitted from safetensors while parameters stay on meta."""

    for module in model.modules():
        meta_names = [
            name
            for name, buffer in module._buffers.items()
            if buffer is not None and buffer.is_meta
        ]
        if not meta_names:
            continue
        if hasattr(module, "config"):
            shadow = type(module)(module.config, device="cpu")
        elif all(hasattr(module, name) for name in ("length", "channels", "max_timescale")):
            shadow = type(module)(
                module.length,
                module.channels,
                module.max_timescale,
            )
        elif all(hasattr(module, name) for name in ("_rope_dim", "_rope_base")):
            shadow = type(module)(
                module._rope_dim,
                interpolation_factor=module.interpolation_factor,
                base=module._rope_base,
            )
        else:
            raise ValueError(
                f"无法重建非持久化 buffer：{type(module).__module__}.{type(module).__name__}"
            )
        for name in meta_names:
            replacement = shadow._buffers.get(name)
            if replacement is None or replacement.is_meta:
                raise ValueError(f"重建 buffer 失败：{type(module).__name__}.{name}")
            module._buffers[name] = replacement


def load_convrot_pretrained(
    model_name_or_path: str | Path,
    *,
    config: FireRedAudioConfig,
    dtype: torch.dtype = torch.bfloat16,
) -> FireRedAudioForCausalLM:
    """Stream a FireRedAudio ConvRot checkpoint into a meta-initialized model."""

    root = Path(model_name_or_path).expanduser().resolve()
    metadata = read_quantization_metadata(root)
    if str(metadata.get("format", "")).lower() != FORMAT:
        raise ValueError(f"不是 FireRedAudio ConvRot 模型：{root}")
    groups: Any = metadata.get("convrot_group_sizes")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("ConvRot 元数据缺少 convrot_group_sizes")
    group_sizes = {str(key): int(value) for key, value in groups.items()}

    configure_convrot_backend()
    with torch.device("meta"):
        model = FireRedAudioForCausalLM(config)

    loaded: set[str] = set()
    for checkpoint in _checkpoint_files(root):
        shard = load_file(str(checkpoint), device="cpu")
        for weight_name, group_size in group_sizes.items():
            if weight_name not in shard:
                continue
            scale_name = f"{weight_name.removesuffix('.weight')}.weight_scale"
            if scale_name not in shard:
                raise ValueError(f"ConvRot 权重缺少 scale：{weight_name}")
            weight = make_convrot_weight(
                shard[weight_name],
                shard[scale_name],
                group_size=group_size,
                original_dtype=dtype,
            )
            _assign_state_tensor(model, weight_name, weight)
            loaded.add(weight_name)

        for name, tensor in shard.items():
            if name.endswith(".weight_scale") or name in loaded:
                continue
            _assign_state_tensor(model, name, tensor)
            loaded.add(name)
        del shard

    _initialize_nonpersistent_buffers(model)
    missing_parameters = [
        name for name, parameter in model.named_parameters() if parameter.is_meta
    ]
    missing_buffers = [name for name, buffer in model.named_buffers() if buffer.is_meta]
    if missing_parameters or missing_buffers:
        examples = (missing_parameters + missing_buffers)[:12]
        raise ValueError(
            f"ConvRot 检查点不完整，仍有 meta 张量：{', '.join(examples)}"
        )
    absent_quantized = sorted(set(group_sizes) - loaded)
    if absent_quantized:
        raise ValueError(
            f"ConvRot 检查点缺少 {len(absent_quantized)} 个量化权重："
            f"{', '.join(absent_quantized[:8])}"
        )

    model.eval()
    model._fireredaudio_convrot = {
        "format": FORMAT,
        "quantized_module_count": len(group_sizes),
        "backend_priority": ["triton", "eager"]
        if os.name == "nt"
        else ["cuda", "triton", "eager"],
    }
    logger.info("Loaded %d ConvRot INT8 weights from %s", len(group_sizes), root)
    return model


__all__ = [
    "FORMAT",
    "configure_convrot_backend",
    "load_convrot_pretrained",
    "make_convrot_weight",
]
