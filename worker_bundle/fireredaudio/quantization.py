"""Quantized model profiles shared by converters and the inference loader.

The stable profile deliberately quantizes only the Qwen3.5 block linears.  Audio
encoders, embeddings, the language-model head, RedAE and the flow modules stay in
their released precision until the acceptance suite proves a broader profile safe.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable


METADATA_NAME = "fireredaudio_quantization.json"
SUPPORTED_PROFILES = (
    "bf16-slim",
    "int8-wo-safe",
    "int8-wo-extended",
    "int8-convrot-experimental",
)

_SCOPE_PATTERNS = OrderedDict(
    (
        (
            "backbone",
            r"backbone_llm\.model\.layers\.\d+\."
            r"(?:linear_attn\.(?:out_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a)"
            r"|self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
            r"|mlp\.(?:gate_proj|up_proj|down_proj))\.weight",
        ),
        (
            "audio_encoder",
            r"audio_encoder\.layers\.\d+\."
            r"(?:self_attn\.(?:k_proj|v_proj|q_proj|out_proj)|fc1|fc2)\.weight",
        ),
        (
            "red_vae",
            r"red_vae\.(?:qwen3\.layers|downsample\.qwen3\.layers)\.\d+\."
            r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
            r"|mlp\.(?:gate_proj|up_proj|down_proj))\.weight",
        ),
        (
            "patch_encoder",
            r"patch_encoder\.blocks\.\d+\."
            r"(?:attn\.(?:to_q|to_k|to_v|to_out\.0)"
            r"|mlp\.ff\.(?:0\.0|2))\.weight",
        ),
        (
            "dit",
            r"dit\.blocks\.\d+\."
            r"(?:attn\.(?:to_q|to_k|to_v|to_out\.0)"
            r"|mlp\.ff\.(?:0\.0|2)|adaLN_modulation\.1)\.weight",
        ),
    )
)


def read_quantization_metadata(model_dir: str | Path) -> dict[str, Any]:
    """Return FireRedAudio quantization metadata, or a BF16 fallback description."""

    root = Path(model_dir)
    target = root / METADATA_NAME
    if target.is_file():
        data = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"量化元数据必须是 JSON 对象：{target}")
        return data

    config_path = root / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        quantization = config.get("quantization_config")
        if isinstance(quantization, dict):
            return {
                "schema_version": 1,
                "profile": "torchao-prequantized",
                "format": str(quantization.get("quant_method") or "quantized"),
                "stable": False,
                "source": "config.json",
            }
    return {
        "schema_version": 1,
        "profile": "bf16-upstream",
        "format": "bfloat16",
        "stable": True,
        "source": "implicit",
    }


def is_torchao_quantized(model_dir: str | Path) -> bool:
    metadata = read_quantization_metadata(model_dir)
    return str(metadata.get("format", "")).lower() in {
        "torchao-int8-weight-only",
        "torchao",
    }


def is_weight_quantized(model_dir: str | Path) -> bool:
    """Return whether the directory uses one of the supported low-bit formats."""

    metadata = read_quantization_metadata(model_dir)
    return str(metadata.get("format", "")).lower() in {
        "torchao-int8-weight-only",
        "torchao",
        "comfy-kitchen-convrot-int8",
    }


def matching_scope(weight_fqn: str, scopes: Iterable[str]) -> str | None:
    """Return the selected scope matching an exact Linear weight FQN."""

    for scope in tuple(dict.fromkeys(str(item).strip() for item in scopes)):
        pattern = _SCOPE_PATTERNS.get(scope)
        if pattern is None:
            raise ValueError(f"未知量化范围：{scope}")
        if re.fullmatch(pattern, weight_fqn):
            return scope
    return None


def convrot_group_size(in_features: int) -> int | None:
    """Choose the largest Comfy ConvRot Hadamard group supported by K."""

    return next((size for size in (256, 64, 16) if int(in_features) % size == 0), None)


def build_torchao_config(scopes: Iterable[str] = ("backbone",)):
    """Build a serializable TorchAO INT8 weight-only FQN configuration.

    Imports remain local so original BF16 models can still be inspected on a
    machine that has not installed the optional quantized runtime.
    """

    selected = tuple(dict.fromkeys(str(scope).strip() for scope in scopes))
    unknown = sorted(set(selected) - set(_SCOPE_PATTERNS))
    if unknown:
        raise ValueError(
            f"未知量化范围：{', '.join(unknown)}；可用值：{', '.join(_SCOPE_PATTERNS)}"
        )
    if not selected:
        raise ValueError("至少选择一个量化范围")

    try:
        from torchao.quantization import FqnToConfig, Int8WeightOnlyConfig
        from transformers import TorchAoConfig
    except ImportError as exc:
        raise RuntimeError(
            "INT8 模型需要 torchao==0.16.0；请先安装整合包固定运行时依赖"
        ) from exc

    mapping = OrderedDict(
        (
            f"re:{_SCOPE_PATTERNS[scope]}",
            Int8WeightOnlyConfig(version=2),
        )
        for scope in selected
    )
    return TorchAoConfig(FqnToConfig(fqn_to_config=mapping))


def stable_acceleration_mode(
    requested: str,
    model_dir: str | Path,
) -> tuple[str, str | None]:
    """Disable acceleration wrappers not yet compatible with tensor subclasses."""

    if not is_weight_quantized(model_dir):
        return requested, None
    normalized = str(requested or "auto_safe").strip().lower()
    if normalized in {"deepspeed", "torch_compile"}:
        return (
            "auto_safe",
            f"INT8 权重模式暂不叠加 {normalized}，已安全回退 FlashAttention/SDPA",
        )
    return normalized, None


__all__ = [
    "METADATA_NAME",
    "SUPPORTED_PROFILES",
    "build_torchao_config",
    "convrot_group_size",
    "is_torchao_quantized",
    "is_weight_quantized",
    "matching_scope",
    "read_quantization_metadata",
    "stable_acceleration_mode",
]
