"""Model loading with selectable, capability-probed acceleration."""

import logging
import os

import torch
from torch import nn
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5

from .acceleration import AccelerationSelection, resolve_acceleration
from .configuration_fireredaudio import FireRedAudioConfig
from .convrot import FORMAT as CONVROT_FORMAT, load_convrot_pretrained
from .modeling_fireredaudio import FireRedAudioForCausalLM
from .quantization import read_quantization_metadata, stable_acceleration_mode

logger = logging.getLogger(__name__)

_QWEN_ORIGINAL_RMS_NORM = qwen3_5.Qwen3_5RMSNorm
_QWEN_ORIGINAL_MLP = qwen3_5.Qwen3_5MLP
_QWEN_FLA_COMPONENTS = {
    "chunk_gated_delta_rule": qwen3_5.chunk_gated_delta_rule,
    "fused_recurrent_gated_delta_rule": qwen3_5.fused_recurrent_gated_delta_rule,
    "FusedRMSNormGated": qwen3_5.FusedRMSNormGated,
}


def load_fireredaudio(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
    acceleration_mode: str = "auto_safe",
) -> FireRedAudioForCausalLM:
    """Load the model for inference.

    Args:
        model_name_or_path: Directory holding config.json and safetensors shards.
        dtype: Weight dtype; bfloat16 matches the released weights.
        device: Moved there when given.

    Returns:
        A FireRedAudioForCausalLM in eval mode.
    """
    config = FireRedAudioConfig.from_pretrained(model_name_or_path)
    quantization = read_quantization_metadata(model_name_or_path)
    selected_mode, quantization_fallback = stable_acceleration_mode(
        acceleration_mode, model_name_or_path
    )
    selection = resolve_acceleration(selected_mode, device)
    if quantization_fallback:
        selection = AccelerationSelection(
            requested=str(acceleration_mode),
            effective=selection.effective,
            attention_backend=selection.attention_backend,
            use_fla=selection.use_fla,
            use_liger=selection.use_liger,
            use_torch_compile=False,
            use_deepspeed=False,
            available=selection.available,
            reason=f"{quantization_fallback}；{selection.reason}",
        )
    _configure_qwen(selection)
    attn = selection.attention_backend

    # dit, patch_encoder and vae/downsample hardcode their attention and ignore this.
    config.backbone_config._attn_implementation = attn
    config.audio_encoder_config._attn_implementation = attn
    config.red_vae_config._attn_implementation = attn

    if str(quantization.get("format", "")).lower() == CONVROT_FORMAT:
        model = load_convrot_pretrained(
            model_name_or_path,
            config=config,
            dtype=dtype,
        )
    else:
        model = FireRedAudioForCausalLM.from_pretrained(
            model_name_or_path, config=config, dtype=dtype
        )
    model.eval()
    if device is not None:
        model.to(device)
    if selection.use_torch_compile:
        model.dit = torch.compile(model.dit, mode="reduce-overhead", dynamic=True)
    if selection.use_deepspeed:
        _apply_deepspeed(model)
    model._fireredaudio_acceleration = selection.to_dict()
    model._fireredaudio_quantization = quantization
    return model


def _configure_qwen(selection: AccelerationSelection) -> None:
    qwen3_5.Qwen3_5RMSNorm = _QWEN_ORIGINAL_RMS_NORM
    qwen3_5.Qwen3_5MLP = _QWEN_ORIGINAL_MLP
    for name, implementation in _QWEN_FLA_COMPONENTS.items():
        setattr(qwen3_5, name, implementation if selection.use_fla else None)
    qwen3_5.is_fast_path_available = bool(
        selection.use_fla
        and qwen3_5.causal_conv1d_fn is not None
        and qwen3_5.causal_conv1d_update is not None
    )
    if selection.use_liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

        apply_liger_kernel_to_qwen3_5(
            rope=False,
            rms_norm=True,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=False,
        )


class _DeepSpeedModelProxy(nn.Module):
    """Keep Qwen attributes available while routing forward through DeepSpeed."""

    def __init__(self, engine: nn.Module):
        super().__init__()
        self.engine = engine

    def forward(self, *args, **kwargs):
        return self.engine(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            engine = super().__getattr__("engine")
            return getattr(engine.module, name)


def _apply_deepspeed(model: FireRedAudioForCausalLM) -> None:
    os.environ.setdefault("DS_BUILD_OPS", "0")
    import deepspeed

    inner = model.backbone_llm.model
    engine = deepspeed.init_inference(
        inner,
        config={
            "dtype": torch.bfloat16,
            "replace_with_kernel_inject": False,
            "enable_cuda_graph": False,
        },
    )
    model.backbone_llm.model = _DeepSpeedModelProxy(engine)
