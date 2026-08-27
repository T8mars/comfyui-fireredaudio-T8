"""Single-GPU acceleration selection with explicit, safe fallbacks."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import asdict, dataclass
from importlib import metadata

import torch


MODES = ("off", "auto_safe", "flash_attention", "fla_liger", "torch_compile", "deepspeed")


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def probe_acceleration(device: str | torch.device | None = None) -> dict:
    requested_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    cuda = bool(requested_device.type == "cuda" and torch.cuda.is_available())
    torch_abi_ok = bool(
        os.name != "nt"
        or (torch.__version__.startswith("2.8.0") and str(torch.version.cuda).startswith("12.8"))
    )
    return {
        "cuda": cuda,
        "torch_abi_ok": torch_abi_ok,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "modules": {
            "flash_attn": _has_module("flash_attn"),
            "fla": _has_module("fla"),
            "causal_conv1d": _has_module("causal_conv1d"),
            "liger_kernel": _has_module("liger_kernel"),
            "triton": _has_module("triton"),
            "deepspeed": _has_module("deepspeed"),
        },
        "packages": {
            name: _version(name)
            for name in (
                "flash-attn",
                "flash-linear-attention",
                "liger-kernel",
                "triton-windows",
                "deepspeed",
            )
        },
    }


@dataclass(frozen=True, slots=True)
class AccelerationSelection:
    requested: str
    effective: str
    attention_backend: str = "sdpa"
    use_fla: bool = False
    use_liger: bool = False
    use_torch_compile: bool = False
    use_deepspeed: bool = False
    available: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_acceleration(
    mode: str | None,
    device: str | torch.device | None = None,
    capabilities: dict | None = None,
) -> AccelerationSelection:
    requested = str(mode or "auto_safe").strip().lower()
    if requested not in MODES:
        raise ValueError(f"acceleration_mode must be one of {', '.join(MODES)}")
    if requested == "off":
        return AccelerationSelection(requested, "off", reason="已关闭可选加速")

    caps = capabilities or probe_acceleration(device)
    if not caps["cuda"]:
        return AccelerationSelection(
            requested, "off", available=False, reason="当前不是 CUDA 设备，已回退 SDPA"
        )
    if not caps["torch_abi_ok"]:
        return AccelerationSelection(
            requested,
            "off",
            available=False,
            reason=f"Windows 加速轮子要求 torch 2.8.0+cu128，当前为 {caps['torch']}",
        )

    modules = caps["modules"]
    flash = bool(modules["flash_attn"])
    fla = bool(modules["fla"] and modules["triton"])
    fla_full = bool(fla and modules.get("causal_conv1d"))
    if requested == "auto_safe":
        if not (flash or fla):
            return AccelerationSelection(
                requested, "off", available=False, reason="FlashAttention/FLA 不可用，已回退 SDPA"
            )
        # FLA's first Triton specialization is expensive on Windows and the audited
        # wheel set has no causal-conv1d binary. Keep the predictable FlashAttention
        # path as the default; expose partial FLA as an explicit experimental mode.
        use_fla = bool(fla and not flash)
        enabled = ["FlashAttention"] if flash else ["FLA（部分快路径）"]
        return AccelerationSelection(
            requested,
            "auto_safe",
            attention_backend="flash_attention_2" if flash else "sdpa",
            use_fla=use_fla,
            reason=" + ".join(enabled) + " 已通过实际内核探测",
        )
    if requested == "flash_attention":
        return AccelerationSelection(
            requested,
            requested if flash else "off",
            attention_backend="flash_attention_2" if flash else "sdpa",
            available=flash,
            reason="FlashAttention 可用" if flash else "FlashAttention 不可用，已回退 SDPA",
        )
    if requested == "fla_liger":
        ready = bool(fla and modules["liger_kernel"])
        detail = (
            "FLA + Liger 完整快路径可用"
            if fla_full
            else "FLA gated-delta + Liger 可用；causal-conv1d 使用 PyTorch 回退"
        )
        return AccelerationSelection(
            requested,
            requested if ready else "off",
            attention_backend="flash_attention_2" if ready and flash else "sdpa",
            use_fla=ready,
            use_liger=ready,
            available=ready,
            reason=detail if ready else "缺少 FLA/Triton/Liger，已回退 SDPA",
        )
    if requested == "torch_compile":
        ready = bool(modules["triton"] and hasattr(torch, "compile"))
        return AccelerationSelection(
            requested,
            requested if ready else "off",
            attention_backend="flash_attention_2" if ready and flash else "sdpa",
            use_fla=ready and fla,
            use_torch_compile=ready,
            available=ready,
            reason="Triton + torch.compile 可用；首次运行会生成缓存" if ready else "Triton/torch.compile 不可用，已回退",
        )

    ready = bool(modules["deepspeed"])
    return AccelerationSelection(
        requested,
        "deepspeed" if ready else "off",
        attention_backend="flash_attention_2" if ready and flash else "sdpa",
        use_deepspeed=ready,
        available=ready,
        reason="DeepSpeed 单卡 BF16 InferenceEngine 可用" if ready else "DeepSpeed 不可用，已回退",
    )


__all__ = ["AccelerationSelection", "MODES", "probe_acceleration", "resolve_acceleration"]
