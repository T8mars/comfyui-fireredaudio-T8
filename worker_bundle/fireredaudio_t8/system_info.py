from __future__ import annotations

import platform
from typing import Any


GIB = 1024**3
FULL_GPU_MIN_FREE_GIB = 36.0


def gpu_inventory(
    *,
    full_gpu_min_free_bytes: int | None = None,
    active_device: str | None = None,
    active_memory_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Return the GPUs visible to the isolated worker without mutating CUDA state."""
    try:
        import torch
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []

    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        total = int(props.total_memory)
        free: int | None = None
        try:
            free_value, total_value = torch.cuda.mem_get_info(index)
            free, total = int(free_value), int(total_value)
        except Exception:
            pass
        free_gib = round(free / GIB, 2) if free is not None else None
        device_id = f"cuda:{index}"
        threshold = int(
            full_gpu_min_free_bytes
            if full_gpu_min_free_bytes is not None
            else FULL_GPU_MIN_FREE_GIB * GIB
        )
        if active_device == device_id and active_memory_mode in {
            "full_gpu",
            "sequential",
            "decoder_cpu",
        }:
            # Once a model is resident, current free VRAM is no longer a valid
            # load-time signal. Report the mode actually selected for that engine.
            recommended = str(active_memory_mode)
        else:
            recommended = (
                "full_gpu"
                if free is not None and free >= threshold
                else "sequential"
            )
        devices.append(
            {
                "id": device_id,
                "index": index,
                "name": props.name,
                "total_bytes": total,
                "total_gib": round(total / GIB, 2),
                "free_bytes": free,
                "free_gib": free_gib,
                "compute_capability": f"{props.major}.{props.minor}",
                "recommended_memory_mode": recommended,
                "recommendation_min_free_bytes": threshold,
            }
        )
    return devices


def runtime_readiness() -> dict[str, Any]:
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_runtime = str(torch.version.cuda or "")
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        torch_version = None
        cuda_runtime = None
        cuda_available = False
        torch_error = str(exc)
    else:
        torch_error = None

    gpus = gpu_inventory()
    issues: list[str] = []
    if torch_error:
        issues.append(f"PyTorch 无法导入：{torch_error}")
    elif not cuda_available:
        issues.append("隔离运行时未检测到可用 NVIDIA CUDA 设备")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch_version,
        "cuda_runtime": cuda_runtime,
        "cuda_available": cuda_available,
        "gpus": gpus,
        "recommended_device": gpus[0]["id"] if gpus else "cpu",
        "recommended_memory_mode": (
            gpus[0]["recommended_memory_mode"] if gpus else "decoder_cpu"
        ),
        "issues": issues,
        "ready": not issues,
    }
