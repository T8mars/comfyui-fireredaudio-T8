from __future__ import annotations

from typing import Any

from .errors import WorkerProtocolError

QUALITY_PRESETS: dict[str, dict[str, int | float]] = {
    "fast": {
        "max_new_audio_steps": 500,
        "min_new_audio_steps": 6,
        "max_new_text_tokens": 384,
        "n_timesteps": 6,
        "inference_cfg": 1.8,
    },
    "balanced": {
        "max_new_audio_steps": 750,
        "min_new_audio_steps": 6,
        "max_new_text_tokens": 512,
        "n_timesteps": 10,
        "inference_cfg": 2.0,
    },
    "high_quality": {
        "max_new_audio_steps": 1200,
        "min_new_audio_steps": 8,
        "max_new_text_tokens": 768,
        "n_timesteps": 16,
        "inference_cfg": 2.2,
    },
}


def apply_quality_preset(request: dict[str, Any]) -> dict[str, Any]:
    """Apply preset defaults while preserving every explicitly supplied value."""
    name = str(request.get("quality_preset") or "balanced")
    if name not in {*QUALITY_PRESETS, "custom"}:
        allowed = "/".join([*QUALITY_PRESETS, "custom"])
        raise WorkerProtocolError(f"quality_preset 必须是 {allowed}")
    resolved = dict(request)
    resolved["quality_preset"] = name
    if name == "custom":
        _validate_custom(resolved)
    else:
        for key, value in QUALITY_PRESETS[name].items():
            resolved.setdefault(key, value)
    return resolved


def _validate_custom(request: dict[str, Any]) -> None:
    integer_ranges = {
        "max_new_audio_steps": (6, 3000),
        "min_new_audio_steps": (1, 3000),
        "max_new_text_tokens": (1, 4096),
        "n_timesteps": (1, 100),
    }
    for key, (minimum, maximum) in integer_ranges.items():
        try:
            value = int(request[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkerProtocolError(f"custom 质量参数缺少有效 {key}") from exc
        if not minimum <= value <= maximum:
            raise WorkerProtocolError(f"custom {key} 必须在 {minimum}–{maximum} 范围内")
        request[key] = value
    if request["min_new_audio_steps"] > request["max_new_audio_steps"]:
        raise WorkerProtocolError("custom min_new_audio_steps 不能大于 max_new_audio_steps")
    try:
        cfg = float(request["inference_cfg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkerProtocolError("custom 质量参数缺少有效 inference_cfg") from exc
    if not 0.0 <= cfg <= 10.0:
        raise WorkerProtocolError("custom inference_cfg 必须在 0–10 范围内")
    request["inference_cfg"] = cfg
