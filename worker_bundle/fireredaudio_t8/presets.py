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

PARAMETER_RANGES: dict[str, tuple[type, float, float]] = {
    "max_new_audio_steps": (int, 6, 3000),
    "min_new_audio_steps": (int, 1, 750),
    "max_new_text_tokens": (int, 1, 4096),
    "n_timesteps": (int, 1, 100),
    "inference_cfg": (float, 0.0, 10.0),
}


def _normalized_parameters(request: dict[str, Any]) -> dict[str, int | float]:
    normalized: dict[str, int | float] = {}
    for key, (converter, minimum, maximum) in PARAMETER_RANGES.items():
        if key not in request or request[key] is None or request[key] == "":
            raise WorkerProtocolError(f"自定义质量参数缺少有效的 {key}")
        try:
            value = converter(request[key])
        except (TypeError, ValueError) as exc:
            raise WorkerProtocolError(f"自定义质量参数缺少有效的 {key}") from exc
        if not minimum <= value <= maximum:
            raise WorkerProtocolError(f"{key} 必须在 {minimum:g}–{maximum:g} 范围内")
        normalized[key] = value
    if normalized["min_new_audio_steps"] > normalized["max_new_audio_steps"]:
        raise WorkerProtocolError("min_new_audio_steps 不能大于 max_new_audio_steps")
    return normalized


def apply_quality_preset(request: dict[str, Any]) -> dict[str, Any]:
    """Apply preset defaults while preserving every explicitly supplied value."""
    name = str(request.get("quality_preset") or "balanced")
    if name != "custom" and name not in QUALITY_PRESETS:
        allowed = "/".join([*QUALITY_PRESETS, "custom"])
        raise WorkerProtocolError(f"quality_preset 必须是 {allowed}")
    resolved = dict(request)
    resolved["quality_preset"] = name
    if name == "custom":
        resolved.update(_normalized_parameters(resolved))
    else:
        for key, value in QUALITY_PRESETS[name].items():
            resolved.setdefault(key, value)
    return resolved
