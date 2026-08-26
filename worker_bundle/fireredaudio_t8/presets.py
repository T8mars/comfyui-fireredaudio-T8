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
    if name not in QUALITY_PRESETS:
        allowed = "/".join(QUALITY_PRESETS)
        raise WorkerProtocolError(f"quality_preset 必须是 {allowed}")
    resolved = dict(request)
    resolved["quality_preset"] = name
    for key, value in QUALITY_PRESETS[name].items():
        resolved.setdefault(key, value)
    return resolved
