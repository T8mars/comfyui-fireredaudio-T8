from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeHandle:
    model_root: str
    device: str = "auto"
    memory_mode: str = "auto"
    acceleration_mode: str = "auto_safe"
    runtime_python: str = ""
    worker_url: str = ""
    worker_token: str = ""
    verify_hashes: bool = False
    release_after: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationSettings:
    quality_preset: str = "balanced"
    seed: int = 42
    max_new_audio_steps: int = 750
    min_new_audio_steps: int = 6
    max_new_text_tokens: int = 512
    n_timesteps: int = 10
    inference_cfg: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        presets = {
            "fast": (500, 384, 6, 1.8),
            "balanced": (750, 512, 10, 2.0),
            "high_quality": (1200, 768, 16, 2.2),
        }
        if self.quality_preset in presets:
            audio, text, timesteps, cfg = presets[self.quality_preset]
            data.update(
                max_new_audio_steps=audio,
                max_new_text_tokens=text,
                n_timesteps=timesteps,
                inference_cfg=cfg,
            )
        return data


@dataclass(frozen=True)
class DeliveryPreset:
    """A named, inspectable post-production contract shared by render and save nodes."""

    name: str
    mode: str
    gap_ms: int
    crossfade_ms: int
    target_lufs: float
    loudness_range_lu: float
    true_peak_dbfs: float
    sample_rate: int
    audio_format: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DELIVERY_PRESETS: dict[str, DeliveryPreset] = {
    "audiobook": DeliveryPreset(
        name="audiobook",
        mode="sequence",
        gap_ms=0,
        crossfade_ms=40,
        target_lufs=-20.0,
        loudness_range_lu=7.0,
        true_peak_dbfs=-3.0,
        sample_rate=24000,
        audio_format="flac",
    ),
    "podcast": DeliveryPreset(
        name="podcast",
        mode="sequence",
        gap_ms=0,
        crossfade_ms=80,
        target_lufs=-16.0,
        loudness_range_lu=7.0,
        true_peak_dbfs=-1.0,
        sample_rate=48000,
        audio_format="wav",
    ),
    "video_dialogue": DeliveryPreset(
        name="video_dialogue",
        mode="timeline",
        gap_ms=0,
        crossfade_ms=30,
        target_lufs=-23.0,
        loudness_range_lu=11.0,
        true_peak_dbfs=-1.0,
        sample_rate=48000,
        audio_format="wav",
    ),
}


def delivery_preset(name: str) -> DeliveryPreset:
    try:
        return DELIVERY_PRESETS[str(name)]
    except KeyError as exc:
        raise ValueError(f"未知交付预设：{name}") from exc
