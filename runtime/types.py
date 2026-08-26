from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeHandle:
    model_root: str
    device: str = "cuda:0"
    memory_mode: str = "auto"
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
