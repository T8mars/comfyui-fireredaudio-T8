from __future__ import annotations

import math
import wave
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .audio_inputs import prepare_audio_path
from .errors import WorkerProtocolError


@dataclass(frozen=True)
class RenderedClip:
    id: str
    source_path: str
    track: str
    requested_start: float
    actual_start: float
    duration: float
    drift_seconds: float
    source_in: float
    source_out: float


@dataclass(frozen=True)
class TimelineRenderResult:
    output_path: str
    strategy: str
    duration_seconds: float
    sample_rate: int
    clips: list[RenderedClip]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "strategy": self.strategy,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "clips": [asdict(clip) for clip in self.clips],
            "warnings": self.warnings,
        }


def render_timeline(
    clips: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    strategy: str = "sequence",
    sample_rate: int = 24_000,
    allow_overlap: bool = False,
    headroom_db: float = 0.5,
) -> TimelineRenderResult:
    mode = str(strategy or "sequence").lower()
    if mode not in {"sequence", "timeline", "overlay"}:
        raise WorkerProtocolError("时间线策略必须是 sequence/timeline/overlay")
    normalized = [_normalize_clip(index, value) for index, value in enumerate(clips)]
    normalized = [clip for clip in normalized if not clip["muted"]]
    if not normalized:
        raise WorkerProtocolError("没有可渲染的时间线片段")
    if mode in {"timeline", "overlay"}:
        normalized.sort(key=lambda value: (value["position"], value["order_index"]))
    decoded: list[tuple[dict[str, Any], np.ndarray]] = []
    for clip in normalized:
        waveform = _read_pcm16_mono(clip["path"], sample_rate)
        start = min(len(waveform), int(round(clip["in_offset"] * sample_rate)))
        if clip["out_offset"] > 0:
            end = min(len(waveform), int(round(clip["out_offset"] * sample_rate)))
        elif clip["duration"] > 0:
            end = min(len(waveform), start + int(round(clip["duration"] * sample_rate)))
        else:
            end = len(waveform)
        if end <= start:
            raise WorkerProtocolError(f"片段 {clip['id']} 的裁剪范围为空")
        waveform = waveform[start:end].astype(np.float32, copy=True)
        waveform *= float(10 ** (clip["gain_db"] / 20.0))
        _apply_fades(waveform, clip["fade_in"], clip["fade_out"], sample_rate)
        decoded.append((clip, waveform))

    placements: list[tuple[dict[str, Any], np.ndarray, int]] = []
    rendered_clips: list[RenderedClip] = []
    warnings: list[str] = []
    cursor = 0
    previous_end = 0
    for clip, waveform in decoded:
        requested = int(round(clip["position"] * sample_rate))
        if mode == "sequence":
            start_sample = cursor
        elif mode == "timeline":
            start_sample = max(requested, cursor)
            if start_sample > requested:
                drift = (start_sample - requested) / sample_rate
                warnings.append(
                    f"片段 {clip['id']} 超出目标时间槽，整体顺延 {drift:.3f} 秒；未截断语音"
                )
        else:
            start_sample = requested
            if start_sample < previous_end and not allow_overlap:
                raise WorkerProtocolError(
                    f"overlay 检测到意外重叠：片段 {clip['id']}；确认后启用 allow_overlap"
                )
        end_sample = start_sample + len(waveform)
        cursor = max(cursor, end_sample)
        previous_end = max(previous_end, end_sample)
        placements.append((clip, waveform, start_sample))
        rendered_clips.append(
            RenderedClip(
                id=clip["id"],
                source_path=str(clip["path"]),
                track=clip["track"],
                requested_start=round(requested / sample_rate, 6),
                actual_start=round(start_sample / sample_rate, 6),
                duration=round(len(waveform) / sample_rate, 6),
                drift_seconds=round((start_sample - requested) / sample_rate, 6),
                source_in=clip["in_offset"],
                source_out=round((int(round(clip["in_offset"] * sample_rate)) + len(waveform)) / sample_rate, 6),
            )
        )
    mix = np.zeros(cursor, dtype=np.float32)
    for _clip, waveform, start_sample in placements:
        mix[start_sample : start_sample + len(waveform)] += waveform
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    ceiling = float(10 ** (-abs(headroom_db) / 20.0))
    if peak > ceiling and peak > 0:
        gain = ceiling / peak
        mix *= gain
        warnings.append(
            f"混音峰值 {20 * math.log10(peak):.2f} dBFS，整体衰减 {20 * math.log10(gain):.2f} dB 防止削波"
        )
    target = Path(output_path).expanduser().resolve()
    if target.suffix.lower() != ".wav":
        raise WorkerProtocolError("时间线基础渲染必须输出 WAV；其他格式在导出阶段转码")
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_pcm16_atomic(target, mix, sample_rate)
    return TimelineRenderResult(
        output_path=str(target),
        strategy=mode,
        duration_seconds=round(len(mix) / sample_rate, 6),
        sample_rate=sample_rate,
        clips=rendered_clips,
        warnings=warnings,
    )


def render_track_stems(
    clips: Iterable[dict[str, Any]],
    output_directory: str | Path,
    *,
    strategy: str = "timeline",
    sample_rate: int = 24_000,
    allow_overlap: bool = False,
) -> dict[str, TimelineRenderResult]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for clip in clips:
        if clip.get("muted"):
            continue
        track = str(clip.get("track") or "dialogue")[:80]
        grouped.setdefault(track, []).append(clip)
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    results: dict[str, TimelineRenderResult] = {}
    for track, values in grouped.items():
        safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in track)
        results[track] = render_timeline(
            values,
            output / f"{safe_name or 'track'}.wav",
            strategy=strategy,
            sample_rate=sample_rate,
            allow_overlap=allow_overlap,
        )
    return results


def _normalize_clip(index: int, value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkerProtocolError(f"第 {index + 1} 个时间线片段不是对象")
    source = Path(str(value.get("path") or value.get("audio_path") or "")).expanduser().resolve()
    if not source.is_file():
        raise WorkerProtocolError(f"时间线素材不存在：{source}")
    position = float(value.get("position") or value.get("target_start") or 0.0)
    if position < 0:
        raise WorkerProtocolError("时间线位置不能小于 0")
    return {
        "id": str(value.get("id") or f"clip-{index + 1}"),
        "order_index": int(value.get("order_index", index)),
        "path": source,
        "track": str(value.get("track") or value.get("speaker") or "dialogue")[:80],
        "position": position,
        "duration": max(0.0, float(value.get("duration") or 0.0)),
        "in_offset": max(0.0, float(value.get("in_offset") or 0.0)),
        "out_offset": max(0.0, float(value.get("out_offset") or 0.0)),
        "gain_db": float(value.get("gain_db") or 0.0),
        "fade_in": max(0.0, float(value.get("fade_in") or 0.0)),
        "fade_out": max(0.0, float(value.get("fade_out") or 0.0)),
        "muted": bool(value.get("muted", False)),
    }


def _read_pcm16_mono(path: Path, sample_rate: int) -> np.ndarray:
    prepared = prepare_audio_path(path)
    with wave.open(str(prepared), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise WorkerProtocolError(f"时间线素材不是 PCM16：{path}")
        if handle.getframerate() != sample_rate:
            raise WorkerProtocolError(f"时间线素材采样率不是 {sample_rate} Hz：{path}")
        channels = handle.getnchannels()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples / 32768.0


def _apply_fades(waveform: np.ndarray, fade_in: float, fade_out: float, sample_rate: int) -> None:
    in_samples = min(len(waveform), int(round(fade_in * sample_rate)))
    out_samples = min(len(waveform), int(round(fade_out * sample_rate)))
    if in_samples > 0:
        waveform[:in_samples] *= np.linspace(0.0, 1.0, in_samples, dtype=np.float32)
    if out_samples > 0:
        waveform[-out_samples:] *= np.linspace(1.0, 0.0, out_samples, dtype=np.float32)


def _write_pcm16_atomic(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pcm = np.round(np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    temporary.replace(path)
