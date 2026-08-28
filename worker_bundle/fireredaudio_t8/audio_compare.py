from __future__ import annotations

import shutil
import uuid
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .audio_inputs import prepare_audio_path
from .audio_post import master_audio
from .errors import WorkerProtocolError
from .production_quality import analyze_production_audio


def prepare_synchronized_ab(
    source_a: str | Path,
    source_b: str | Path,
    output_directory: str | Path,
    *,
    target_lufs: float = -20.0,
    sync_onset: bool = True,
    match_loudness: bool = True,
    sample_rate: int = 24_000,
) -> dict[str, Any]:
    """Create non-destructive, onset-aligned and loudness-matched A/B previews."""
    left = Path(source_a).expanduser().resolve()
    right = Path(source_b).expanduser().resolve()
    if not left.is_file() or not right.is_file():
        raise WorkerProtocolError("A/B 对比音频不存在")
    if left == right:
        raise WorkerProtocolError("A/B 对比必须选择两个不同版本")
    if not -35.0 <= float(target_lufs) <= -12.0:
        raise WorkerProtocolError("A/B 匹配响度必须在 -35…-12 LUFS")

    output_root = Path(output_directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    session = output_root / f"ab-{uuid.uuid4().hex}"
    session.mkdir(parents=False, exist_ok=False)
    synced_a = session / ".a-synced.wav"
    synced_b = session / ".b-synced.wav"
    output_a = session / "A.wav"
    output_b = session / "B.wav"
    try:
        waveform_a, source_rate_a = _read_mono(left)
        waveform_b, source_rate_b = _read_mono(right)
        waveform_a = _resample(waveform_a, source_rate_a, sample_rate)
        waveform_b = _resample(waveform_b, source_rate_b, sample_rate)
        onset_a = _detect_onset(waveform_a, sample_rate) if sync_onset else 0
        onset_b = _detect_onset(waveform_b, sample_rate) if sync_onset else 0
        pre_roll = int(round(0.02 * sample_rate))
        trim_a = max(0, onset_a - pre_roll)
        trim_b = max(0, onset_b - pre_roll)
        waveform_a = waveform_a[trim_a:]
        waveform_b = waveform_b[trim_b:]
        if waveform_a.size == 0 or waveform_b.size == 0:
            raise WorkerProtocolError("A/B 起点同步后没有有效音频")
        _write_mono(synced_a, waveform_a, sample_rate)
        _write_mono(synced_b, waveform_b, sample_rate)

        mastering: dict[str, Any] = {}
        if match_loudness:
            mastering["a"] = master_audio(
                synced_a,
                output_a,
                target_lufs=float(target_lufs),
                loudness_range_lu=7.0,
                true_peak_dbfs=-2.0,
            )
            mastering["b"] = master_audio(
                synced_b,
                output_b,
                target_lufs=float(target_lufs),
                loudness_range_lu=7.0,
                true_peak_dbfs=-2.0,
            )
        else:
            shutil.copy2(synced_a, output_a)
            shutil.copy2(synced_b, output_b)
        _pad_pair(output_a, output_b)
        quality_a = analyze_production_audio(
            output_a, target_lufs=float(target_lufs), tolerance_lu=1.0, true_peak_ceiling_dbfs=-2.0
        )
        quality_b = analyze_production_audio(
            output_b, target_lufs=float(target_lufs), tolerance_lu=1.0, true_peak_ceiling_dbfs=-2.0
        )
        return {
            "session_id": session.name,
            "a_path": str(output_a),
            "b_path": str(output_b),
            "source_a": str(left),
            "source_b": str(right),
            "sync_onset": bool(sync_onset),
            "match_loudness": bool(match_loudness),
            "target_lufs": float(target_lufs),
            "trimmed_leading_seconds": {
                "a": round(trim_a / sample_rate, 6),
                "b": round(trim_b / sample_rate, 6),
            },
            "quality": {"a": quality_a, "b": quality_b},
            "mastering": mastering,
            "source_preserved": True,
        }
    except Exception:
        shutil.rmtree(session, ignore_errors=True)
        raise
    finally:
        synced_a.unlink(missing_ok=True)
        synced_b.unlink(missing_ok=True)


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    prepared = Path(prepare_audio_path(path))
    try:
        with wave.open(str(prepared), "rb") as handle:
            if handle.getsampwidth() != 2:
                raise WorkerProtocolError(f"A/B 音频不是 PCM16：{path}")
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            waveform = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32)
    except (wave.Error, OSError) as exc:
        raise WorkerProtocolError(f"A/B 音频无法读取：{path}") from exc
    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(axis=1)
    return waveform / 32768.0, sample_rate


def _resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or waveform.size == 0:
        return waveform.astype(np.float32, copy=False)
    target_length = max(1, int(round(len(waveform) * target_rate / source_rate)))
    positions = np.linspace(0.0, max(0, len(waveform) - 1), target_length)
    return np.interp(positions, np.arange(len(waveform)), waveform).astype(np.float32)


def _detect_onset(waveform: np.ndarray, sample_rate: int) -> int:
    if waveform.size == 0:
        return 0
    peak = float(np.max(np.abs(waveform)))
    if peak <= 1e-7:
        return 0
    window = max(1, int(round(0.02 * sample_rate)))
    hop = max(1, int(round(0.005 * sample_rate)))
    threshold = max(float(10 ** (-45.0 / 20.0)), peak * 0.02)
    squared = np.square(waveform.astype(np.float64, copy=False))
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    for start in range(0, max(1, len(waveform) - window + 1), hop):
        end = min(len(waveform), start + window)
        rms = float(np.sqrt((cumulative[end] - cumulative[start]) / max(1, end - start)))
        if rms >= threshold:
            return start
    return 0


def _pad_pair(left: Path, right: Path) -> None:
    waveform_a, rate_a = _read_mono(left)
    waveform_b, rate_b = _read_mono(right)
    if rate_a != rate_b:
        waveform_b = _resample(waveform_b, rate_b, rate_a)
    length = max(len(waveform_a), len(waveform_b))
    if len(waveform_a) < length:
        waveform_a = np.pad(waveform_a, (0, length - len(waveform_a)))
    if len(waveform_b) < length:
        waveform_b = np.pad(waveform_b, (0, length - len(waveform_b)))
    _write_mono(left, waveform_a, rate_a)
    _write_mono(right, waveform_b, rate_a)


def _write_mono(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pcm = np.round(np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    temporary.replace(path)
