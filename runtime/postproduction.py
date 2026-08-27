from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import uuid
import wave
from pathlib import Path
from typing import Any

LOUDNORM_JSON_RE = re.compile(r'\{\s*"input_i".*?\}', re.DOTALL)


def prepare_synchronized_ab(
    source_a: str | Path,
    source_b: str | Path,
    output_a: str | Path,
    output_b: str | Path,
    *,
    synchronize_onset: bool = True,
    match_loudness: bool = True,
    target_lufs: float = -20.0,
    onset_threshold_dbfs: float = -42.0,
    preroll_ms: int = 20,
    sample_rate: int = 24000,
) -> dict[str, Any]:
    """Create source-preserving A/B files with aligned speech onset and loudness."""
    import torch

    if not -35.0 <= float(target_lufs) <= -12.0:
        raise ValueError("A/B 目标响度必须在 -35…-12 LUFS")
    if not -70.0 <= float(onset_threshold_dbfs) <= -10.0:
        raise ValueError("起点阈值必须在 -70…-10 dBFS")
    if not 0 <= int(preroll_ms) <= 500:
        raise ValueError("起点预留必须在 0…500 ms")
    first_source = Path(source_a).resolve()
    second_source = Path(source_b).resolve()
    first_target = Path(output_a).resolve()
    second_target = Path(output_b).resolve()
    if not first_source.is_file() or not second_source.is_file():
        raise FileNotFoundError("A/B 输入音频不存在")
    if first_source == first_target or second_source == second_target:
        raise ValueError("A/B 处理不能覆盖输入音频")

    first, first_rate = _read_pcm16(first_source)
    second, second_rate = _read_pcm16(second_source)
    first = _resample(first, first_rate, sample_rate)
    second = _resample(second, second_rate, sample_rate)
    channels = max(int(first.shape[0]), int(second.shape[0]))
    first = _match_channels(first, channels)
    second = _match_channels(second, channels)
    first_onset = _detect_onset(first, sample_rate, onset_threshold_dbfs)
    second_onset = _detect_onset(second, sample_rate, onset_threshold_dbfs)
    preroll = round(int(preroll_ms) * sample_rate / 1000)
    first_trim = max(0, first_onset - preroll) if synchronize_onset else 0
    second_trim = max(0, second_onset - preroll) if synchronize_onset else 0
    first = first[:, first_trim:]
    second = second[:, second_trim:]
    if first.numel() == 0 or second.numel() == 0:
        raise ValueError("A/B 起点同步后音频为空，请降低阈值")

    workspace = first_target.parent / f".ab-{uuid.uuid4().hex}"
    workspace.mkdir(parents=True, exist_ok=False)
    raw_a = workspace / "A-raw.wav"
    raw_b = workspace / "B-raw.wav"
    normalized_a = workspace / "A-normalized.wav"
    normalized_b = workspace / "B-normalized.wav"
    try:
        _write_pcm16(raw_a, first, sample_rate)
        _write_pcm16(raw_b, second, sample_rate)
        before_a = analyze_loudness(raw_a)
        before_b = analyze_loudness(raw_b)
        if match_loudness:
            master_wav(
                raw_a,
                normalized_a,
                target_lufs=target_lufs,
                loudness_range_lu=7.0,
                true_peak_dbfs=-1.0,
                sample_rate=sample_rate,
            )
            master_wav(
                raw_b,
                normalized_b,
                target_lufs=target_lufs,
                loudness_range_lu=7.0,
                true_peak_dbfs=-1.0,
                sample_rate=sample_rate,
            )
        else:
            shutil.copy2(raw_a, normalized_a)
            shutil.copy2(raw_b, normalized_b)
        final_a, rate_a = _read_pcm16(normalized_a)
        final_b, rate_b = _read_pcm16(normalized_b)
        if rate_a != sample_rate or rate_b != sample_rate:
            raise RuntimeError("A/B 输出采样率异常")
        length = max(int(final_a.shape[-1]), int(final_b.shape[-1]))
        first_padded = torch.nn.functional.pad(final_a, (0, length - final_a.shape[-1]))
        second_padded = torch.nn.functional.pad(final_b, (0, length - final_b.shape[-1]))
        _write_pcm16(first_target, first_padded, sample_rate)
        _write_pcm16(second_target, second_padded, sample_rate)
        after_a = analyze_loudness(first_target)
        after_b = analyze_loudness(second_target)
        return {
            "source_preserved": True,
            "synchronize_onset": bool(synchronize_onset),
            "match_loudness": bool(match_loudness),
            "target_lufs": float(target_lufs) if match_loudness else None,
            "sample_rate": sample_rate,
            "duration_seconds": length / sample_rate,
            "A": {
                "output_path": str(first_target),
                "detected_onset_seconds": first_onset / sample_rate,
                "trimmed_seconds": first_trim / sample_rate,
                "before": before_a,
                "after": after_a,
            },
            "B": {
                "output_path": str(second_target),
                "detected_onset_seconds": second_onset / sample_rate,
                "trimmed_seconds": second_trim / sample_rate,
                "before": before_b,
                "after": after_b,
            },
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def master_wav(
    source_path: str | Path,
    output_path: str | Path,
    *,
    target_lufs: float,
    loudness_range_lu: float,
    true_peak_dbfs: float,
    sample_rate: int,
) -> dict[str, Any]:
    """Run deterministic two-pass FFmpeg EBU R128 normalization."""
    source = Path(source_path).resolve()
    target = Path(output_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"母带输入不存在：{source}")
    if not -70.0 <= float(target_lufs) <= -5.0:
        raise ValueError("target_lufs 必须在 -70…-5")
    if not 1.0 <= float(loudness_range_lu) <= 20.0:
        raise ValueError("loudness_range_lu 必须在 1…20")
    if not -9.0 <= float(true_peak_dbfs) <= 0.0:
        raise ValueError("true_peak_dbfs 必须在 -9…0")
    loudnorm = (
        f"loudnorm=I={float(target_lufs):g}:LRA={float(loudness_range_lu):g}:"
        f"TP={float(true_peak_dbfs):g}"
    )
    first = _run_ffmpeg(
        [_ffmpeg_path(), "-hide_banner", "-nostats", "-i", str(source), "-af", f"{loudnorm}:print_format=json", "-f", "null", "NUL" if os.name == "nt" else "/dev/null"],
        timeout=600,
    )
    matches = LOUDNORM_JSON_RE.findall(first.stderr)
    if not matches:
        raise RuntimeError(f"FFmpeg loudnorm 未返回测量结果：{first.stderr[-1200:].strip()}")
    measured = json.loads(matches[-1])
    required = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    if any(key not in measured for key in required):
        raise RuntimeError("FFmpeg loudnorm 测量结果缺字段")
    second = (
        f"{loudnorm}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=true:print_format=summary"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp.wav")
    try:
        _run_ffmpeg(
            [_ffmpeg_path(), "-hide_banner", "-nostats", "-y", "-i", str(source), "-af", second, "-ar", str(int(sample_rate)), "-c:a", "pcm_s16le", str(temporary)],
            timeout=600,
        )
        if not temporary.is_file() or temporary.stat().st_size <= 44:
            raise RuntimeError("FFmpeg 母带未生成有效 WAV")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "target_lufs": float(target_lufs),
        "loudness_range_lu": float(loudness_range_lu),
        "true_peak_dbfs": float(true_peak_dbfs),
        "first_pass": {key: measured[key] for key in required},
        "after": analyze_loudness(target),
    }


def analyze_loudness(path: str | Path) -> dict[str, Any]:
    result = _run_ffmpeg(
        [_ffmpeg_path(), "-hide_banner", "-nostats", "-i", str(Path(path).resolve()), "-af", "loudnorm=I=-23:LRA=7:TP=-1:print_format=json", "-f", "null", "NUL" if os.name == "nt" else "/dev/null"],
        timeout=600,
    )
    matches = LOUDNORM_JSON_RE.findall(result.stderr)
    if not matches:
        raise RuntimeError("FFmpeg 未返回响度分析")
    value = json.loads(matches[-1])
    return {
        "integrated_lufs": _finite_number(value.get("input_i")),
        "true_peak_dbfs": _finite_number(value.get("input_tp")),
        "loudness_range_lu": _finite_number(value.get("input_lra")),
    }


def _detect_onset(audio: Any, sample_rate: int, threshold_dbfs: float) -> int:
    import torch

    mono = audio.float().mean(dim=0).abs()
    if mono.numel() == 0 or float(mono.max().item()) < 1e-6:
        raise ValueError("A/B 输入是静音，无法检测起点")
    window = max(1, round(sample_rate * 0.01))
    energy = torch.nn.functional.avg_pool1d(
        mono.square().view(1, 1, -1), window, stride=max(1, window // 2), ceil_mode=True
    ).sqrt().flatten()
    absolute = 10 ** (float(threshold_dbfs) / 20.0)
    relative = float(energy.max().item()) * (10 ** (-35.0 / 20.0))
    indices = torch.nonzero(energy >= max(absolute, relative), as_tuple=False)
    if indices.numel() == 0:
        raise ValueError("未检测到有效声音起点，请降低阈值")
    return min(int(audio.shape[-1]) - 1, int(indices[0].item()) * max(1, window // 2))


def _read_pcm16(path: str | Path):
    import torch

    with wave.open(str(path), "rb") as reader:
        if reader.getsampwidth() != 2:
            raise ValueError(f"仅支持 PCM16 WAV：{path}")
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
    audio = torch.frombuffer(bytearray(frames), dtype=torch.int16).clone()
    audio = audio.reshape(-1, channels).transpose(0, 1).float() / 32768.0
    return audio, sample_rate


def _write_pcm16(path: str | Path, audio: Any, sample_rate: int) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pcm = (audio.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).transpose(0, 1).contiguous()
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(int(audio.shape[0]))
        writer.setsampwidth(2)
        writer.setframerate(int(sample_rate))
        writer.writeframes(pcm.numpy().tobytes())


def _resample(audio: Any, source_rate: int, target_rate: int):
    if source_rate == target_rate:
        return audio
    import torch.nn.functional as functional

    length = max(1, round(audio.shape[-1] * target_rate / source_rate))
    return functional.interpolate(audio.unsqueeze(0), size=length, mode="linear", align_corners=False).squeeze(0)


def _match_channels(audio: Any, channels: int):
    if int(audio.shape[0]) == channels:
        return audio
    if int(audio.shape[0]) == 1:
        return audio.repeat(channels, 1)
    raise ValueError("A/B 音频声道数不兼容")


def _ffmpeg_path() -> str:
    configured = os.environ.get("FIREREDAUDIO_FFMPEG", "").strip()
    if configured and Path(configured).is_file():
        return configured
    bundled = Path(__file__).resolve().parents[1] / "tools" / "ffmpeg.exe"
    if os.name == "nt" and bundled.is_file():
        return str(bundled)
    located = shutil.which("ffmpeg")
    if located:
        return located
    raise RuntimeError("同步 A/B 与交付母带需要 FFmpeg")


def _run_ffmpeg(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg 处理失败：{completed.stderr[-1600:].strip()}")
    return completed


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
