from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path
from typing import Any

from .audio_inputs import prepare_audio_path
from .errors import WorkerProtocolError


def analyze_audio(value: str | Path) -> dict[str, Any]:
    """Analyze an input after the normal decoder path has converted it to PCM16 WAV."""
    prepared = Path(prepare_audio_path(value))
    with wave.open(str(prepared), "rb") as reader:
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        sample_width = reader.getsampwidth()
        frame_count = reader.getnframes()
        if sample_width != 2 or reader.getcomptype() != "NONE":
            raise WorkerProtocolError("音频质检要求 PCM16 WAV")
        payload = reader.readframes(frame_count)

    samples = array("h")
    samples.frombytes(payload)
    if not samples:
        raise WorkerProtocolError("音频没有可分析的采样")
    count = len(samples)
    peak = max(abs(int(value)) for value in samples)
    square_sum = sum(int(value) * int(value) for value in samples)
    rms = math.sqrt(square_sum / count)
    dc = sum(int(value) for value in samples) / count
    clip_count = sum(1 for value in samples if abs(int(value)) >= 32760)
    silence_threshold = int(32767 * 10 ** (-50 / 20))
    silence_count = sum(1 for value in samples if abs(int(value)) <= silence_threshold)
    duration = frame_count / sample_rate if sample_rate else 0.0

    issues: list[str] = []
    if duration < 1.0:
        issues.append("参考音频短于 1 秒，音色稳定性可能不足")
    elif duration > 30.0:
        issues.append("参考音频长于 30 秒，建议裁剪为干净的代表性片段")
    clipping_ratio = clip_count / count
    if clipping_ratio > 0.001:
        issues.append("检测到明显削波")
    silence_ratio = silence_count / count
    if silence_ratio > 0.5:
        issues.append("静音比例过高")
    dc_ratio = abs(dc) / 32768.0
    if dc_ratio > 0.01:
        issues.append("检测到明显直流偏移")
    if channels > 2:
        issues.append("输入超过双声道，建议先转换为单声道或双声道")

    suggested_actions: list[str] = []
    if silence_ratio > 0.5:
        suggested_actions.append("可生成非破坏式清理副本，自动裁掉首尾长静音")
    if dc_ratio > 0.01:
        suggested_actions.append("可生成清理副本，通过语音安全高通减轻直流偏移")
    if channels != 1 or sample_rate != 24000:
        suggested_actions.append("可生成 24 kHz 单声道 PCM16 标准副本")
    if clipping_ratio > 0.001:
        suggested_actions.append("削波无法可靠还原，优先更换未削波的原始录音")
    if duration < 1.0:
        suggested_actions.append("补录至少 1 秒，推荐使用 3–15 秒连续干净语音")
    elif duration > 30.0:
        suggested_actions.append("人工选取 3–15 秒最清晰、内容与逐字稿一致的片段")
    suggested_actions.append("清理副本不执行降噪或去混响；存在背景声时应优先更换素材")

    return {
        "source_path": str(Path(value).expanduser().resolve()),
        "prepared_path": str(prepared),
        "duration_seconds": round(duration, 3),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bits": sample_width * 8,
        "frames": frame_count,
        "peak_dbfs": _dbfs(float(peak)),
        "rms_dbfs": _dbfs(rms),
        "clipping_ratio": round(clipping_ratio, 6),
        "silence_ratio": round(silence_ratio, 6),
        "dc_offset_ratio": round(dc_ratio, 6),
        "issues": issues,
        "recommended": not issues,
        "suggested_actions": suggested_actions,
        "automatic_cleanup_available": True,
    }


def _dbfs(value: float) -> float | None:
    if value <= 0:
        return None
    return round(20.0 * math.log10(value / 32767.0), 2)
