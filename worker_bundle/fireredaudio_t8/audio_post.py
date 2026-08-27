from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio_inputs import prepare_audio_path
from .errors import WorkerProtocolError


LOUDNORM_JSON_RE = re.compile(r"\{\s*\"input_i\".*?\}", re.DOTALL)


def master_audio(
    source_path: str | Path,
    output_path: str | Path,
    *,
    target_lufs: float = -16.0,
    loudness_range_lu: float = 11.0,
    true_peak_dbfs: float = -1.0,
    highpass_hz: float | None = None,
) -> dict[str, Any]:
    """Two-pass EBU R128 normalization with an optional speech-safe high-pass."""
    source = Path(source_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise WorkerProtocolError(f"母带处理输入不存在：{source}")
    if target.suffix.lower() != ".wav":
        raise WorkerProtocolError("母带处理当前要求 WAV 输出")
    if not -70.0 <= float(target_lufs) <= -5.0:
        raise WorkerProtocolError("target_lufs 必须在 -70…-5 之间")
    if not -9.0 <= float(true_peak_dbfs) <= 0.0:
        raise WorkerProtocolError("true_peak_dbfs 必须在 -9…0 之间")
    if highpass_hz is not None and not 20.0 <= float(highpass_hz) <= 300.0:
        raise WorkerProtocolError("highpass_hz 必须在 20…300 Hz 之间")

    ffmpeg = _find_ffmpeg()
    base_filters = []
    if highpass_hz is not None:
        base_filters.append(f"highpass=f={float(highpass_hz):g}")
    loudnorm = (
        f"loudnorm=I={float(target_lufs):g}:LRA={float(loudness_range_lu):g}:"
        f"TP={float(true_peak_dbfs):g}"
    )
    first_filter = ",".join([*base_filters, f"{loudnorm}:print_format=json"])
    first = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(source),
            "-af",
            first_filter,
            "-f",
            "null",
            "NUL" if os.name == "nt" else "/dev/null",
        ],
        timeout=600,
    )
    matches = LOUDNORM_JSON_RE.findall(first.stderr)
    if not matches:
        raise WorkerProtocolError(f"FFmpeg loudnorm 未返回测量 JSON：{first.stderr[-1200:].strip()}")
    try:
        measured = json.loads(matches[-1])
    except json.JSONDecodeError as exc:
        raise WorkerProtocolError("FFmpeg loudnorm 测量 JSON 无效") from exc
    required = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    if any(key not in measured for key in required):
        raise WorkerProtocolError("FFmpeg loudnorm 测量结果缺字段")
    second_loudnorm = (
        f"{loudnorm}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
        f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
        f"offset={measured['target_offset']}:linear=true:print_format=summary"
    )
    second_filter = ",".join([*base_filters, second_loudnorm])
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp.wav")
    try:
        _run(
            [
                ffmpeg,
                "-hide_banner",
                "-nostats",
                "-y",
                "-i",
                str(source),
                "-af",
                second_filter,
                "-ar",
                "24000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            timeout=600,
        )
        if not temporary.is_file() or temporary.stat().st_size <= 44:
            raise WorkerProtocolError("FFmpeg 母带处理未生成有效 WAV")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "output_path": str(target),
        "target_lufs": float(target_lufs),
        "loudness_range_lu": float(loudness_range_lu),
        "true_peak_dbfs": float(true_peak_dbfs),
        "highpass_hz": None if highpass_hz is None else float(highpass_hz),
        "first_pass": {key: measured.get(key) for key in required},
    }


def prepare_reference_audio(
    source_path: str | Path,
    output_path: str | Path,
    *,
    trim_silence: bool = True,
    normalize_loudness: bool = False,
    target_lufs: float = -23.0,
    highpass_hz: float | None = 60.0,
) -> dict[str, Any]:
    """Create a conservative, non-destructive TTS reference copy.

    This deliberately does not denoise, dereverberate, repair clipping, or overwrite
    the source. Those operations can alter speaker identity and must not be hidden
    behind an automatic "repair" button.
    """
    from .audio_quality import analyze_audio

    source = Path(source_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise WorkerProtocolError(f"参考音频不存在：{source}")
    if source == target:
        raise WorkerProtocolError("参考音频清理不能覆盖原文件")
    if target.suffix.lower() != ".wav":
        raise WorkerProtocolError("参考音频清理只输出 WAV")
    if target.exists():
        raise WorkerProtocolError(f"参考音频清理目标已存在：{target}")
    if highpass_hz is not None and not 20.0 <= float(highpass_hz) <= 300.0:
        raise WorkerProtocolError("highpass_hz 必须在 20…300 Hz 之间")
    if not -35.0 <= float(target_lufs) <= -12.0:
        raise WorkerProtocolError("参考音频 target_lufs 必须在 -35…-12 之间")

    before = analyze_audio(source)
    prepared = Path(prepare_audio_path(source))
    ffmpeg = _find_ffmpeg()
    target.parent.mkdir(parents=True, exist_ok=True)
    intermediate = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.prepared.wav")
    filters: list[str] = []
    if highpass_hz is not None:
        filters.append(f"highpass=f={float(highpass_hz):g}")
    if trim_silence:
        filters.append(
            "silenceremove=start_periods=1:start_duration=0.05:start_threshold=-45dB:"
            "stop_periods=-1:stop_duration=0.10:stop_threshold=-45dB"
        )
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-y",
        "-i",
        str(prepared),
    ]
    if filters:
        command.extend(["-af", ",".join(filters)])
    command.extend(["-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(intermediate)])
    metadata_path = target.with_suffix(target.suffix + ".json")
    try:
        _run(command, timeout=600)
        if not intermediate.is_file() or intermediate.stat().st_size <= 44:
            raise WorkerProtocolError("参考音频清理后没有有效语音，请降低裁剪强度或更换素材")
        if normalize_loudness:
            master_audio(
                intermediate,
                target,
                target_lufs=float(target_lufs),
                loudness_range_lu=7.0,
                true_peak_dbfs=-2.0,
                highpass_hz=None,
            )
        else:
            intermediate.replace(target)
        after = analyze_audio(target)
        report = {
            "schema_version": 1,
            "operation": "reference_copy_cleanup",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_path": str(source),
            "output_path": str(target),
            "metadata_path": str(metadata_path),
            "trim_silence": bool(trim_silence),
            "normalize_loudness": bool(normalize_loudness),
            "target_lufs": float(target_lufs) if normalize_loudness else None,
            "highpass_hz": None if highpass_hz is None else float(highpass_hz),
            "denoise_applied": False,
            "dereverb_applied": False,
            "source_preserved": True,
            "before": before,
            "after": after,
            "warnings": [
                "未执行降噪、去混响或削波修复；这些处理可能改变音色，存在背景声时应更换原素材。"
            ],
        }
        temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
        temporary_metadata.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary_metadata.replace(metadata_path)
        return report
    except Exception:
        target.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        raise
    finally:
        intermediate.unlink(missing_ok=True)


def _run(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
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
    if process.returncode != 0:
        raise WorkerProtocolError(f"FFmpeg 母带处理失败：{process.stderr[-1600:].strip()}")
    return process


def _find_ffmpeg() -> str:
    configured = os.environ.get("FIREREDAUDIO_FFMPEG", "").strip()
    if configured and Path(configured).is_file():
        return configured
    located = shutil.which("ffmpeg")
    if located:
        return located
    raise WorkerProtocolError("母带处理需要 FFmpeg")
