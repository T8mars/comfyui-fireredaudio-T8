from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

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
