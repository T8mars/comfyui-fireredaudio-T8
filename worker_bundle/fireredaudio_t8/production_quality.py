from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

from .audio_quality import analyze_audio
from .errors import WorkerProtocolError


INTEGRATED_RE = re.compile(r"^\s*I:\s*(-?(?:\d+(?:\.\d+)?|inf))\s+LUFS\s*$", re.MULTILINE)
LRA_RE = re.compile(r"^\s*LRA:\s*(-?(?:\d+(?:\.\d+)?|inf))\s+LU\s*$", re.MULTILINE)
TRUE_PEAK_RE = re.compile(r"^\s*Peak:\s*(-?(?:\d+(?:\.\d+)?|inf))\s+dBFS\s*$", re.MULTILINE)


def analyze_production_audio(
    path: str | Path,
    *,
    target_lufs: float = -16.0,
    tolerance_lu: float = 2.0,
    true_peak_ceiling_dbfs: float = -1.0,
) -> dict[str, Any]:
    report = analyze_audio(path)
    ffmpeg = _find_ffmpeg()
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-i",
        str(Path(path).expanduser().resolve()),
        "-filter_complex",
        "ebur128=peak=true",
        "-f",
        "null",
        "NUL" if os.name == "nt" else "/dev/null",
    ]
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=300,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    output = process.stderr
    if process.returncode != 0:
        raise WorkerProtocolError(f"FFmpeg 响度分析失败：{output[-1000:].strip()}")
    integrated = _last_float(INTEGRATED_RE, output)
    loudness_range = _last_float(LRA_RE, output)
    true_peak = _last_float(TRUE_PEAK_RE, output)
    # ``analyze_audio`` is also used as a reference-voice gate and therefore
    # reports reference-specific duration guidance.  A finished production
    # render can legitimately be longer than 30 seconds, so keep the signal
    # defects while removing only those reference-asset recommendations.
    issues = [
        issue
        for issue in list(report.get("issues") or [])
        if not str(issue).startswith("参考音频")
    ]
    if integrated is None:
        issues.append("无法读取综合响度 LUFS")
    elif integrated > target_lufs + tolerance_lu:
        issues.append(f"综合响度 {integrated:.1f} LUFS 高于目标 {target_lufs:.1f} LUFS")
    elif integrated < target_lufs - tolerance_lu:
        issues.append(f"综合响度 {integrated:.1f} LUFS 低于目标 {target_lufs:.1f} LUFS")
    if true_peak is None:
        issues.append("无法读取 True Peak")
    elif true_peak > true_peak_ceiling_dbfs:
        issues.append(
            f"True Peak {true_peak:.1f} dBFS 超过上限 {true_peak_ceiling_dbfs:.1f} dBFS"
        )
    report.update(
        {
            "integrated_lufs": integrated,
            "loudness_range_lu": loudness_range,
            "true_peak_dbfs": true_peak,
            "target_lufs": target_lufs,
            "loudness_tolerance_lu": tolerance_lu,
            "true_peak_ceiling_dbfs": true_peak_ceiling_dbfs,
            "issues": issues,
            "recommended": not issues,
        }
    )
    return report


def text_diff_metrics(reference: str, hypothesis: str, *, language: str = "zh") -> dict[str, Any]:
    reference_text = str(reference or "").strip()
    hypothesis_text = str(hypothesis or "").strip()
    if str(language).lower().startswith(("zh", "ja", "ko")):
        reference_units = [character for character in reference_text if not character.isspace()]
        hypothesis_units = [character for character in hypothesis_text if not character.isspace()]
        metric = "cer"
    else:
        reference_units = reference_text.lower().split()
        hypothesis_units = hypothesis_text.lower().split()
        metric = "wer"
    distance = _levenshtein(reference_units, hypothesis_units)
    denominator = max(1, len(reference_units))
    return {
        "metric": metric,
        "distance": distance,
        "reference_units": len(reference_units),
        "hypothesis_units": len(hypothesis_units),
        "error_rate": round(distance / denominator, 6),
        "exact_match": reference_units == hypothesis_units,
    }


def _levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    value = matches[-1]
    if value.lower() in {"inf", "-inf"}:
        return None
    return float(value)


def _find_ffmpeg() -> str:
    configured = os.environ.get("FIREREDAUDIO_FFMPEG", "").strip()
    if configured and Path(configured).is_file():
        return configured
    located = shutil.which("ffmpeg")
    if located:
        return located
    raise WorkerProtocolError("成片响度质检需要 FFmpeg")
