from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path


def prepare_audio_path(value: str | Path) -> str:
    """Return a WAV path; compressed inputs are decoded with bundled FFmpeg."""
    source = Path(value).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"音频文件不存在：{source}")
    if source.suffix.lower() == ".wav" and _is_pcm16_wav(source):
        return str(source)
    ffmpeg = _find_ffmpeg()
    stat = source.stat()
    key = hashlib.sha256(
        f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:24]
    target_root = Path(tempfile.gettempdir()) / "fireredaudio-t8" / "decoded"
    target_root.mkdir(parents=True, exist_ok=True)
    target = target_root / f"{key}.wav"
    if target.is_file() and target.stat().st_size > 44:
        return str(target)
    completed = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(target),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )
    if completed.returncode != 0 or not target.is_file():
        target.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg 解码失败：{completed.stderr.strip()}")
    return str(target)


def _is_pcm16_wav(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as reader:
            return reader.getcomptype() == "NONE" and reader.getsampwidth() == 2
    except (wave.Error, EOFError, OSError):
        return False


def _find_ffmpeg() -> str:
    configured = os.environ.get("FIREREDAUDIO_FFMPEG", "").strip()
    if configured and Path(configured).is_file():
        return str(Path(configured).resolve())
    located = shutil.which("ffmpeg")
    if located:
        return located
    raise RuntimeError(
        "输入是压缩音频，但未找到 FFmpeg。便携桌面包应内置 ffmpeg.exe；"
        "开发环境可设置 FIREREDAUDIO_FFMPEG。"
    )
