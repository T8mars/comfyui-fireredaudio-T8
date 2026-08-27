from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
import wave
from datetime import datetime
from pathlib import Path


def temp_root() -> Path:
    try:
        import folder_paths

        root = Path(folder_paths.get_temp_directory()) / "fireredaudio-t8"
    except Exception:
        root = Path(tempfile.gettempdir()) / "fireredaudio-t8"
    root.mkdir(parents=True, exist_ok=True)
    return root


def audio_to_wav(audio: dict, label: str = "audio") -> Path:
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise TypeError(f"{label} 必须是 ComfyUI AUDIO")
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if getattr(waveform, "ndim", 0) != 3:
        raise ValueError(f"{label}.waveform 必须是 [B,C,T]")
    if waveform.shape[0] != 1:
        raise ValueError(f"{label} 当前只支持 batch=1")
    pcm = waveform[0].detach().to(device="cpu", dtype=_torch().float32).clamp(-1, 1)
    pcm = (pcm * 32767.0).round().to(_torch().int16).transpose(0, 1).contiguous()
    payload = pcm.numpy().tobytes()
    digest = hashlib.sha256(payload + str(sample_rate).encode()).hexdigest()[:20]
    target = temp_root() / f"{label}-{digest}.wav"
    if not target.exists():
        with wave.open(str(target), "wb") as writer:
            writer.setnchannels(int(pcm.shape[1]))
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(payload)
    return target


def wav_to_audio(path: str | Path) -> dict:
    target = Path(path)
    with wave.open(str(target), "rb") as reader:
        if reader.getsampwidth() != 2:
            raise ValueError("Worker 输出必须是 PCM16 WAV")
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
    tensor = _torch().frombuffer(bytearray(frames), dtype=_torch().int16).clone()
    tensor = tensor.reshape(-1, channels).transpose(0, 1).float() / 32768.0
    return {"waveform": tensor.unsqueeze(0), "sample_rate": sample_rate}


def output_wav_path(task: str) -> Path:
    import uuid

    return temp_root() / f"{task}-{uuid.uuid4().hex}.wav"


def save_audio_file(
    audio: dict,
    *,
    filename_prefix: str = "fireredaudio",
    subfolder: str = "fireredaudio",
    audio_format: str = "wav",
) -> Path:
    source = audio_to_wav(audio, "save")
    target_dir = _safe_output_dir(subfolder)
    prefix = _safe_name(filename_prefix, "fireredaudio")
    extension = str(audio_format).lower()
    if extension not in {"wav", "flac", "mp3", "ogg"}:
        raise ValueError("audio_format 必须是 wav/flac/mp3/ogg")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    target = target_dir / f"{prefix}-{stamp}.{extension}"
    if extension == "wav":
        shutil.copy2(source, target)
        return target
    ffmpeg = _ffmpeg_path()
    codecs = {"flac": "flac", "mp3": "libmp3lame", "ogg": "libvorbis"}
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
            "-codec:a",
            codecs[extension],
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
        raise RuntimeError(f"FFmpeg 导出失败：{completed.stderr.strip()}")
    return target


def export_audio_path(
    source_path: str | Path,
    target_path: str | Path,
    *,
    audio_format: str = "wav",
) -> Path:
    """Copy or transcode an existing audio file to an exact output target."""
    source = Path(source_path).resolve()
    target = Path(target_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"待导出音频不存在：{source}")
    extension = str(audio_format).lower()
    if extension not in {"wav", "flac", "mp3", "ogg"}:
        raise ValueError("audio_format 必须是 wav/flac/mp3/ogg")
    if target.suffix.lower() != f".{extension}":
        raise ValueError("导出目标扩展名与 audio_format 不一致")
    target.parent.mkdir(parents=True, exist_ok=True)
    if source == target:
        return target
    if extension == "wav" and source.suffix.lower() == ".wav":
        shutil.copy2(source, target)
        return target
    ffmpeg = _ffmpeg_path()
    codecs = {"wav": "pcm_s16le", "flac": "flac", "mp3": "libmp3lame", "ogg": "libvorbis"}
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
            "-codec:a",
            codecs[extension],
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
        raise RuntimeError(f"FFmpeg 导出失败：{completed.stderr.strip()}")
    return target


def save_text_file(
    content: str,
    *,
    filename_prefix: str = "fireredaudio",
    subfolder: str = "fireredaudio",
    text_format: str = "srt",
) -> Path:
    extension = str(text_format).lower()
    if extension not in {"srt", "vtt", "txt", "jsonl"}:
        raise ValueError("text_format 必须是 srt/vtt/txt/jsonl")
    target_dir = _safe_output_dir(subfolder)
    prefix = _safe_name(filename_prefix, "fireredaudio")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    target = target_dir / f"{prefix}-{stamp}.{extension}"
    target.write_text(str(content), encoding="utf-8")
    return target


def saved_audio_ui(path: str | Path) -> dict:
    """Return ComfyUI's native saved-audio descriptor for an existing output file."""
    target = Path(path).resolve()
    root = output_root()
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ValueError("只有 ComfyUI output 目录内的音频可以注册为下载资产") from exc
    subfolder = relative.parent.as_posix()
    if subfolder == ".":
        subfolder = ""
    return {
        "audio": [
            {
                "filename": relative.name,
                "subfolder": subfolder,
                "type": "output",
            }
        ]
    }


def saved_audio_files_ui(paths: list[str | Path]) -> dict:
    """Return one native ComfyUI audio/download list for multiple output assets."""
    descriptors: list[dict[str, str]] = []
    for path in paths:
        descriptors.extend(saved_audio_ui(path)["audio"])
    return {"audio": descriptors}


def _safe_output_dir(subfolder: str) -> Path:
    root = output_root()
    relative = Path(str(subfolder or "fireredaudio").replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("输出子目录不能使用绝对路径或 ..")
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("输出目录越界") from exc
    target.mkdir(parents=True, exist_ok=True)
    return target


def output_root() -> Path:
    try:
        import folder_paths

        root = Path(folder_paths.get_output_directory()).resolve()
    except Exception:
        root = (Path.cwd() / "output").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", str(value)).strip(".-_")
    return cleaned[:80] or fallback


def _ffmpeg_path() -> str:
    bundled = Path(__file__).resolve().parents[1] / "tools" / "ffmpeg.exe"
    if bundled.is_file():
        return str(bundled)
    located = shutil.which("ffmpeg")
    if located:
        return located
    raise RuntimeError("未找到 FFmpeg，无法导出压缩音频格式")


def _torch():
    import torch

    return torch
