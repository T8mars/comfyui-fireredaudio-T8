from __future__ import annotations

import hashlib
import tempfile
import wave
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


def _torch():
    import torch

    return torch
