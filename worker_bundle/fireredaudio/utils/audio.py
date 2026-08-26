"""Audio loading with a portable PCM WAV fast path."""

import wave
from pathlib import Path

import torch
import torchaudio

# The audio encoder consumes 16 kHz mel; RedAE operates at 24 kHz.
UNDERSTAND_SAMPLE_RATE = 16000
GENERATION_SAMPLE_RATE = 24000


def read_audio(path: str, target_sample_rate: int) -> torch.Tensor:
    """Load as a 1-D mono waveform resampled to `target_sample_rate`."""
    audio, ori_sr = _load_audio(path)
    audio = audio.mean(dim=0) if audio.shape[0] > 1 else audio[0]
    if ori_sr != target_sample_rate:
        audio = torchaudio.functional.resample(audio, ori_sr, target_sample_rate)
    return audio


def _load_audio(path: str) -> tuple[torch.Tensor, int]:
    """Read PCM16 WAV without TorchCodec/FFmpeg, otherwise use Torchaudio."""
    if Path(path).suffix.lower() == ".wav":
        try:
            with wave.open(path, "rb") as reader:
                if reader.getcomptype() != "NONE" or reader.getsampwidth() != 2:
                    raise ValueError("not PCM16")
                channels = reader.getnchannels()
                sample_rate = reader.getframerate()
                frames = reader.readframes(reader.getnframes())
            tensor = torch.frombuffer(bytearray(frames), dtype=torch.int16).clone()
            tensor = tensor.reshape(-1, channels).transpose(0, 1).float() / 32768.0
            return tensor, sample_rate
        except (wave.Error, EOFError, ValueError):
            pass
    return torchaudio.load(path)  # type: ignore[no-any-return]


def write_pcm16_wav(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    """Write a portable PCM16 WAV without TorchCodec."""
    audio = waveform.detach().to(device="cpu", dtype=torch.float32)
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio.squeeze(0)
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError("waveform must have shape [T], [C,T], or singleton-batched variants")
    pcm = (audio.clamp(-1, 1) * 32767.0).round().to(torch.int16)
    payload = pcm.transpose(0, 1).contiguous().numpy().tobytes()
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(int(pcm.shape[0]))
        writer.setsampwidth(2)
        writer.setframerate(int(sample_rate))
        writer.writeframes(payload)
