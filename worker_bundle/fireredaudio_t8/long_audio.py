from __future__ import annotations

import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from .errors import WorkerProtocolError


@dataclass(frozen=True)
class AudioSegment:
    index: int
    path: str
    start_seconds: float
    end_seconds: float

    def public_dict(self, text: str) -> dict[str, object]:
        data = asdict(self)
        data.pop("path")
        data["text"] = text
        return data


def split_pcm16_wav(
    source: str | Path,
    target_dir: str | Path,
    *,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 1.0,
) -> tuple[list[AudioSegment], float]:
    if not 5.0 <= chunk_seconds <= 300.0:
        raise WorkerProtocolError("chunk_seconds 必须在 5 到 300 秒之间")
    if not 0.0 <= overlap_seconds < min(chunk_seconds, 10.0):
        raise WorkerProtocolError("overlap_seconds 必须不小于 0，且小于分段长度与 10 秒")

    source_path = Path(source).resolve()
    output_root = Path(target_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source_path), "rb") as reader:
        if reader.getcomptype() != "NONE" or reader.getsampwidth() != 2:
            raise WorkerProtocolError("长音频分段要求 PCM16 WAV 输入")
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        total_frames = reader.getnframes()
        chunk_frames = max(1, int(round(chunk_seconds * sample_rate)))
        overlap_frames = int(round(overlap_seconds * sample_rate))
        stride_frames = chunk_frames - overlap_frames
        segments: list[AudioSegment] = []
        start = 0
        index = 1
        while start < total_frames:
            end = min(total_frames, start + chunk_frames)
            reader.setpos(start)
            payload = reader.readframes(end - start)
            target = output_root / f"segment-{index:04d}.wav"
            with wave.open(str(target), "wb") as writer:
                writer.setnchannels(channels)
                writer.setsampwidth(sample_width)
                writer.setframerate(sample_rate)
                writer.writeframes(payload)
            segments.append(
                AudioSegment(
                    index=index,
                    path=str(target),
                    start_seconds=start / sample_rate,
                    end_seconds=end / sample_rate,
                )
            )
            if end >= total_frames:
                break
            start += stride_frames
            index += 1
    return segments, total_frames / sample_rate


def render_srt(segments: list[dict[str, object]]) -> str:
    blocks = []
    for index, segment in enumerate(segments, 1):
        blocks.append(
            f"{index}\n{_srt_time(float(segment['start_seconds']))} --> "
            f"{_srt_time(float(segment['end_seconds']))}\n{segment['text']}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
