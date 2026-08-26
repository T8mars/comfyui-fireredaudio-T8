from __future__ import annotations

import audioop
import json
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
    silence_search_seconds: float = 0.0,
) -> tuple[list[AudioSegment], float]:
    if not 5.0 <= chunk_seconds <= 300.0:
        raise WorkerProtocolError("chunk_seconds 必须在 5 到 300 秒之间")
    if not 0.0 <= overlap_seconds < min(chunk_seconds, 10.0):
        raise WorkerProtocolError("overlap_seconds 必须不小于 0，且小于分段长度与 10 秒")
    if not 0.0 <= silence_search_seconds <= 5.0:
        raise WorkerProtocolError("silence_search_seconds 必须在 0 到 5 秒之间")

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
            if end < total_frames and silence_search_seconds > 0:
                end = _quiet_boundary(
                    reader,
                    nominal=end,
                    minimum=start + max(1, int(5.0 * sample_rate)),
                    total_frames=total_frames,
                    sample_rate=sample_rate,
                    channels=channels,
                    sample_width=sample_width,
                    search_frames=int(round(silence_search_seconds * sample_rate)),
                )
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


def _quiet_boundary(
    reader: wave.Wave_read,
    *,
    nominal: int,
    minimum: int,
    total_frames: int,
    sample_rate: int,
    channels: int,
    sample_width: int,
    search_frames: int,
) -> int:
    lower = max(minimum, nominal - search_frames)
    upper = min(total_frames, nominal + search_frames)
    window_frames = max(1, int(round(sample_rate * 0.04)))
    best_frame = nominal
    best_rms: int | None = None
    frame = lower
    while frame < upper:
        reader.setpos(frame)
        payload = reader.readframes(min(window_frames, upper - frame))
        if not payload:
            break
        rms = audioop.rms(payload, sample_width)
        if best_rms is None or rms < best_rms:
            best_rms = rms
            best_frame = frame + max(1, len(payload) // (channels * sample_width * 2))
        frame += window_frames
    return max(minimum, min(total_frames, best_frame))


def deduplicate_segment_texts(
    segments: list[dict[str, object]], max_overlap_chars: int = 120
) -> tuple[str, list[dict[str, object]]]:
    """Remove exact suffix/prefix duplication introduced by overlapping windows."""
    cleaned: list[dict[str, object]] = []
    history = ""
    parts: list[str] = []
    for raw in segments:
        segment = dict(raw)
        text = str(segment.get("text") or "").strip()
        overlap = 0
        if history and text:
            limit = min(max_overlap_chars, len(history), len(text))
            for size in range(limit, 1, -1):
                if history[-size:] == text[:size]:
                    overlap = size
                    break
        segment["deduplicated_prefix_chars"] = overlap
        segment["text"] = text[overlap:].lstrip()
        cleaned_text = str(segment["text"])
        history += cleaned_text
        if cleaned_text:
            parts.append(cleaned_text)
        cleaned.append(segment)
    return "\n".join(parts), cleaned


def render_srt(segments: list[dict[str, object]]) -> str:
    blocks = []
    for index, segment in enumerate(segments, 1):
        blocks.append(
            f"{index}\n{_srt_time(float(segment['start_seconds']))} --> "
            f"{_srt_time(float(segment['end_seconds']))}\n{segment['text']}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(segments: list[dict[str, object]]) -> str:
    blocks = ["WEBVTT"]
    for segment in segments:
        blocks.append(
            f"{_vtt_time(float(segment['start_seconds']))} --> "
            f"{_vtt_time(float(segment['end_seconds']))}\n{segment['text']}"
        )
    return "\n\n".join(blocks) + "\n"


def render_jsonl(segments: list[dict[str, object]]) -> str:
    return "".join(json.dumps(segment, ensure_ascii=False) + "\n" for segment in segments)


def _srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _vtt_time(seconds: float) -> str:
    return _srt_time(seconds).replace(",", ".")
