from __future__ import annotations

import json
import re
import wave
from pathlib import Path
from typing import Any


TIME_KEYS_START = ("start_seconds", "start_time", "start", "begin", "from", "position", "time")
TIME_KEYS_END = ("end_seconds", "end_time", "end", "finish", "to")
LABEL_KEYS = ("title", "topic", "label", "match", "transcript_or_event", "summary", "text", "event")
LIST_KEYS = ("timeline", "items", "segments", "matches", "chapters", "events", "results", "evidence")


def parse_structured_json(value: str | dict | list) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    candidates = [text]
    if "```" in text:
        for block in text.split("```"):
            candidate = block.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].lstrip()
            if candidate.startswith(("{", "[")):
                candidates.append(candidate)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (TypeError, ValueError):
            continue
    raise ValueError("定位结果不是有效 JSON；请连接长音频时间定位节点的结构化 JSON 输出")


def extract_evidence_ranges(
    structured: Any,
    *,
    default_clip_seconds: float = 8.0,
    max_clips: int = 20,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_objects: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict) or id(value) in seen_objects:
            return
        seen_objects.add(id(value))
        start_raw = _first(value, TIME_KEYS_START)
        if start_raw is not None:
            start = parse_time_seconds(start_raw)
            end_raw = _first(value, TIME_KEYS_END)
            end = parse_time_seconds(end_raw) if end_raw is not None else start + default_clip_seconds
            if end > start:
                label = str(_first(value, LABEL_KEYS) or f"证据 {len(candidates) + 1}").strip()
                candidates.append(
                    {
                        "start_seconds": round(start, 3),
                        "end_seconds": round(end, 3),
                        "label": label[:500],
                        "source": value,
                    }
                )
        for key in LIST_KEYS:
            if key in value:
                visit(value[key])

    visit(structured)
    unique: list[dict[str, Any]] = []
    seen_ranges: set[tuple[int, int, str]] = set()
    for item in sorted(candidates, key=lambda entry: (entry["start_seconds"], entry["end_seconds"])):
        identity = (
            round(item["start_seconds"] * 1000),
            round(item["end_seconds"] * 1000),
            item["label"],
        )
        if identity in seen_ranges:
            continue
        seen_ranges.add(identity)
        unique.append(item)
        if len(unique) >= max(1, int(max_clips)):
            break
    return unique


def render_evidence_clips(
    source_path: str | Path,
    ranges: list[dict[str, Any]],
    output_directory: str | Path,
    *,
    filename_prefix: str = "evidence",
    padding_seconds: float = 0.25,
) -> list[dict[str, Any]]:
    source = Path(source_path).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as reader:
        if reader.getsampwidth() != 2:
            raise ValueError("证据片段当前只支持 PCM16 WAV")
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        total_frames = reader.getnframes()
        frames = reader.readframes(total_frames)
    frame_bytes = channels * 2
    duration = total_frames / sample_rate
    prefix = _safe_name(filename_prefix, "evidence")
    results: list[dict[str, Any]] = []
    for index, item in enumerate(ranges, 1):
        requested_start = float(item["start_seconds"])
        requested_end = float(item["end_seconds"])
        start = max(0.0, requested_start - max(0.0, float(padding_seconds)))
        end = min(duration, requested_end + max(0.0, float(padding_seconds)))
        if start >= duration or end <= start:
            raise ValueError(
                f"证据片段 {index} 的时间范围超出源音频："
                f"{requested_start:.3f}–{requested_end:.3f}s / {duration:.3f}s"
            )
        start_frame = min(total_frames, max(0, round(start * sample_rate)))
        end_frame = min(total_frames, round(end * sample_rate))
        if end_frame <= start_frame:
            raise ValueError(f"证据片段 {index} 裁切后没有有效采样")
        label = _safe_name(str(item.get("label") or f"clip-{index}"), f"clip-{index}")
        target = output / f"{prefix}-{index:03d}-{label}.wav"
        with wave.open(str(target), "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(frames[start_frame * frame_bytes : end_frame * frame_bytes])
        results.append(
            {
                "line_id": f"evidence-{index:03d}",
                "index": index,
                "status": "complete",
                "output_path": str(target),
                "label": str(item.get("label") or f"证据 {index}"),
                "requested_start_seconds": requested_start,
                "requested_end_seconds": requested_end,
                "start_seconds": round(start_frame / sample_rate, 3),
                "end_seconds": round(end_frame / sample_rate, 3),
                "duration_seconds": round((end_frame - start_frame) / sample_rate, 3),
                "sample_rate": sample_rate,
                "channels": channels,
                "source": item.get("source"),
            }
        )
    return results


def parse_time_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value or "").strip().replace(",", ".")
    if not text:
        raise ValueError("时间值为空")
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return max(0.0, float(text))
    parts = text.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"无法解析时间：{value}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"无法解析时间：{value}") from exc
    if len(numbers) == 2:
        minutes, seconds = numbers
        return max(0.0, minutes * 60 + seconds)
    hours, minutes, seconds = numbers
    return max(0.0, hours * 3600 + minutes * 60 + seconds)


def _first(value: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in value and value[key] not in (None, ""):
            return value[key]
    return None


def _safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip(".-_")
    return cleaned[:60] or fallback
