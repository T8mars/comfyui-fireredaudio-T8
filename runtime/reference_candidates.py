from __future__ import annotations

import hashlib
import math
import wave
from array import array
from pathlib import Path
from typing import Any


def discover_reference_candidates(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    min_seconds: float = 3.0,
    preferred_seconds: float = 8.0,
    max_seconds: float = 15.0,
    padding_seconds: float = 0.2,
    max_candidates: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Find and rank speech-like reference regions without modifying the source.

    The detector deliberately uses inspectable signal heuristics instead of claiming
    speaker separation, denoising, or perceptual quality prediction.  Candidate WAVs
    preserve the source sample rate and channel count.
    """
    source = Path(source_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"参考录音不存在：{source}")
    minimum = float(min_seconds)
    preferred = float(preferred_seconds)
    maximum = float(max_seconds)
    if not 1.0 <= minimum <= preferred <= maximum <= 30.0:
        raise ValueError("候选时长必须满足 1 ≤ 最短 ≤ 推荐 ≤ 最长 ≤ 30 秒")
    limit = max(1, min(20, int(max_candidates)))
    padding = max(0.0, min(2.0, float(padding_seconds)))

    source_sha256 = _file_digest(source)
    with wave.open(str(source), "rb") as reader:
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        sample_width = reader.getsampwidth()
        frame_count = reader.getnframes()
        compression = reader.getcomptype()
        payload = reader.readframes(frame_count)
    if sample_width != 2 or compression != "NONE":
        raise ValueError("参考候选筛选要求 PCM16 WAV；请先通过 ComfyUI AUDIO 输入转换")
    if channels < 1 or sample_rate < 1 or frame_count < 1:
        raise ValueError("参考录音没有可分析的音频帧")
    duration = frame_count / sample_rate
    if duration < minimum:
        raise ValueError(f"参考录音仅 {duration:.3f} 秒，短于候选最短时长 {minimum:.3f} 秒")

    samples = array("h")
    samples.frombytes(payload)
    envelope = _energy_envelope(samples, channels, sample_rate, frame_count)
    finite_levels = [level for level in envelope if level > -119.0]
    if finite_levels:
        ordered = sorted(finite_levels)
        noise_floor = ordered[min(len(ordered) - 1, int(len(ordered) * 0.2))]
    else:
        noise_floor = -120.0
    speech_threshold = min(-24.0, max(-48.0, noise_floor + 10.0))
    active = [level >= speech_threshold for level in envelope]
    regions = _active_regions(active, hop_seconds=0.02, merge_gap_seconds=0.35)
    proposed = _candidate_regions(
        regions,
        duration=duration,
        minimum=minimum,
        preferred=preferred,
        maximum=maximum,
        padding=padding,
    )
    detection_fallback = False
    if not proposed:
        detection_fallback = True
        proposed = _sliding_regions(duration, minimum, preferred)

    scored: list[dict[str, Any]] = []
    for start_seconds, end_seconds in proposed:
        start_frame = max(0, min(frame_count - 1, round(start_seconds * sample_rate)))
        end_frame = max(start_frame + 1, min(frame_count, round(end_seconds * sample_rate)))
        metrics = _region_metrics(
            samples,
            channels=channels,
            sample_rate=sample_rate,
            start_frame=start_frame,
            end_frame=end_frame,
            preferred_seconds=preferred,
        )
        scored.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_seconds": round(start_frame / sample_rate, 6),
                "end_seconds": round(end_frame / sample_rate, 6),
                "duration_seconds": round((end_frame - start_frame) / sample_rate, 6),
                "signal_score": metrics.pop("signal_score"),
                "metrics": metrics,
            }
        )
    scored.sort(key=lambda item: (-float(item["signal_score"]), float(item["start_seconds"])))
    selected = _deduplicate_ranked(scored, limit)
    if not selected:
        raise ValueError("没有找到可导出的参考候选片段")

    destination.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for rank, candidate in enumerate(selected, 1):
        target = destination / f"reference-candidate-{rank:02d}.wav"
        start_sample = int(candidate["start_frame"]) * channels
        end_sample = int(candidate["end_frame"]) * channels
        segment = array("h", samples[start_sample:end_sample])
        with wave.open(str(target), "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(segment.tobytes())
        item = {
            "line_id": f"reference-{rank:02d}",
            "index": rank,
            "speaker": "reference",
            "status": "complete",
            "output_path": str(target),
            "output_sha256": _file_digest(target),
            **candidate,
        }
        items.append(item)

    source_after = _file_digest(source)
    report = {
        "kind": "reference_candidate_ranking",
        "source_path": str(source),
        "source_sha256_before": source_sha256,
        "source_sha256_after": source_after,
        "source_preserved": source_after == source_sha256,
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": round(duration, 6),
        "detector": {
            "method": "adaptive_energy_v1",
            "noise_floor_dbfs_proxy": round(noise_floor, 3),
            "speech_threshold_dbfs_proxy": round(speech_threshold, 3),
            "detection_fallback": detection_fallback,
            "claims_speaker_separation": False,
            "claims_denoising": False,
        },
        "settings": {
            "min_seconds": minimum,
            "preferred_seconds": preferred,
            "max_seconds": maximum,
            "padding_seconds": padding,
            "max_candidates": limit,
        },
        "candidate_count": len(items),
        "ranking": [
            {
                "line_id": item["line_id"],
                "start_seconds": item["start_seconds"],
                "end_seconds": item["end_seconds"],
                "signal_score": item["signal_score"],
                "metrics": item["metrics"],
            }
            for item in items
        ],
        "ranking_notice": "分数只用于候选排序；建 VoiceProfile 前必须人工试听，并核对逐字稿。",
    }
    return items, report


def asr_intelligibility_proxy(transcript: str, duration_seconds: float, language: str) -> dict[str, Any]:
    """Return a labeled ASR readability proxy; this is not WER/CER accuracy."""
    clean = "".join(str(transcript or "").split())
    if language == "en":
        tokens = [token for token in str(transcript or "").lower().split() if token]
        rate = len(tokens) / max(float(duration_seconds), 0.001)
        low, ideal_low, ideal_high, high = 0.25, 1.0, 4.5, 7.0
    else:
        tokens = list(clean)
        rate = len(tokens) / max(float(duration_seconds), 0.001)
        low, ideal_low, ideal_high, high = 0.5, 1.5, 7.0, 11.0
    if not tokens:
        return {
            "score": 0.0,
            "transcript": "",
            "units": 0,
            "units_per_second": 0.0,
            "unique_ratio": 0.0,
            "notice": "ASR 未返回文字；该分数不是准确率。",
        }
    if ideal_low <= rate <= ideal_high:
        rate_score = 1.0
    elif rate < ideal_low:
        rate_score = max(0.0, (rate - low) / max(ideal_low - low, 0.001))
    else:
        rate_score = max(0.0, (high - rate) / max(high - ideal_high, 0.001))
    unique_ratio = len(set(tokens)) / len(tokens)
    diversity_score = min(1.0, unique_ratio / 0.35)
    score = 100.0 * (0.7 * rate_score + 0.3 * diversity_score)
    return {
        "score": round(score, 3),
        "transcript": str(transcript or "").strip(),
        "units": len(tokens),
        "units_per_second": round(rate, 3),
        "unique_ratio": round(unique_ratio, 3),
        "notice": "这是基于 ASR 非空、语速和重复度的可懂度代理，不是有真值逐字稿的 WER/CER。",
    }


def _energy_envelope(
    samples: array,
    channels: int,
    sample_rate: int,
    frame_count: int,
    *,
    hop_seconds: float = 0.02,
) -> list[float]:
    hop_frames = max(1, round(sample_rate * hop_seconds))
    decimation = max(1, sample_rate // 8000)
    levels: list[float] = []
    for start_frame in range(0, frame_count, hop_frames):
        end_frame = min(frame_count, start_frame + hop_frames)
        square_sum = 0.0
        count = 0
        for frame in range(start_frame, end_frame, decimation):
            base = frame * channels
            for channel in range(channels):
                value = int(samples[base + channel])
                square_sum += value * value
                count += 1
        rms = math.sqrt(square_sum / max(1, count))
        levels.append(_dbfs(rms))
    return levels


def _active_regions(
    active: list[bool], *, hop_seconds: float, merge_gap_seconds: float
) -> list[tuple[float, float]]:
    raw: list[tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate(active + [False]):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            raw.append((start, index))
            start = None
    if not raw:
        return []
    merged: list[tuple[int, int]] = [raw[0]]
    max_gap = max(0, round(merge_gap_seconds / hop_seconds))
    for region_start, region_end in raw[1:]:
        last_start, last_end = merged[-1]
        if region_start - last_end <= max_gap:
            merged[-1] = (last_start, region_end)
        else:
            merged.append((region_start, region_end))
    return [(start_index * hop_seconds, end_index * hop_seconds) for start_index, end_index in merged]


def _candidate_regions(
    regions: list[tuple[float, float]],
    *,
    duration: float,
    minimum: float,
    preferred: float,
    maximum: float,
    padding: float,
) -> list[tuple[float, float]]:
    proposed: list[tuple[float, float]] = []
    for speech_start, speech_end in regions:
        start = max(0.0, speech_start - padding)
        end = min(duration, speech_end + padding)
        span = end - start
        if span < minimum:
            center = (start + end) / 2.0
            start = max(0.0, center - minimum / 2.0)
            end = min(duration, start + minimum)
            start = max(0.0, end - minimum)
            span = end - start
        if span <= maximum:
            if span >= minimum:
                proposed.append((start, end))
            continue
        window = min(maximum, preferred)
        step = max(minimum / 2.0, window * 0.6)
        cursor = start
        while cursor < end:
            window_end = min(end, cursor + window)
            window_start = max(start, window_end - window)
            if window_end - window_start >= minimum:
                proposed.append((window_start, window_end))
            if window_end >= end:
                break
            cursor += step
    return proposed


def _sliding_regions(duration: float, minimum: float, preferred: float) -> list[tuple[float, float]]:
    window = min(duration, preferred)
    if window < minimum:
        return []
    step = max(minimum / 2.0, window / 2.0)
    regions: list[tuple[float, float]] = []
    cursor = 0.0
    while cursor < duration:
        end = min(duration, cursor + window)
        start = max(0.0, end - window)
        regions.append((start, end))
        if end >= duration:
            break
        cursor += step
    return regions


def _region_metrics(
    samples: array,
    *,
    channels: int,
    sample_rate: int,
    start_frame: int,
    end_frame: int,
    preferred_seconds: float,
) -> dict[str, Any]:
    start_sample = start_frame * channels
    end_sample = end_frame * channels
    region = samples[start_sample:end_sample]
    count = max(1, len(region))
    peak = max((abs(int(value)) for value in region), default=0)
    square_sum = sum(int(value) * int(value) for value in region)
    rms = math.sqrt(square_sum / count)
    clipping_ratio = sum(1 for value in region if abs(int(value)) >= 32760) / count
    silence_threshold = int(32767 * 10 ** (-50.0 / 20.0))
    silence_ratio = sum(1 for value in region if abs(int(value)) <= silence_threshold) / count
    duration = (end_frame - start_frame) / sample_rate
    local_envelope = _energy_envelope(region, channels, sample_rate, end_frame - start_frame)
    ordered = sorted(local_envelope)
    low = ordered[min(len(ordered) - 1, int(len(ordered) * 0.2))] if ordered else -120.0
    high = ordered[min(len(ordered) - 1, int(len(ordered) * 0.8))] if ordered else -120.0
    contrast = max(0.0, high - low)
    activity_ratio = (
        sum(1 for level in local_envelope if level >= -42.0) / max(1, len(local_envelope))
    )
    duration_score = max(0.0, 1.0 - abs(duration - preferred_seconds) / max(preferred_seconds, 0.001))
    contrast_score = max(0.0, min(1.0, (contrast - 4.0) / 20.0))
    rms_dbfs = _dbfs(rms)
    level_score = max(0.0, 1.0 - abs(rms_dbfs + 20.0) / 22.0)
    activity_score = max(0.0, min(1.0, activity_ratio / 0.8))
    clean_score = max(0.0, 1.0 - silence_ratio) * max(0.0, 1.0 - clipping_ratio * 500.0)
    score = 100.0 * (
        0.20 * duration_score
        + 0.22 * contrast_score
        + 0.23 * level_score
        + 0.20 * activity_score
        + 0.15 * clean_score
    )
    return {
        "signal_score": round(max(0.0, min(100.0, score)), 3),
        "peak_dbfs": round(_dbfs(float(peak)), 3),
        "rms_dbfs": round(rms_dbfs, 3),
        "clipping_ratio": round(clipping_ratio, 8),
        "silence_ratio": round(silence_ratio, 6),
        "activity_ratio": round(activity_ratio, 6),
        "energy_contrast_db_proxy": round(contrast, 3),
    }


def _deduplicate_ranked(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        start = float(candidate["start_seconds"])
        end = float(candidate["end_seconds"])
        duplicate = False
        for existing in selected:
            overlap = max(0.0, min(end, float(existing["end_seconds"])) - max(start, float(existing["start_seconds"])))
            union = max(end, float(existing["end_seconds"])) - min(start, float(existing["start_seconds"]))
            if union > 0 and overlap / union >= 0.7:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
        if len(selected) >= limit:
            break
    return selected


def _dbfs(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return 20.0 * math.log10(value / 32767.0)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
