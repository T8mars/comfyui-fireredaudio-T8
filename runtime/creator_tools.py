from __future__ import annotations

import json
import os
import re
import subprocess
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Any

from .audio_adapter import _ffmpeg_path, _safe_name
from .production import (
    AudioBatch,
    ScriptPlan,
    file_digest,
    load_manifest,
    wav_metrics,
)

_ZH_DIGITS = "零一二三四五六七八九"
_DECISION_ALIASES = {
    "auto": "auto",
    "approve": "approve",
    "approved": "approve",
    "accept": "approve",
    "pass": "approve",
    "通过": "approve",
    "采用": "approve",
    "review": "review",
    "人工复核": "review",
    "复核": "review",
    "retry": "retry",
    "regenerate": "retry",
    "重做": "retry",
    "返修": "retry",
}


def parse_json_mapping(
    value: str | dict[str, Any] | None, label: str
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    try:
        parsed = json.loads(str(value or "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}不是有效 JSON") from exc
    if not isinstance(parsed, dict):
        raise TypeError(f"{label}必须是 JSON 对象")
    return {str(key): item for key, item in parsed.items()}


def normalize_script_plan(
    script_plan: ScriptPlan,
    *,
    replacements: dict[str, Any] | None = None,
    normalize_unicode: bool = True,
    normalize_whitespace: bool = True,
    expand_zh_dates: bool = True,
    expand_zh_numbers: bool = False,
) -> tuple[ScriptPlan, dict[str, Any]]:
    if not isinstance(script_plan, ScriptPlan):
        raise TypeError("朗读规范化必须连接脚本计划")
    dictionary: dict[str, str] = {}
    for source, target in (replacements or {}).items():
        source_text = str(source)
        if not source_text:
            raise ValueError("替换词典不能包含空键")
        dictionary[source_text] = str(target)

    normalized_lines = []
    changed: list[dict[str, Any]] = []
    for line in script_plan.lines:
        original = line.source_text or line.text
        spoken = line.text
        changes: list[str] = []
        if normalize_unicode:
            value = unicodedata.normalize("NFKC", spoken)
            if value != spoken:
                changes.append("unicode_nfkc")
            spoken = value
        if normalize_whitespace:
            value = re.sub(r"[\t\r\n ]+", " ", spoken).strip()
            value = re.sub(r"\s+([，。！？；：、,.!?;:])", r"\1", value)
            if value != spoken:
                changes.append("whitespace")
            spoken = value
        for source in sorted(dictionary, key=lambda item: (-len(item), item)):
            value = spoken.replace(source, dictionary[source])
            if value != spoken:
                changes.append(f"dictionary:{source}")
            spoken = value
        if line.language == "zh" and expand_zh_dates:
            value = _expand_zh_dates(spoken)
            if value != spoken:
                changes.append("zh_dates")
            spoken = value
        if line.language == "zh" and expand_zh_numbers:
            value = re.sub(
                r"(?<![A-Za-z])\d+(?![A-Za-z])", _replace_zh_cardinal, spoken
            )
            if value != spoken:
                changes.append("zh_numbers")
            spoken = value
        if not spoken:
            raise ValueError(f"朗读规范化后台词为空：{line.line_id}")
        unique_changes = tuple(dict.fromkeys(changes))
        normalized_lines.append(
            replace(
                line,
                text=spoken,
                source_text=original,
                normalization=unique_changes,
            )
        )
        if spoken != line.text or unique_changes:
            changed.append(
                {
                    "line_id": line.line_id,
                    "index": line.index,
                    "speaker": line.speaker,
                    "source_text": original,
                    "previous_spoken_text": line.text,
                    "spoken_text": spoken,
                    "changes": list(unique_changes),
                }
            )
    plan = ScriptPlan(
        script_plan.source_format, tuple(normalized_lines), script_plan.issues
    )
    report = {
        "total": len(plan.lines),
        "changed": len(changed),
        "unchanged": len(plan.lines) - len(changed),
        "dictionary_entries": len(dictionary),
        "options": {
            "normalize_unicode": bool(normalize_unicode),
            "normalize_whitespace": bool(normalize_whitespace),
            "expand_zh_dates": bool(expand_zh_dates),
            "expand_zh_numbers": bool(expand_zh_numbers),
        },
        "items": changed,
    }
    return plan, report


def load_audio_batch_from_manifest(
    manifest_path: str | Path,
    *,
    allowed_root: str | Path | None = None,
    missing_policy: str = "mark_missing",
    verify_hashes: bool = False,
) -> tuple[AudioBatch, dict[str, Any]]:
    if missing_policy not in {"mark_missing", "error"}:
        raise ValueError("missing_policy 必须是 mark_missing 或 error")
    target = Path(manifest_path).expanduser().resolve()
    root: Path | None = None
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("Manifest 必须位于 ComfyUI output 目录") from exc
    payload = load_manifest(target)
    if payload is None:
        raise FileNotFoundError(f"Manifest 不存在：{target}")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise TypeError("Manifest 缺少 items 数组")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    missing: list[str] = []
    hash_mismatches: list[str] = []
    for position, raw in enumerate(raw_items, 1):
        if not isinstance(raw, dict):
            raise TypeError(f"Manifest 第 {position} 个条目不是对象")
        item = dict(raw)
        line_id = str(item.get("line_id") or "").strip()
        if not line_id:
            raise ValueError(f"Manifest 第 {position} 个条目缺少 line_id")
        if line_id in seen:
            raise ValueError(f"Manifest line_id 重复：{line_id}")
        seen.add(line_id)
        raw_path = str(item.get("output_path") or "").strip()
        path = Path(raw_path)
        if raw_path and not path.is_absolute():
            path = (target.parent / path).resolve()
        elif raw_path:
            path = path.resolve()
        if raw_path and root is not None:
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Manifest 音频必须位于 ComfyUI output 目录：{line_id}"
                ) from exc
        item["output_path"] = str(path) if raw_path else ""
        if not raw_path or not path.is_file():
            missing.append(line_id)
            if missing_policy == "error":
                raise FileNotFoundError(f"Manifest 音频不存在：{line_id} -> {path}")
            item["resume_original_status"] = item.get("status")
            item["status"] = "missing"
            item["error"] = "Manifest 记录的音频文件不存在"
        elif verify_hashes:
            expected = str(
                item.get("sha256")
                or item.get("export_sha256")
                or (item.get("worker_report") or {}).get("sha256")
                or ""
            ).lower()
            if expected and file_digest(path).lower() != expected:
                hash_mismatches.append(line_id)
                raise ValueError(f"Manifest 音频哈希不一致：{line_id}")
        items.append(item)
    batch = AudioBatch(str(target), tuple(items))
    report = {
        "manifest_path": str(target),
        "kind": payload.get("kind") or "batch",
        "total": len(items),
        "playable": len(batch.successful_items()),
        "missing": missing,
        "hash_mismatches": hash_mismatches,
        "reviewed": sum(isinstance(item.get("human_review"), dict) for item in items),
    }
    return batch, report


def build_line_review(
    audio_batch: AudioBatch,
    *,
    qa: dict[str, Any] | None = None,
    decisions: dict[str, Any] | None = None,
    ratings: dict[str, Any] | None = None,
    notes: dict[str, Any] | None = None,
) -> tuple[AudioBatch, AudioBatch, dict[str, Any]]:
    if not isinstance(audio_batch, AudioBatch):
        raise TypeError("逐句审核必须连接 AudioBatch")
    decisions = decisions or {}
    ratings = ratings or {}
    notes = notes or {}
    known_ids = {str(item.get("line_id") or "") for item in audio_batch.items}
    unknown = sorted((set(decisions) | set(ratings) | set(notes)) - known_ids)
    if unknown:
        raise ValueError("审核数据包含未知 line ID：" + ", ".join(unknown))
    qa_by_id = {
        str(item.get("line_id") or ""): item
        for item in ((qa or {}).get("items") or [])
        if isinstance(item, dict) and item.get("line_id")
    }
    reviewed: list[dict[str, Any]] = []
    delivery: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    approved_ids: list[str] = []
    review_ids: list[str] = []
    retry_ids: list[str] = []
    for position, source in enumerate(audio_batch.items, 1):
        item = dict(source)
        line_id = str(item.get("line_id") or "")
        qa_item = qa_by_id.get(line_id)
        suggested, reason = _suggest_review_decision(qa_item)
        existing = (
            item.get("human_review")
            if isinstance(item.get("human_review"), dict)
            else {}
        )
        raw_decision = decisions.get(
            line_id,
            existing.get(
                "requested_decision", existing.get("effective_decision", "auto")
            ),
        )
        requested = _normalize_decision(raw_decision, line_id)
        effective = suggested if requested == "auto" else requested
        rating_value = ratings.get(line_id, existing.get("rating"))
        rating = None
        if rating_value is not None and rating_value != "":
            try:
                rating = round(float(rating_value), 2)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{line_id} 的评分必须是 1–5") from exc
            if not 1.0 <= rating <= 5.0:
                raise ValueError(f"{line_id} 的评分必须是 1–5")
        note = str(notes.get(line_id, existing.get("note", ""))).strip()
        human_review = {
            "requested_decision": requested,
            "effective_decision": effective,
            "suggested_decision": suggested,
            "suggestion_reason": reason,
            "rating": rating,
            "note": note,
            "manual_override": requested != "auto",
        }
        item["human_review"] = human_review
        reviewed.append(item)
        delivery_item = dict(item)
        if (
            effective == "approve"
            and item.get("status") == "complete"
            and Path(str(item.get("output_path") or "")).is_file()
        ):
            approved_ids.append(line_id)
        else:
            delivery_item["review_source_status"] = delivery_item.get("status")
            delivery_item["status"] = "review_hold"
            if effective == "retry":
                retry_ids.append(line_id)
            else:
                review_ids.append(line_id)
        delivery.append(delivery_item)
        rows.append(
            {
                "position": position,
                "line_id": line_id,
                "index": item.get("index", position),
                "speaker": item.get("speaker") or "",
                "text": item.get("text") or "",
                "source_text": item.get("source_text") or item.get("text") or "",
                "status": item.get("status"),
                "output_path": item.get("output_path") or "",
                "qa": qa_item,
                "review": human_review,
            }
        )
    summary = {
        "source_manifest_path": audio_batch.manifest_path,
        "total": len(reviewed),
        "approved_count": len(approved_ids),
        "review_count": len(review_ids),
        "retry_count": len(retry_ids),
        "approved_line_ids": approved_ids,
        "review_line_ids": review_ids,
        "retry_line_ids": retry_ids,
        "rows": rows,
    }
    return (
        AudioBatch(audio_batch.manifest_path, tuple(reviewed)),
        AudioBatch(audio_batch.manifest_path, tuple(delivery)),
        summary,
    )


def fit_audio_batch_to_cues(
    audio_batch: AudioBatch,
    output_dir: str | Path,
    *,
    strategy: str = "speech_aware",
    tolerance_seconds: float = 0.10,
    maximum_speed: float = 1.15,
    minimum_speed: float = 0.90,
    fit_underrun: bool = False,
    edge_silence_threshold_db: float = -40.0,
    edge_silence_min_seconds: float = 0.05,
    edge_padding_seconds: float = 0.12,
) -> tuple[AudioBatch, dict[str, Any]]:
    if not isinstance(audio_batch, AudioBatch):
        raise TypeError("字幕时长适配必须连接 AudioBatch")
    if strategy not in {"report_only", "safe_stretch", "speech_aware"}:
        raise ValueError("strategy 必须是 report_only、safe_stretch 或 speech_aware")
    if not 1.0 <= float(maximum_speed) <= 2.0:
        raise ValueError("最大加速倍率必须在 1.0–2.0")
    if not 0.5 <= float(minimum_speed) <= 1.0:
        raise ValueError("最小减速倍率必须在 0.5–1.0")
    if not -80.0 <= float(edge_silence_threshold_db) <= -20.0:
        raise ValueError("首尾静音阈值必须在 -80–-20 dB")
    if not 0.01 <= float(edge_silence_min_seconds) <= 2.0:
        raise ValueError("首尾静音最短时长必须在 0.01–2.0 秒")
    if not 0.0 <= float(edge_padding_seconds) <= 2.0:
        raise ValueError("保留首尾缓冲必须在 0–2.0 秒")
    target_root = Path(output_dir).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    fitted: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    retry_ids: list[str] = []
    adapted_ids: list[str] = []
    for position, source in enumerate(audio_batch.items, 1):
        item = dict(source)
        line_id = str(item.get("line_id") or position)
        source_path = Path(str(item.get("output_path") or ""))
        start = item.get("start_seconds")
        end = item.get("end_seconds")
        row: dict[str, Any] = {
            "line_id": line_id,
            "index": item.get("index", position),
            "speaker": item.get("speaker") or "",
        }
        if item.get("status") != "complete" or not source_path.is_file():
            row.update(action="skip_unplayable", reason="音频未完成或文件不存在")
        elif start is None or end is None:
            row.update(action="skip_untimed", reason="台词没有字幕时间槽")
        else:
            slot = float(end) - float(start)
            if slot <= 0:
                raise ValueError(f"字幕时间槽无效：{line_id}")
            metrics = wav_metrics(source_path)
            duration = float(metrics["duration_seconds"])
            required_rate = duration / slot
            overrun = max(0.0, duration - slot)
            underrun = max(0.0, slot - duration)
            row.update(
                slot_seconds=slot,
                source_duration_seconds=duration,
                overrun_seconds=overrun,
                underrun_seconds=underrun,
                required_tempo=required_rate,
            )
            should_speed = overrun > float(tolerance_seconds)
            should_slow = bool(fit_underrun) and underrun > float(tolerance_seconds)
            if not should_speed and not should_slow:
                row.update(action="within_tolerance", output_duration_seconds=duration)
            elif strategy == "report_only":
                row.update(
                    action="report_overrun" if should_speed else "report_underrun"
                )
                if should_speed:
                    retry_ids.append(line_id)
            elif should_speed and strategy == "speech_aware":
                silence = _boundary_silence_seconds(
                    source_path,
                    duration_seconds=duration,
                    threshold_db=float(edge_silence_threshold_db),
                    minimum_seconds=float(edge_silence_min_seconds),
                )
                padding = float(edge_padding_seconds)
                available_leading = max(0.0, float(silence["leading_seconds"]) - padding)
                available_trailing = max(0.0, float(silence["trailing_seconds"]) - padding)
                trim_needed = max(0.0, duration - slot)
                trim_leading = min(available_leading, trim_needed)
                trim_trailing = min(
                    available_trailing,
                    max(0.0, trim_needed - trim_leading),
                )
                trimmed_duration = max(0.0, duration - trim_leading - trim_trailing)
                residual_overrun = max(0.0, trimmed_duration - slot)
                residual_tempo = trimmed_duration / slot
                row.update(
                    boundary_silence=silence,
                    edge_padding_seconds=padding,
                    available_edge_trim_seconds=available_leading + available_trailing,
                    trim_leading_seconds=trim_leading,
                    trim_trailing_seconds=trim_trailing,
                    trim_total_seconds=trim_leading + trim_trailing,
                    estimated_post_trim_seconds=trimmed_duration,
                    residual_overrun_seconds=residual_overrun,
                    residual_tempo=residual_tempo,
                )
                if residual_overrun > float(tolerance_seconds) and residual_tempo > float(maximum_speed):
                    row.update(
                        action="regenerate",
                        reason="裁掉首尾多余静音后，所需加速仍超过安全上限",
                    )
                    retry_ids.append(line_id)
                else:
                    target = target_root / (
                        _safe_name(
                            f"{int(item.get('index') or position):04d}-{item.get('speaker') or 'take'}-{line_id}-fit",
                            f"line-{position:04d}-fit",
                        )
                        + ".wav"
                    )
                    effective_tempo = (
                        residual_tempo
                        if residual_overrun > float(tolerance_seconds)
                        else 1.0
                    )
                    _trim_and_time_stretch_wav(
                        source_path,
                        target,
                        trim_start_seconds=trim_leading,
                        trim_end_seconds=trim_trailing,
                        source_duration_seconds=duration,
                        tempo=effective_tempo,
                    )
                    output_metrics = wav_metrics(target)
                    output_duration = float(output_metrics["duration_seconds"])
                    action = (
                        "silence_trimmed_and_time_stretched"
                        if abs(effective_tempo - 1.0) > 1e-6
                        else "silence_trimmed"
                    )
                    item["duration_fit"] = {
                        "source_output_path": str(source_path.resolve()),
                        "source_sha256": file_digest(source_path),
                        "method": "speech_aware",
                        "raw_required_tempo": required_rate,
                        "tempo": effective_tempo,
                        "trim_leading_seconds": trim_leading,
                        "trim_trailing_seconds": trim_trailing,
                        "slot_seconds": slot,
                        "source_duration_seconds": duration,
                        "output_duration_seconds": output_duration,
                        "boundary_silence": silence,
                        "non_destructive": True,
                    }
                    item["output_path"] = str(target)
                    item["adapted"] = True
                    adapted_ids.append(line_id)
                    row.update(
                        action=action,
                        output_path=str(target),
                        output_duration_seconds=output_duration,
                        output_delta_seconds=output_duration - slot,
                        tempo=effective_tempo,
                    )
            elif should_speed and required_rate > float(maximum_speed):
                row.update(action="regenerate", reason="所需加速超过安全上限")
                retry_ids.append(line_id)
            elif should_slow and required_rate < float(minimum_speed):
                row.update(action="keep_short", reason="所需减速超过安全下限")
            else:
                target = target_root / (
                    _safe_name(
                        f"{int(item.get('index') or position):04d}-{item.get('speaker') or 'take'}-{line_id}-fit",
                        f"line-{position:04d}-fit",
                    )
                    + ".wav"
                )
                _time_stretch_wav(source_path, target, required_rate)
                output_metrics = wav_metrics(target)
                output_duration = float(output_metrics["duration_seconds"])
                item["duration_fit"] = {
                    "source_output_path": str(source_path.resolve()),
                    "source_sha256": file_digest(source_path),
                    "tempo": required_rate,
                    "slot_seconds": slot,
                    "source_duration_seconds": duration,
                    "output_duration_seconds": output_duration,
                    "non_destructive": True,
                }
                item["output_path"] = str(target)
                item["adapted"] = True
                adapted_ids.append(line_id)
                row.update(
                    action="time_stretched",
                    output_path=str(target),
                    output_duration_seconds=output_duration,
                    output_delta_seconds=output_duration - slot,
                    tempo=required_rate,
                )
        item["duration_fit_report"] = dict(row)
        fitted.append(item)
        reports.append(row)
    report = {
        "source_manifest_path": audio_batch.manifest_path,
        "strategy": strategy,
        "tolerance_seconds": float(tolerance_seconds),
        "maximum_speed": float(maximum_speed),
        "minimum_speed": float(minimum_speed),
        "fit_underrun": bool(fit_underrun),
        "edge_silence_threshold_db": float(edge_silence_threshold_db),
        "edge_silence_min_seconds": float(edge_silence_min_seconds),
        "edge_padding_seconds": float(edge_padding_seconds),
        "total": len(fitted),
        "adapted_count": len(adapted_ids),
        "retry_count": len(retry_ids),
        "adapted_line_ids": adapted_ids,
        "retry_line_ids": retry_ids,
        "items": reports,
        "source_files_overwritten": False,
    }
    return AudioBatch(audio_batch.manifest_path, tuple(fitted)), report


def _expand_zh_dates(value: str) -> str:
    pattern = re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")

    def replace_date(match: re.Match[str]) -> str:
        year = "".join(_ZH_DIGITS[int(character)] for character in match.group(1))
        month = _zh_cardinal(int(match.group(2)))
        day = _zh_cardinal(int(match.group(3)))
        return f"{year}年{month}月{day}日"

    return pattern.sub(replace_date, value)


def _replace_zh_cardinal(match: re.Match[str]) -> str:
    raw = match.group(0)
    if len(raw) > 8 or raw.startswith("0") and len(raw) > 1:
        return "".join(_ZH_DIGITS[int(character)] for character in raw)
    return _zh_cardinal(int(raw))


def _zh_cardinal(value: int) -> str:
    if value == 0:
        return _ZH_DIGITS[0]
    if value >= 100_000_000:
        return "".join(_ZH_DIGITS[int(character)] for character in str(value))

    def under_ten_thousand(number: int) -> str:
        units = ("", "十", "百", "千")
        digits = []
        zero_pending = False
        for position in range(3, -1, -1):
            divisor = 10**position
            digit = number // divisor % 10
            if digit:
                if zero_pending and digits:
                    digits.append("零")
                if not (digit == 1 and position == 1 and not digits):
                    digits.append(_ZH_DIGITS[digit])
                digits.append(units[position])
                zero_pending = False
            elif digits and number % divisor:
                zero_pending = True
        return "".join(digits)

    high, low = divmod(value, 10_000)
    if not high:
        return under_ten_thousand(low)
    result = under_ten_thousand(high) + "万"
    if low:
        if low < 1000:
            result += "零"
        result += under_ten_thousand(low)
    return result


def _normalize_decision(value: Any, line_id: str) -> str:
    key = str(value or "auto").strip().casefold()
    try:
        return _DECISION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"{line_id} 的审核决定无效：{value}") from exc


def _suggest_review_decision(qa_item: dict[str, Any] | None) -> tuple[str, str]:
    if qa_item is None:
        return "review", "没有对应 QA 证据"
    if qa_item.get("error"):
        return "retry", "QA 执行失败"
    if qa_item.get("passed") is True:
        return "approve", "所有 QA 门槛通过"
    checks = qa_item.get("checks") if isinstance(qa_item.get("checks"), dict) else {}
    failed = {name for name, passed in checks.items() if passed is False}
    if failed and failed <= {"silence"}:
        return "review", "仅静音比例异常，需要人工判断是否为表演停顿"
    if failed:
        return "retry", "未通过：" + ", ".join(sorted(failed))
    return "review", "QA 未给出完整判定"


def _time_stretch_wav(source: Path, target: Path, tempo: float) -> None:
    if not 0.5 <= float(tempo) <= 2.0:
        raise ValueError("FFmpeg atempo 仅接受 0.5–2.0，本节点的安全范围必须落在其中")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".tmp.wav")
    command = [
        _ffmpeg_path(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-filter:a",
        f"atempo={float(tempo):.10f}",
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
        creationflags=creationflags,
        check=False,
    )
    if completed.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout or "FFmpeg 未产生输出").strip()
        raise RuntimeError(f"字幕时长适配失败：{detail[-1000:]}")
    temporary.replace(target)


def _boundary_silence_seconds(
    source: Path,
    *,
    duration_seconds: float,
    threshold_db: float,
    minimum_seconds: float,
) -> dict[str, Any]:
    command = [
        _ffmpeg_path(),
        "-hide_banner",
        "-nostats",
        "-i",
        str(source),
        "-af",
        f"silencedetect=noise={float(threshold_db):.3f}dB:d={float(minimum_seconds):.6f}",
        "-f",
        "null",
        "-",
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
        creationflags=creationflags,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "FFmpeg 静音检测失败").strip()
        raise RuntimeError(f"首尾静音检测失败：{detail[-1000:]}")
    events = re.finditer(
        r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)|silence_end:\s*([0-9]+(?:\.[0-9]+)?)",
        completed.stderr or "",
    )
    intervals: list[tuple[float, float]] = []
    current_start: float | None = None
    for event in events:
        if event.group(1) is not None:
            current_start = float(event.group(1))
        elif event.group(2) is not None and current_start is not None:
            intervals.append((current_start, float(event.group(2))))
            current_start = None
    duration = max(0.0, float(duration_seconds))
    leading = 0.0
    trailing = 0.0
    if intervals and intervals[0][0] <= 0.01:
        leading = min(duration, max(0.0, intervals[0][1]))
    if intervals and intervals[-1][1] >= duration - 0.02:
        trailing = max(0.0, duration - max(0.0, intervals[-1][0]))
    return {
        "threshold_db": float(threshold_db),
        "minimum_seconds": float(minimum_seconds),
        "leading_seconds": leading,
        "trailing_seconds": trailing,
        "interval_count": len(intervals),
    }


def _trim_and_time_stretch_wav(
    source: Path,
    target: Path,
    *,
    trim_start_seconds: float,
    trim_end_seconds: float,
    source_duration_seconds: float,
    tempo: float,
) -> None:
    if not 0.5 <= float(tempo) <= 2.0:
        raise ValueError("FFmpeg atempo 仅接受 0.5–2.0，本节点的安全范围必须落在其中")
    trim_start = max(0.0, float(trim_start_seconds))
    trim_stop = max(trim_start + 0.001, float(source_duration_seconds) - max(0.0, float(trim_end_seconds)))
    filters = [f"atrim=start={trim_start:.10f}:end={trim_stop:.10f}", "asetpts=PTS-STARTPTS"]
    if abs(float(tempo) - 1.0) > 1e-6:
        filters.append(f"atempo={float(tempo):.10f}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".tmp.wav")
    command = [
        _ffmpeg_path(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-filter:a",
        ",".join(filters),
        "-c:a",
        "pcm_s16le",
        str(temporary),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
        creationflags=creationflags,
        check=False,
    )
    if completed.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout or "FFmpeg 未产生输出").strip()
        raise RuntimeError(f"语音感知时长适配失败：{detail[-1000:]}")
    temporary.replace(target)


__all__ = [
    "build_line_review",
    "fit_audio_batch_to_cues",
    "load_audio_batch_from_manifest",
    "normalize_script_plan",
    "parse_json_mapping",
]
