from __future__ import annotations

import csv
import io
import json
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import WorkerProtocolError


SRT_RANGE = re.compile(
    r"^\s*(\d{1,3}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(\d{1,3}:\d{2}:\d{2}[,.]\d{3})(?:\s+.*)?$"
)
SPEAKER_PREFIX = re.compile(
    r"^\s*(?:\[([^\]]+)\]|<v\s+([^>]+)>|([^:\n：]{1,80})[:：])\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ScriptIssue:
    severity: str
    code: str
    message: str
    line_index: int | None = None


@dataclass(frozen=True)
class ParsedScript:
    format: str
    lines: list[dict[str, Any]]
    speakers: list[str]
    issues: list[ScriptIssue]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "lines": self.lines,
            "speakers": self.speakers,
            "issues": [asdict(issue) for issue in self.issues],
            "valid": self.valid,
        }


def parse_script_file(
    path: str | Path,
    *,
    known_speakers: set[str] | None = None,
    default_speaker: str = "旁白",
) -> ParsedScript:
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise WorkerProtocolError(f"脚本不存在：{target}")
    text = target.read_text(encoding="utf-8-sig", errors="replace")
    return parse_script(
        text,
        format_hint=target.suffix.lower().lstrip("."),
        known_speakers=known_speakers,
        default_speaker=default_speaker,
    )


def parse_script(
    text: str,
    *,
    format_hint: str = "auto",
    known_speakers: set[str] | None = None,
    default_speaker: str = "旁白",
) -> ParsedScript:
    content = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    hint = str(format_hint or "auto").lower().lstrip(".")
    if hint == "auto":
        stripped = content.lstrip()
        if "-->" in content and any(SRT_RANGE.match(line) for line in content.split("\n")):
            hint = "srt"
        elif stripped.startswith(("[", "{")):
            hint = "json"
        elif _looks_like_csv(content):
            hint = "csv"
        else:
            hint = "txt"
    if hint in {"srt", "vtt"}:
        lines, issues = _parse_srt(content, default_speaker)
        format_name = "srt"
    elif hint == "json":
        lines, issues = _parse_json(content, default_speaker)
        format_name = "json"
    elif hint == "csv":
        lines, issues = _parse_csv(content, default_speaker)
        format_name = "csv"
    elif hint in {"txt", "text"}:
        lines, issues = _parse_text(content, default_speaker)
        format_name = "txt"
    else:
        raise WorkerProtocolError(f"不支持的脚本格式：{format_hint}")
    issues.extend(_validate_lines(lines, known_speakers=known_speakers))
    speakers = sorted({str(line.get("speaker") or default_speaker) for line in lines})
    return ParsedScript(format_name, lines, speakers, issues)


def _parse_srt(text: str, default_speaker: str) -> tuple[list[dict[str, Any]], list[ScriptIssue]]:
    blocks = re.split(r"\n\s*\n", text.strip()) if text.strip() else []
    lines: list[dict[str, Any]] = []
    issues: list[ScriptIssue] = []
    for block_index, block in enumerate(blocks):
        raw_lines = [line.rstrip() for line in block.split("\n") if line.strip()]
        if not raw_lines:
            continue
        cursor = 0
        source_index: str | None = None
        if raw_lines[0].strip().isdigit():
            source_index = raw_lines[0].strip()
            cursor = 1
        if cursor >= len(raw_lines):
            issues.append(ScriptIssue("error", "missing_time", "字幕块缺少时间范围", block_index))
            continue
        match = SRT_RANGE.match(raw_lines[cursor])
        if not match:
            issues.append(
                ScriptIssue("error", "invalid_time", f"无法解析时间范围：{raw_lines[cursor]}", block_index)
            )
            continue
        cursor += 1
        body = "\n".join(raw_lines[cursor:]).strip()
        speaker, body = _extract_speaker(body, default_speaker)
        lines.append(
            {
                "id": str(uuid.uuid4()),
                "order_index": len(lines),
                "source_index": source_index,
                "speaker": speaker,
                "text": body,
                "target_start": _parse_timestamp(match.group(1)),
                "target_end": _parse_timestamp(match.group(2)),
                "language": "zh",
                "preset": "balanced",
                "metadata": {"source_block": block_index + 1},
            }
        )
    return lines, issues


def _parse_csv(text: str, default_speaker: str) -> tuple[list[dict[str, Any]], list[ScriptIssue]]:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    if not rows:
        return [], [ScriptIssue("error", "empty", "CSV 没有数据行")]
    normalized_headers = {str(name or "").strip().lower(): name for name in rows[0]}
    text_key = next(
        (normalized_headers[key] for key in ("text", "line", "台词", "内容") if key in normalized_headers),
        None,
    )
    if text_key is None:
        return [], [ScriptIssue("error", "missing_text_column", "CSV 缺少 text/台词列")]
    speaker_key = next(
        (
            normalized_headers[key]
            for key in ("speaker", "role", "character", "角色", "说话人")
            if key in normalized_headers
        ),
        None,
    )
    lines: list[dict[str, Any]] = []
    issues: list[ScriptIssue] = []
    for index, row in enumerate(rows):
        body = str(row.get(text_key) or "").strip()
        speaker = str(row.get(speaker_key) or default_speaker).strip() if speaker_key else default_speaker
        start = _first_number(row, normalized_headers, ("start", "start_seconds", "开始"))
        end = _first_number(row, normalized_headers, ("end", "end_seconds", "结束"))
        lines.append(
            {
                "id": str(row.get(normalized_headers.get("id")) or uuid.uuid4()),
                "order_index": index,
                "speaker": speaker or default_speaker,
                "text": body,
                "target_start": start,
                "target_end": end,
                "language": str(row.get(normalized_headers.get("language")) or "zh"),
                "preset": str(row.get(normalized_headers.get("preset")) or "balanced"),
                "metadata": {"source_row": index + 2},
            }
        )
    return lines, issues


def _parse_json(text: str, default_speaker: str) -> tuple[list[dict[str, Any]], list[ScriptIssue]]:
    try:
        payload = json.loads(text)
    except ValueError as exc:
        return [], [ScriptIssue("error", "invalid_json", f"JSON 解析失败：{exc}")]
    if isinstance(payload, dict):
        payload = payload.get("lines") or payload.get("segments") or []
    if not isinstance(payload, list):
        return [], [ScriptIssue("error", "invalid_json_shape", "JSON 必须是数组或包含 lines 数组")]
    lines: list[dict[str, Any]] = []
    issues: list[ScriptIssue] = []
    for index, value in enumerate(payload):
        if not isinstance(value, dict):
            issues.append(ScriptIssue("error", "invalid_line", "台词项必须是对象", index))
            continue
        line = dict(value)
        line.setdefault("id", str(uuid.uuid4()))
        line["order_index"] = int(line.get("order_index", index))
        line["speaker"] = str(line.get("speaker") or line.get("role") or default_speaker)
        line["text"] = str(line.get("text") or line.get("line") or "").strip()
        line["target_start"] = _optional_float(
            line.get("target_start", line.get("start_seconds", line.get("start")))
        )
        line["target_end"] = _optional_float(
            line.get("target_end", line.get("end_seconds", line.get("end")))
        )
        line.setdefault("language", "zh")
        line.setdefault("preset", "balanced")
        line.setdefault("metadata", {})
        lines.append(line)
    return lines, issues


def _parse_text(text: str, default_speaker: str) -> tuple[list[dict[str, Any]], list[ScriptIssue]]:
    lines: list[dict[str, Any]] = []
    for raw in text.split("\n"):
        body = raw.strip()
        if not body:
            continue
        speaker, body = _extract_speaker(body, default_speaker)
        lines.append(
            {
                "id": str(uuid.uuid4()),
                "order_index": len(lines),
                "speaker": speaker,
                "text": body,
                "target_start": None,
                "target_end": None,
                "language": "zh",
                "preset": "balanced",
                "metadata": {},
            }
        )
    issues = [] if lines else [ScriptIssue("error", "empty", "脚本没有有效台词")]
    return lines, issues


def _validate_lines(
    lines: list[dict[str, Any]], *, known_speakers: set[str] | None
) -> list[ScriptIssue]:
    issues: list[ScriptIssue] = []
    ids: set[str] = set()
    speakers: set[str] = set()
    previous_start: float | None = None
    previous_end: float | None = None
    for index, line in enumerate(lines):
        line_id = str(line.get("id") or "")
        if line_id in ids:
            issues.append(ScriptIssue("error", "duplicate_id", f"重复台词 ID：{line_id}", index))
        ids.add(line_id)
        text = str(line.get("text") or "").strip()
        if not text:
            issues.append(ScriptIssue("error", "empty_text", "台词文本为空", index))
        speaker = str(line.get("speaker") or "旁白").strip() or "旁白"
        line["speaker"] = speaker
        speakers.add(speaker)
        if known_speakers is not None and speaker not in known_speakers:
            issues.append(
                ScriptIssue("error", "unknown_speaker", f"角色未映射：{speaker}", index)
            )
        start = _optional_float(line.get("target_start"))
        end = _optional_float(line.get("target_end"))
        if start is not None and end is not None:
            if end <= start:
                issues.append(ScriptIssue("error", "invalid_range", "结束时间不晚于开始时间", index))
            if previous_start is not None and start < previous_start:
                issues.append(ScriptIssue("error", "out_of_order", "时间范围发生倒序", index))
            if previous_end is not None and start < previous_end:
                issues.append(ScriptIssue("warning", "overlap", "与上一条台词时间重叠", index))
            previous_start, previous_end = start, end
    if len(speakers) > 8:
        issues.append(ScriptIssue("error", "too_many_speakers", f"角色数 {len(speakers)} 超过上限 8"))
    return issues


def _extract_speaker(text: str, default_speaker: str) -> tuple[str, str]:
    match = SPEAKER_PREFIX.match(text)
    if not match:
        return default_speaker, text.strip()
    speaker = next((value for value in match.groups()[:3] if value), default_speaker)
    body = match.group(4).strip()
    return speaker.strip() or default_speaker, body


def _parse_timestamp(value: str) -> float:
    hours, minutes, tail = value.replace(",", ".").split(":")
    seconds = float(tail)
    return int(hours) * 3600 + int(minutes) * 60 + seconds


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _first_number(
    row: dict[str, Any], headers: dict[str, str], candidates: tuple[str, ...]
) -> float | None:
    for candidate in candidates:
        key = headers.get(candidate)
        if key and row.get(key) not in (None, ""):
            return float(row[key])
    return None


def _looks_like_csv(text: str) -> bool:
    first = text.split("\n", 1)[0].lower()
    return any(separator in first for separator in (",", "\t", ";")) and any(
        name in first for name in ("text", "台词", "speaker", "角色")
    )
