from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import wave
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VoiceProfile:
    profile_id: str
    name: str
    prompt_audio: str
    prompt_audio_sha256: str
    prompt_text: str
    language: str
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags)
        return data


@dataclass(frozen=True)
class VoiceBank:
    profiles: tuple[VoiceProfile, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profiles": [profile.to_dict() for profile in self.profiles],
            "count": len(self.profiles),
            "digest": stable_digest([profile.to_dict() for profile in self.profiles]),
        }

    def resolve(self, name: str) -> VoiceProfile | None:
        key = str(name).strip().casefold()
        return next((item for item in self.profiles if item.name.casefold() == key), None)


@dataclass(frozen=True)
class ScriptLine:
    line_id: str
    index: int
    speaker: str
    text: str
    language: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    scene: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScriptPlan:
    source_format: str
    lines: tuple[ScriptLine, ...]
    issues: tuple[dict[str, Any], ...]

    @property
    def valid(self) -> bool:
        return not any(item.get("severity") == "error" for item in self.issues)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source_format": self.source_format,
            "lines": [line.to_dict() for line in self.lines],
            "issues": list(self.issues),
            "valid": self.valid,
        }
        payload["digest"] = stable_digest(payload["lines"])
        return payload


@dataclass(frozen=True)
class AudioBatch:
    manifest_path: str
    items: tuple[dict[str, Any], ...]

    def successful_items(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.items
            if item.get("status") == "complete" and Path(str(item.get("output_path", ""))).is_file()
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "items": list(self.items),
            "complete": len(self.successful_items()),
            "total": len(self.items),
        }


def parse_line_ids(value: str | Iterable[str]) -> tuple[str, ...]:
    """Parse QA/selection line IDs from JSON, newline or comma separated text."""
    if isinstance(value, str):
        content = value.strip()
        if not content:
            return ()
        parsed: Any = None
        if content.startswith("["):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = None
        values = parsed if isinstance(parsed, list) else re.split(r"[\s,，;；]+", content)
    else:
        values = value
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        line_id = str(raw).strip()
        if line_id and line_id not in seen:
            output.append(line_id)
            seen.add(line_id)
    return tuple(output)


def select_audio_batch_item(
    audio_batch: AudioBatch,
    *,
    mode: str = "position",
    position: int = 1,
    line_id: str = "",
    speaker: str = "",
) -> dict[str, Any]:
    """Select one playable item without changing the source AudioBatch."""
    if not isinstance(audio_batch, AudioBatch):
        raise TypeError("必须连接 AudioBatch")
    playable = audio_batch.successful_items()
    if not playable:
        raise ValueError("AudioBatch 中没有可试听的成功音频")
    if mode == "position":
        index = int(position) - 1
        if index < 0 or index >= len(playable):
            raise IndexError(f"试听序号超出范围：1–{len(playable)}")
        return dict(playable[index])
    if mode == "line_id":
        wanted = str(line_id).strip()
        if not wanted:
            raise ValueError("按 line_id 选择时必须填写 line_id")
        selected = next((item for item in playable if str(item.get("line_id")) == wanted), None)
        if selected is None:
            raise ValueError(f"没有找到可试听 line_id：{wanted}")
        return dict(selected)
    if mode == "speaker":
        wanted = str(speaker).strip().casefold()
        if not wanted:
            raise ValueError("按角色选择时必须填写角色名称")
        selected = next(
            (item for item in playable if str(item.get("speaker") or "").strip().casefold() == wanted),
            None,
        )
        if selected is None:
            raise ValueError(f"没有找到角色的可试听音频：{speaker}")
        return dict(selected)
    raise ValueError(f"不支持的 AudioBatch 选择模式：{mode}")


def merge_audio_batch_items(
    audio_batch: AudioBatch,
    replacements: Iterable[dict[str, Any]],
    manifest_path: str | Path,
) -> AudioBatch:
    """Return a non-destructive merged batch while preserving original item order."""
    if not isinstance(audio_batch, AudioBatch):
        raise TypeError("必须连接 AudioBatch")
    by_id: dict[str, dict[str, Any]] = {}
    for replacement in replacements:
        line_id = str(replacement.get("line_id") or "").strip()
        if not line_id:
            raise ValueError("返修条目缺少 line_id")
        if line_id in by_id:
            raise ValueError(f"返修条目 line_id 重复：{line_id}")
        by_id[line_id] = dict(replacement)
    known = {str(item.get("line_id") or "") for item in audio_batch.items}
    unknown = sorted(set(by_id) - known)
    if unknown:
        raise ValueError("返修条目不属于原 AudioBatch：" + ", ".join(unknown))
    merged = tuple(
        by_id.get(str(item.get("line_id") or ""), dict(item))
        for item in audio_batch.items
    )
    return AudioBatch(str(Path(manifest_path)), merged)


def create_voice_profile(
    name: str,
    prompt_audio: str | Path,
    prompt_text: str,
    language: str,
    tags: str | Iterable[str] = (),
) -> VoiceProfile:
    clean_name = str(name).strip()
    clean_text = str(prompt_text).strip()
    audio_path = Path(prompt_audio).resolve()
    if not clean_name:
        raise ValueError("音色名称不能为空")
    if not clean_text:
        raise ValueError("参考音频逐字稿不能为空")
    if language not in {"zh", "en"}:
        raise ValueError("音色语言必须是 zh 或 en")
    if not audio_path.is_file():
        raise FileNotFoundError(f"参考音频不存在：{audio_path}")
    if isinstance(tags, str):
        clean_tags = tuple(item.strip() for item in re.split(r"[,，]", tags) if item.strip())
    else:
        clean_tags = tuple(str(item).strip() for item in tags if str(item).strip())
    audio_sha256 = file_digest(audio_path)
    profile_id = stable_digest(
        {
            "name": clean_name,
            "audio": audio_sha256,
            "prompt_text": clean_text,
            "language": language,
        }
    )[:16]
    return VoiceProfile(
        profile_id=profile_id,
        name=clean_name,
        prompt_audio=str(audio_path),
        prompt_audio_sha256=audio_sha256,
        prompt_text=clean_text,
        language=language,
        tags=clean_tags,
    )


def create_voice_bank(profiles: Iterable[VoiceProfile]) -> VoiceBank:
    values = tuple(profiles)
    if not 1 <= len(values) <= 8:
        raise ValueError("音色库必须包含 1–8 个音色档案")
    names: set[str] = set()
    ids: set[str] = set()
    for profile in values:
        if not isinstance(profile, VoiceProfile):
            raise TypeError("音色库输入必须是 VoiceProfile")
        key = profile.name.casefold()
        if key in names:
            raise ValueError(f"音色名称重复：{profile.name}")
        if profile.profile_id in ids:
            raise ValueError(f"音色档案重复：{profile.name}")
        names.add(key)
        ids.add(profile.profile_id)
    return VoiceBank(values)


def load_project_exchange(
    path: str | Path,
) -> tuple[VoiceBank, ScriptPlan, AudioBatch, dict[str, Any]]:
    target = Path(path).expanduser().resolve()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取桌面项目交换 JSON：{target}") from exc
    if not isinstance(payload, dict) or payload.get("format") != "t8.firered.project.exchange":
        raise ValueError("不是 T8 FireRedAudio 项目交换 JSON")
    if int(payload.get("version") or 0) != 1:
        raise ValueError(f"不支持的项目交换版本：{payload.get('version')}")
    project_root = (target.parent / str(payload.get("project_root") or ".")).resolve()
    raw_profiles = (payload.get("voice_bank") or {}).get("profiles") or []
    profiles = []
    for raw in raw_profiles:
        if not isinstance(raw, dict):
            raise ValueError("项目交换音色档案必须是 JSON 对象")
        relative = str(raw.get("prompt_audio") or "").strip()
        absolute = str(raw.get("prompt_audio_absolute") or "").strip()
        audio = (project_root / relative).resolve() if relative else Path(absolute).resolve()
        if not audio.is_file():
            raise FileNotFoundError(f"项目音色参考音频不存在：{audio}")
        digest = file_digest(audio)
        expected = str(raw.get("prompt_audio_sha256") or "").lower()
        if expected and digest != expected:
            raise ValueError(f"项目音色参考音频哈希变化：{raw.get('name')}")
        profiles.append(
            VoiceProfile(
                profile_id=str(raw.get("profile_id") or stable_digest(raw)[:16]),
                name=str(raw.get("name") or "").strip(),
                prompt_audio=str(audio),
                prompt_audio_sha256=digest,
                prompt_text=str(raw.get("prompt_text") or "").strip(),
                language=str(raw.get("language") or "zh"),
                tags=tuple(str(item) for item in (raw.get("tags") or [])),
            )
        )
    bank = create_voice_bank(profiles)
    raw_plan = payload.get("script_plan") or {}
    lines = tuple(
        ScriptLine(
            line_id=str(raw.get("line_id") or stable_digest(raw)[:12]),
            index=int(raw.get("index") or index + 1),
            speaker=str(raw.get("speaker") or "旁白"),
            text=str(raw.get("text") or "").strip(),
            language=str(raw.get("language") or "zh"),
            start_seconds=(None if raw.get("start_seconds") is None else float(raw["start_seconds"])),
            end_seconds=(None if raw.get("end_seconds") is None else float(raw["end_seconds"])),
            scene=str(raw.get("scene") or "").strip(),
        )
        for index, raw in enumerate(raw_plan.get("lines") or [])
        if isinstance(raw, dict)
    )
    if not lines or any(not line.text for line in lines):
        raise ValueError("项目交换台词计划为空或含空文本")
    plan = ScriptPlan(
        source_format=str(raw_plan.get("source_format") or "desktop-project"),
        lines=lines,
        issues=tuple(raw_plan.get("issues") or ()),
    )
    batch_items = []
    for raw in (payload.get("audio_batch") or {}).get("items") or []:
        if not isinstance(raw, dict):
            continue
        relative = str(raw.get("output_path") or "").strip()
        absolute = str(raw.get("output_path_absolute") or "").strip()
        output = (project_root / relative).resolve() if relative else Path(absolute).resolve()
        item = dict(raw)
        item["output_path"] = str(output)
        if item.get("status") == "complete" and not output.is_file():
            item["status"] = "missing"
            item["error"] = "桌面项目中的 adopted take 文件不存在"
        batch_items.append(item)
    batch = AudioBatch(str(target), tuple(batch_items))
    summary = {
        "path": str(target),
        "project": payload.get("project") or {},
        "voice_profiles": len(bank.profiles),
        "script_lines": len(plan.lines),
        "adopted_takes": len(batch.successful_items()),
    }
    return bank, plan, batch, summary


_TIME_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)
_INLINE_TIME_RE = re.compile(r"^\s*\[(?P<time>[^]]+-->[^]]+)\]\s*(?P<body>.+)$")
_SPEAKER_BRACKET_RE = re.compile(r"^\s*\[(?P<speaker>[^]]+)]\s*(?P<text>.*)$")
_SPEAKER_COLON_RE = re.compile(r"^\s*(?P<speaker>[^:：\t]{1,64})\s*[:：\t]\s*(?P<text>.+)$")


def parse_timestamp(value: str) -> float:
    parts = value.strip().replace(",", ".").split(":")
    if len(parts) != 3:
        raise ValueError(f"无效时间码：{value}")
    hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"无效时间码：{value}")
    return hours * 3600 + minutes * 60 + seconds


def _speaker_and_text(
    value: str,
    *,
    default_speaker: str,
    known_speakers: set[str],
    colon_is_speaker: bool,
) -> tuple[str, str]:
    bracket = _SPEAKER_BRACKET_RE.match(value)
    if bracket:
        return bracket.group("speaker").strip(), bracket.group("text").strip()
    colon = _SPEAKER_COLON_RE.match(value)
    if colon:
        candidate = colon.group("speaker").strip()
        if colon_is_speaker or candidate.casefold() in known_speakers:
            return candidate, colon.group("text").strip()
    return default_speaker.strip(), value.strip()


def _line_id_payload(
    speaker: str,
    text: str,
    language: str,
    start: float | None,
    end: float | None,
) -> dict[str, Any]:
    return {
        "speaker": speaker,
        "text": text,
        "language": language,
        "start_seconds": start,
        "end_seconds": end,
    }


def _assign_ids(raw_lines: list[dict[str, Any]]) -> tuple[ScriptLine, ...]:
    occurrences: dict[str, int] = {}
    output: list[ScriptLine] = []
    for index, item in enumerate(raw_lines, 1):
        payload = _line_id_payload(
            item["speaker"], item["text"], item["language"], item.get("start_seconds"), item.get("end_seconds")
        )
        base = stable_digest(payload)[:12]
        occurrence = occurrences.get(base, 0) + 1
        occurrences[base] = occurrence
        output.append(
            ScriptLine(
                line_id=f"{base}-{occurrence}",
                index=index,
                speaker=item["speaker"],
                text=item["text"],
                language=item["language"],
                start_seconds=item.get("start_seconds"),
                end_seconds=item.get("end_seconds"),
                scene=str(item.get("scene") or "").strip(),
            )
        )
    return tuple(output)


def _parse_srt(script: str, bank: VoiceBank, default_speaker: str) -> list[dict[str, Any]]:
    normalized = script.replace("\r\n", "\n").replace("\r", "\n").strip("\ufeff\n ")
    blocks = re.split(r"\n\s*\n", normalized)
    names = {item.name.casefold() for item in bank.profiles}
    output: list[dict[str, Any]] = []
    for block in blocks:
        rows = [row.strip() for row in block.splitlines() if row.strip()]
        if rows and rows[0].isdigit():
            rows.pop(0)
        if not rows:
            continue
        match = _TIME_RE.fullmatch(rows[0])
        if not match:
            raise ValueError(f"SRT 字幕块缺少合法时间码：{block[:80]}")
        rows.pop(0)
        body = " ".join(rows).strip()
        speaker, text = _speaker_and_text(
            body,
            default_speaker=default_speaker,
            known_speakers=names,
            colon_is_speaker=False,
        )
        profile = bank.resolve(speaker)
        output.append(
            {
                "speaker": speaker,
                "text": text,
                "language": profile.language if profile else "zh",
                "start_seconds": parse_timestamp(match.group("start")),
                "end_seconds": parse_timestamp(match.group("end")),
            }
        )
    return output


def _parse_role_script(script: str, bank: VoiceBank, default_speaker: str) -> list[dict[str, Any]]:
    names = {item.name.casefold() for item in bank.profiles}
    output: list[dict[str, Any]] = []
    current_scene = ""
    for source_index, raw in enumerate(script.replace("\r", "").splitlines(), 1):
        value = raw.strip()
        if not value:
            continue
        scene_match = re.match(r"^#(?:#\s*|\s*(?:scene|场景)\s*[:：]\s*)(.+)$", value, re.IGNORECASE)
        if scene_match:
            current_scene = scene_match.group(1).strip()
            continue
        if value.startswith("#"):
            continue
        start = end = None
        inline = _INLINE_TIME_RE.match(value)
        if inline:
            time_match = _TIME_RE.fullmatch(inline.group("time").strip())
            if not time_match:
                raise ValueError(f"第 {source_index} 行时间码无效")
            start = parse_timestamp(time_match.group("start"))
            end = parse_timestamp(time_match.group("end"))
            value = inline.group("body")
        speaker, text = _speaker_and_text(
            value,
            default_speaker=default_speaker,
            known_speakers=names,
            colon_is_speaker=True,
        )
        profile = bank.resolve(speaker)
        output.append(
            {
                "speaker": speaker,
                "text": text,
                "language": profile.language if profile else "zh",
                "start_seconds": start,
                "end_seconds": end,
                "scene": current_scene,
            }
        )
    return output


def _parse_json_script(script: str, bank: VoiceBank, default_speaker: str) -> list[dict[str, Any]]:
    parsed = json.loads(script)
    if isinstance(parsed, dict):
        parsed = parsed.get("lines")
    if not isinstance(parsed, list):
        raise ValueError("JSON 脚本必须是数组或包含 lines 数组的对象")
    output: list[dict[str, Any]] = []
    for index, item in enumerate(parsed, 1):
        if not isinstance(item, dict):
            raise ValueError(f"JSON 第 {index} 项必须是对象")
        speaker = str(item.get("speaker") or default_speaker).strip()
        profile = bank.resolve(speaker)
        start = item.get("start_seconds", item.get("start"))
        end = item.get("end_seconds", item.get("end"))
        output.append(
            {
                "speaker": speaker,
                "text": str(item.get("text") or "").strip(),
                "language": str(item.get("language") or (profile.language if profile else "zh")),
                "start_seconds": float(start) if start is not None else None,
                "end_seconds": float(end) if end is not None else None,
                "scene": str(item.get("scene") or "").strip(),
            }
        )
    return output


def parse_script(
    script: str,
    source_format: str,
    bank: VoiceBank,
    default_speaker: str = "",
) -> ScriptPlan:
    if not isinstance(bank, VoiceBank):
        raise TypeError("脚本预检必须连接音色库")
    content = str(script).strip()
    if not content:
        raise ValueError("脚本不能为空")
    fallback = default_speaker.strip() or bank.profiles[0].name
    fmt = source_format
    if fmt == "auto":
        stripped = content.lstrip()
        if stripped.startswith("{") or stripped.startswith("[") and not _INLINE_TIME_RE.match(stripped):
            try:
                json.loads(content)
                fmt = "json"
            except json.JSONDecodeError:
                fmt = "srt" if any(_TIME_RE.fullmatch(row.strip()) for row in content.splitlines()) else "role_script"
        else:
            fmt = "srt" if any(_TIME_RE.fullmatch(row.strip()) for row in content.splitlines()) else "role_script"
    if fmt == "srt":
        raw_lines = _parse_srt(content, bank, fallback)
    elif fmt == "role_script":
        raw_lines = _parse_role_script(content, bank, fallback)
    elif fmt == "json":
        raw_lines = _parse_json_script(content, bank, fallback)
    else:
        raise ValueError(f"不支持的脚本格式：{source_format}")
    lines = _assign_ids(raw_lines)
    issues: list[dict[str, Any]] = []
    previous_timed: ScriptLine | None = None
    for line in lines:
        if not line.text:
            issues.append({"severity": "error", "line_id": line.line_id, "message": "台词为空"})
        if bank.resolve(line.speaker) is None:
            issues.append(
                {"severity": "error", "line_id": line.line_id, "message": f"音色库中没有角色：{line.speaker}"}
            )
        if line.language not in {"zh", "en"}:
            issues.append(
                {"severity": "error", "line_id": line.line_id, "message": f"不支持的语言：{line.language}"}
            )
        if (line.start_seconds is None) != (line.end_seconds is None):
            issues.append({"severity": "error", "line_id": line.line_id, "message": "开始和结束时间必须同时提供"})
        if line.start_seconds is not None and line.end_seconds is not None:
            if line.start_seconds < 0 or line.end_seconds <= line.start_seconds:
                issues.append({"severity": "error", "line_id": line.line_id, "message": "时间范围无效"})
            if previous_timed and line.start_seconds < float(previous_timed.start_seconds or 0):
                issues.append({"severity": "error", "line_id": line.line_id, "message": "时间码不是递增顺序"})
            elif previous_timed and line.start_seconds < float(previous_timed.end_seconds or 0):
                issues.append({"severity": "warning", "line_id": line.line_id, "message": "时间范围与上一条重叠"})
            previous_timed = line
    if not lines:
        issues.append({"severity": "error", "line_id": "", "message": "脚本中没有可生成的台词"})
    return ScriptPlan(fmt, lines, tuple(issues))


def line_fingerprint(
    line: ScriptLine,
    profile: VoiceProfile,
    settings: dict[str, Any],
    model_identity: str,
) -> str:
    return stable_digest(
        {
            "line": line.to_dict(),
            "profile": profile.to_dict(),
            "settings": settings,
            "model": model_identity,
        }
    )


def load_manifest(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.is_file():
        return None
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"批量配音 manifest 损坏：{target}") from exc
    if not isinstance(value, dict) or value.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"不支持的批量配音 manifest：{target}")
    return value


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def manifest_items_by_id(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not manifest:
        return {}
    return {
        str(item.get("line_id")): item
        for item in manifest.get("items", [])
        if isinstance(item, dict) and item.get("line_id")
    }


def can_reuse_manifest_item(
    item: dict[str, Any] | None,
    expected_fingerprint: str,
    expected_output: str | Path,
) -> bool:
    if not item or item.get("status") != "complete" or item.get("fingerprint") != expected_fingerprint:
        return False
    expected = Path(expected_output).resolve()
    recorded_value = item.get("output_path")
    if not recorded_value:
        return False
    try:
        recorded = Path(str(recorded_value)).resolve()
    except OSError:
        return False
    return recorded == expected and expected.is_file()


def _read_wav_tensor(path: str | Path):
    import torch

    with wave.open(str(path), "rb") as reader:
        if reader.getsampwidth() != 2:
            raise ValueError(f"仅支持 PCM16 WAV：{path}")
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        frames = reader.readframes(reader.getnframes())
    tensor = torch.frombuffer(bytearray(frames), dtype=torch.int16).clone()
    tensor = tensor.reshape(-1, channels).transpose(0, 1).float() / 32768.0
    return tensor, sample_rate


def _resample_linear(audio, source_rate: int, target_rate: int):
    if source_rate == target_rate:
        return audio
    from torch.nn import functional

    length = max(1, round(audio.shape[-1] * target_rate / source_rate))
    return functional.interpolate(audio.unsqueeze(0), size=length, mode="linear", align_corners=False).squeeze(0)


def _write_wav_tensor(path: str | Path, audio, sample_rate: int) -> None:
    import torch

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pcm = (
        (audio.clamp(-1.0, 32767.0 / 32768.0) * 32768.0)
        .round()
        .clamp(-32768, 32767)
        .to(torch.int16)
        .transpose(0, 1)
        .contiguous()
    )
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(int(audio.shape[0]))
        writer.setsampwidth(2)
        writer.setframerate(int(sample_rate))
        writer.writeframes(pcm.numpy().tobytes())


def crop_wav_region(
    source_path: str | Path,
    output_path: str | Path,
    *,
    start_seconds: float,
    end_seconds: float,
    context_ms: int = 0,
) -> dict[str, Any]:
    """Create a PCM16 repair clip without modifying or resampling its source."""
    source = Path(source_path).resolve()
    target = Path(output_path).resolve()
    if source == target:
        raise ValueError("局部修复片段不能覆盖源音频")
    with wave.open(str(source), "rb") as reader:
        if reader.getsampwidth() != 2:
            raise ValueError("局部修复当前只支持 PCM16 WAV")
        channels = reader.getnchannels()
        sample_rate = reader.getframerate()
        total_frames = reader.getnframes()
        frames = reader.readframes(total_frames)
    duration = total_frames / sample_rate
    requested_start = float(start_seconds)
    requested_end = float(end_seconds)
    if requested_start < 0 or requested_end <= requested_start:
        raise ValueError("局部修复时间范围无效：开始必须 >= 0 且结束必须大于开始")
    if requested_start >= duration:
        raise ValueError(f"局部修复开始时间超出源音频：{requested_start:.3f}s / {duration:.3f}s")
    requested_end = min(requested_end, duration)
    context = max(0, min(5000, int(context_ms))) / 1000.0
    start_frame = max(0, round((requested_start - context) * sample_rate))
    end_frame = min(total_frames, round((requested_end + context) * sample_rate))
    if end_frame <= start_frame:
        raise ValueError("局部修复裁切后没有有效采样")
    frame_bytes = channels * 2
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames[start_frame * frame_bytes : end_frame * frame_bytes])
    return {
        "source_path": str(source),
        "source_sha256": file_digest(source),
        "clip_path": str(target),
        "clip_sha256": file_digest(target),
        "sample_rate": sample_rate,
        "channels": channels,
        "source_frames": total_frames,
        "source_duration_seconds": duration,
        "requested_start_seconds": float(start_seconds),
        "requested_end_seconds": float(end_seconds),
        "replace_start_frame": start_frame,
        "replace_end_frame": end_frame,
        "replace_start_seconds": start_frame / sample_rate,
        "replace_end_seconds": end_frame / sample_rate,
        "context_ms": int(context_ms),
        "source_modified": False,
    }


def replace_wav_region(
    source_path: str | Path,
    edited_path: str | Path,
    output_path: str | Path,
    *,
    replace_start_frame: int,
    replace_end_frame: int,
    crossfade_ms: int = 40,
    expected_source_sha256: str = "",
) -> dict[str, Any]:
    """Splice an edited clip into a source with equal-power boundary blends."""
    import torch

    source = Path(source_path).resolve()
    edited = Path(edited_path).resolve()
    target = Path(output_path).resolve()
    if source == target:
        raise ValueError("局部修复输出不能覆盖源音频")
    before_digest = file_digest(source)
    if expected_source_sha256 and before_digest.lower() != expected_source_sha256.lower():
        raise ValueError("源音频自创建修复计划后已变化，拒绝回填")
    original, sample_rate = _read_wav_tensor(source)
    replacement, edited_rate = _read_wav_tensor(edited)
    replacement = _resample_linear(replacement, edited_rate, sample_rate)
    if replacement.shape[0] == 1 and original.shape[0] > 1:
        replacement = replacement.repeat(original.shape[0], 1)
    elif original.shape[0] == 1 and replacement.shape[0] > 1:
        replacement = replacement.mean(dim=0, keepdim=True)
    elif replacement.shape[0] != original.shape[0]:
        raise ValueError("编辑片段与源音频声道数不兼容")
    start = max(0, int(replace_start_frame))
    end = min(int(original.shape[-1]), int(replace_end_frame))
    if end <= start:
        raise ValueError("修复计划中的回填范围无效")
    if replacement.shape[-1] <= 0:
        raise ValueError("编辑片段为空")
    source_region = original[:, start:end]
    requested_fade = round(max(0, min(2000, int(crossfade_ms))) * sample_rate / 1000)
    actual_fade = min(
        requested_fade,
        int(source_region.shape[-1]) // 2,
        int(replacement.shape[-1]) // 2,
    )
    if actual_fade:
        phase = torch.linspace(0.0, math.pi / 2.0, actual_fade, dtype=torch.float32)
        fade_out = torch.cos(phase).unsqueeze(0)
        fade_in = torch.sin(phase).unsqueeze(0)
        replacement[:, :actual_fade] = (
            source_region[:, :actual_fade] * fade_out
            + replacement[:, :actual_fade] * fade_in
        )
        replacement[:, -actual_fade:] = (
            replacement[:, -actual_fade:] * fade_out
            + source_region[:, -actual_fade:] * fade_in
        )
    repaired = torch.cat((original[:, :start], replacement, original[:, end:]), dim=-1)
    peak_before_write = float(repaired.abs().max().item()) if repaired.numel() else 0.0
    clipped_samples = int((repaired.abs() > 1.0).sum().item())
    _write_wav_tensor(target, repaired, sample_rate)
    after_digest = file_digest(source)
    return {
        "source_path": str(source),
        "edited_clip_path": str(edited),
        "output_path": str(target),
        "source_sha256": before_digest,
        "edited_clip_sha256": file_digest(edited),
        "output_sha256": file_digest(target),
        "source_preserved": before_digest == after_digest,
        "sample_rate": sample_rate,
        "channels": int(original.shape[0]),
        "replace_start_seconds": start / sample_rate,
        "replace_end_seconds": end / sample_rate,
        "source_region_duration_seconds": (end - start) / sample_rate,
        "edited_duration_seconds": int(replacement.shape[-1]) / sample_rate,
        "output_duration_seconds": int(repaired.shape[-1]) / sample_rate,
        "duration_delta_seconds": (int(replacement.shape[-1]) - (end - start)) / sample_rate,
        "crossfade_requested_ms": int(crossfade_ms),
        "crossfade_actual_ms": actual_fade * 1000 / sample_rate,
        "crossfade_curve": "equal_power",
        "edited_source_sample_rate": edited_rate,
        "edited_resampled": edited_rate != sample_rate,
        "peak_before_write": peak_before_write,
        "clipped_samples_before_write": clipped_samples,
    }


def pad_wav_to_frames(path: str | Path, total_frames: int) -> None:
    """Pad a PCM16 WAV with silence so every stem shares an exact endpoint."""
    target = Path(path)
    with wave.open(str(target), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frame_count = reader.getnframes()
        frames = reader.readframes(frame_count)
    if sample_width != 2:
        raise ValueError("分轨补齐当前只支持 PCM16 WAV")
    wanted = max(frame_count, int(total_frames))
    if wanted == frame_count:
        return
    frames += bytes((wanted - frame_count) * channels * sample_width)
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)


def render_grouped_stems(
    items: Iterable[dict[str, Any]],
    placements: Iterable[dict[str, Any]],
    output_directory: str | Path,
    *,
    group_key: str,
    filename_prefix: str,
    sample_rate: int,
    total_frames: int,
) -> list[dict[str, Any]]:
    """Render timeline-aligned speaker or scene stems and pad them to master length."""
    playable = [
        dict(item)
        for item in items
        if item.get("status") == "complete" and Path(str(item.get("output_path") or "")).is_file()
    ]
    offsets = {
        str(item.get("line_id") or ""): float(item.get("offset_seconds") or 0.0)
        for item in placements
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in playable:
        label = str(item.get(group_key) or "unassigned").strip() or "unassigned"
        aligned = dict(item)
        aligned["start_seconds"] = offsets.get(str(item.get("line_id") or ""), 0.0)
        aligned["end_seconds"] = None
        groups.setdefault(label, []).append(aligned)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for label, values in sorted(groups.items(), key=lambda pair: pair[0].casefold()):
        safe = "".join(
            character if character.isalnum() or character in "._-" else "-"
            for character in label
        ).strip(".-") or "unassigned"
        base = f"{filename_prefix}-{safe}"[:100]
        name = base
        suffix = 2
        while name.casefold() in used_names:
            name = f"{base}-{suffix}"
            suffix += 1
        used_names.add(name.casefold())
        target = output / f"{name}.wav"
        report = render_timeline_to_wav(
            values,
            target,
            mode="timeline",
            gap_ms=0,
            crossfade_ms=0,
            peak_policy="none",
            sample_rate=sample_rate,
        )
        pad_wav_to_frames(target, total_frames)
        reports.append(
            {
                "group": label,
                "group_key": group_key,
                "path": str(target),
                "sha256": file_digest(target),
                "items": [str(item.get("line_id") or "") for item in values],
                "sample_rate": sample_rate,
                "channels": report["channels"],
                "duration_seconds": total_frames / sample_rate,
            }
        )
    return reports


def build_batch_subtitles(
    items: Iterable[dict[str, Any]], placements: Iterable[dict[str, Any]]
) -> tuple[str, str, list[dict[str, Any]]]:
    """Build actual-timing SRT and VTT from rendered batch placements."""
    by_id = {str(item.get("line_id") or ""): item for item in items}
    cues: list[dict[str, Any]] = []
    for placement in placements:
        line_id = str(placement.get("line_id") or "")
        source = by_id.get(line_id, {})
        start = float(placement.get("offset_seconds") or 0.0)
        end = start + float(placement.get("duration_seconds") or 0.0)
        if end <= start:
            continue
        speaker = str(source.get("speaker") or "").strip()
        text = str(source.get("text") or "").strip()
        body = f"[{speaker}] {text}" if speaker and text else text or speaker or line_id
        cues.append({"line_id": line_id, "start_seconds": start, "end_seconds": end, "text": body})
    cues.sort(key=lambda cue: (cue["start_seconds"], cue["end_seconds"], cue["line_id"]))

    def timestamp(value: float, separator: str) -> str:
        milliseconds = max(0, round(value * 1000))
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, millis = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"

    srt_blocks = []
    vtt_blocks = ["WEBVTT", ""]
    for index, cue in enumerate(cues, 1):
        srt_blocks.append(
            f"{index}\n{timestamp(cue['start_seconds'], ',')} --> {timestamp(cue['end_seconds'], ',')}\n{cue['text']}"
        )
        vtt_blocks.append(
            f"{timestamp(cue['start_seconds'], '.')} --> {timestamp(cue['end_seconds'], '.')}\n{cue['text']}\n"
        )
    srt = "\n\n".join(srt_blocks) + ("\n" if srt_blocks else "")
    vtt = "\n".join(vtt_blocks).rstrip() + "\n"
    return srt, vtt, cues


def render_timeline_to_wav(
    items: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    mode: str = "timeline",
    gap_ms: int = 120,
    crossfade_ms: int = 0,
    peak_policy: str = "limit",
    sample_rate: int = 24000,
    gap_fill_path: str | Path | None = None,
    auto_fill_gaps: bool = False,
    target_lufs: float | None = None,
    loudness_range_lu: float = 7.0,
    true_peak_dbfs: float = -1.0,
) -> dict[str, Any]:
    import torch

    values = [item for item in items if item.get("status") == "complete" and Path(str(item.get("output_path", ""))).is_file()]
    if not values:
        raise ValueError("没有可渲染的成功音频")
    if mode not in {"sequence", "timeline", "overlay"}:
        raise ValueError(f"不支持的时间线模式：{mode}")
    if peak_policy not in {"limit", "clip", "none"}:
        raise ValueError(f"不支持的峰值策略：{peak_policy}")
    if not 0 <= int(crossfade_ms) <= 2000:
        raise ValueError("交叉淡化必须在 0…2000 ms")
    if auto_fill_gaps and not gap_fill_path:
        raise ValueError("启用自动补空隙时必须连接 room tone 音频")
    decoded: list[tuple[dict[str, Any], Any, int]] = []
    channels = 1
    for item in values:
        audio, source_rate = _read_wav_tensor(item["output_path"])
        audio = _resample_linear(audio, source_rate, sample_rate)
        channels = max(channels, int(audio.shape[0]))
        decoded.append((item, audio, source_rate))
    placements: list[dict[str, Any]] = []
    render_entries: list[tuple[dict[str, Any], Any, int]] = []
    cursor = 0
    gap_samples = round(max(0, gap_ms) * sample_rate / 1000)
    requested_crossfade = round(max(0, crossfade_ms) * sample_rate / 1000)
    max_end = 0
    for item, audio, source_rate in decoded:
        if audio.shape[0] == 1 and channels > 1:
            audio = audio.repeat(channels, 1)
        elif audio.shape[0] != channels:
            raise ValueError("时间线音频声道数不兼容")
        if mode == "overlay":
            offset = 0
        elif mode == "timeline" and item.get("start_seconds") is not None:
            offset = round(max(0.0, float(item["start_seconds"])) * sample_rate)
        elif mode == "sequence" and render_entries and requested_crossfade:
            previous_audio = render_entries[-1][1]
            actual = min(
                requested_crossfade,
                max(0, int(previous_audio.shape[-1]) - 1),
                max(0, int(audio.shape[-1]) - 1),
            )
            offset = max(0, cursor - gap_samples - actual)
        else:
            offset = cursor
        end = offset + audio.shape[-1]
        cue_end = item.get("end_seconds")
        placements.append(
            {
                "line_id": item.get("line_id"),
                "speaker": item.get("speaker"),
                "offset_seconds": offset / sample_rate,
                "duration_seconds": audio.shape[-1] / sample_rate,
                "source_sample_rate": source_rate,
                "cue_overrun_seconds": (
                    max(0.0, end / sample_rate - float(cue_end)) if mode == "timeline" and cue_end is not None else 0.0
                ),
            }
        )
        render_entries.append((item, audio, offset))
        cursor = end + gap_samples
        max_end = max(max_end, end)
    crossfades: list[dict[str, Any]] = []
    if mode != "overlay" and requested_crossfade:
        for index in range(1, len(render_entries)):
            previous_item, previous_audio, previous_offset = render_entries[index - 1]
            current_item, current_audio, current_offset = render_entries[index]
            previous_end = previous_offset + int(previous_audio.shape[-1])
            overlap = max(0, previous_end - current_offset)
            actual = min(
                requested_crossfade,
                overlap,
                int(previous_audio.shape[-1]),
                int(current_audio.shape[-1]),
            )
            if actual <= 0:
                continue
            import torch

            phase = torch.linspace(0.0, math.pi / 2.0, actual, dtype=torch.float32)
            current_fade_end = min(
                int(current_audio.shape[-1]),
                max(actual, previous_end - current_offset),
            )
            current_fade_start = current_fade_end - actual
            previous_audio[:, -actual:] *= torch.cos(phase)
            current_audio[:, current_fade_start:current_fade_end] *= torch.sin(phase)
            crossfades.append(
                {
                    "from_line_id": previous_item.get("line_id"),
                    "to_line_id": current_item.get("line_id"),
                    "duration_seconds": actual / sample_rate,
                }
            )
    mixed = torch.zeros((channels, max_end), dtype=torch.float32)
    for _item, audio, offset in render_entries:
        mixed[:, offset : offset + audio.shape[-1]] += audio
    filled_gaps: list[dict[str, float]] = []
    if auto_fill_gaps and gap_fill_path:
        filler, filler_rate = _read_wav_tensor(gap_fill_path)
        filler = _resample_linear(filler, filler_rate, sample_rate)
        if filler.shape[-1] == 0:
            raise ValueError("room tone 音频为空")
        if filler.shape[0] == 1 and channels > 1:
            filler = filler.repeat(channels, 1)
        elif filler.shape[0] != channels:
            raise ValueError("room tone 与时间线声道数不兼容")
        repetitions = max(1, math.ceil(max_end / int(filler.shape[-1])))
        filler = filler.repeat(1, repetitions)[:, :max_end]
        envelope = torch.zeros(max_end, dtype=torch.float32)
        intervals = sorted(
            (offset, offset + int(audio.shape[-1]))
            for _item, audio, offset in render_entries
        )
        merged: list[list[int]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        gaps: list[tuple[int, int]] = []
        gap_start = 0
        for start, end in merged:
            if start > gap_start:
                gaps.append((gap_start, start))
            gap_start = max(gap_start, end)
        if gap_start < max_end:
            gaps.append((gap_start, max_end))
        edge = max(1, round(0.02 * sample_rate))
        for start, end in gaps:
            if end <= start:
                continue
            envelope[start:end] = 1.0
            fade = min(edge, (end - start) // 2)
            if fade:
                envelope[start : start + fade] *= torch.linspace(0.0, 1.0, fade)
                envelope[end - fade : end] *= torch.linspace(1.0, 0.0, fade)
            filled_gaps.append(
                {
                    "start_seconds": start / sample_rate,
                    "end_seconds": end / sample_rate,
                    "duration_seconds": (end - start) / sample_rate,
                }
            )
        mixed += filler * envelope.unsqueeze(0)
    peak_before = float(mixed.abs().max().item()) if mixed.numel() else 0.0
    gain = 1.0
    if peak_policy == "limit" and peak_before > 0.98:
        gain = 0.98 / peak_before
        mixed.mul_(gain)
    clipped_samples = int((mixed.abs() > 1.0).sum().item())
    if peak_policy in {"limit", "clip"}:
        mixed.clamp_(-1.0, 1.0)
    pcm = (mixed.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).transpose(0, 1).contiguous()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_target = target if target_lufs is None else target.with_name(
        f".{target.stem}-{stable_digest(str(target))[:8]}.raw.wav"
    )
    with wave.open(str(raw_target), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm.numpy().tobytes())
    mastering = None
    if target_lufs is not None:
        from .postproduction import master_wav

        try:
            mastering = master_wav(
                raw_target,
                target,
                target_lufs=float(target_lufs),
                loudness_range_lu=float(loudness_range_lu),
                true_peak_dbfs=float(true_peak_dbfs),
                sample_rate=sample_rate,
            )
        finally:
            raw_target.unlink(missing_ok=True)
    return {
        "mode": mode,
        "output_path": str(target),
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_seconds": max_end / sample_rate,
        "peak_before": peak_before,
        "applied_gain": gain,
        "clipped_samples_before_write": clipped_samples,
        "placements": placements,
        "crossfade_ms": int(crossfade_ms),
        "crossfades": crossfades,
        "auto_fill_gaps": bool(auto_fill_gaps),
        "filled_gaps": filled_gaps,
        "mastering": mastering,
    }


def normalize_text(value: str, language: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    if language == "en":
        return re.findall(r"[a-z0-9']+", normalized)
    return [character for character in normalized if not character.isspace() and not unicodedata.category(character).startswith("P")]


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_token in enumerate(reference, 1):
        current = [row]
        for column, hypothesis_token in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def text_error_rate(reference: str, hypothesis: str, language: str) -> tuple[str, float]:
    expected = normalize_text(reference, language)
    actual = normalize_text(hypothesis, language)
    metric = "wer" if language == "en" else "cer"
    if not expected:
        return metric, 0.0 if not actual else 1.0
    return metric, edit_distance(expected, actual) / len(expected)


def wav_metrics(path: str | Path) -> dict[str, Any]:
    import torch

    audio, sample_rate = _read_wav_tensor(path)
    absolute = audio.abs()
    duration = audio.shape[-1] / sample_rate
    rms = float(torch.sqrt(torch.mean(audio.square())).item()) if audio.numel() else 0.0
    peak = float(absolute.max().item()) if audio.numel() else 0.0
    clipping_ratio = float((absolute >= 32760 / 32768).float().mean().item()) if audio.numel() else 0.0
    silence_ratio = float((absolute <= 10 ** (-50 / 20)).float().mean().item()) if audio.numel() else 1.0
    return {
        "sample_rate": sample_rate,
        "channels": int(audio.shape[0]),
        "duration_seconds": duration,
        "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
        "peak": peak,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
    }
