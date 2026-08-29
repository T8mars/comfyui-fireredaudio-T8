from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .audio_compare import prepare_synchronized_ab
from .errors import WorkerProtocolError


PROJECT_SCHEMA_VERSION = 9
PROJECT_SUFFIX = ".firered"
PROJECT_DIRECTORIES = (
    "assets",
    "backups",
    "voices",
    "scripts",
    "segments",
    "renders",
    "cache",
    "logs",
    "previews",
)


def _replace_with_retry(source: Path, target: Path, attempts: int = 8) -> None:
    """Atomically replace a project file despite short Windows scanner locks."""
    for attempt in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt + 1 >= attempts:
                raise
            time.sleep(min(0.2, 0.01 * (2 ** attempt)))


JOB_STATES = {
    "draft",
    "queued",
    "validating",
    "running",
    "completed",
    "failed",
    "cancelled",
    "interrupted",
}
JOB_STAGES = {"generate", "decode", "qa", "render"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _clean_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[<>:\"/\\|?*\x00-\x1f]+", "_", str(value or "").strip())
    cleaned = cleaned.strip(" .")[:120]
    return cleaned or fallback


def normalize_project_root(value: str | Path, *, create_suffix: bool = False) -> Path:
    raw = Path(value).expanduser()
    if create_suffix and raw.suffix.lower() != PROJECT_SUFFIX:
        raw = raw.with_name(raw.name + PROJECT_SUFFIX)
    return raw.resolve()


def project_root_from_parent(parent: str | Path, name: str) -> Path:
    """Resolve the desktop UI's parent-directory + project-name input safely.

    A Windows drive root has an empty ``Path.name``.  It therefore must never be
    passed through ``with_name`` directly.  If the field currently contains an
    opened ``.firered`` project, a new project is created beside it.
    """
    cleaned = _clean_name(name, "")
    if not cleaned:
        raise WorkerProtocolError("请填写有效的新项目名称")
    base = Path(parent).expanduser()
    if base.suffix.lower() == PROJECT_SUFFIX:
        base = base.parent
    return (base / f"{cleaned}{PROJECT_SUFFIX}").resolve()


def safe_project_path(root: str | Path, relative: str | Path) -> Path:
    base = Path(root).resolve()
    candidate = (base / Path(relative)).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise WorkerProtocolError("项目路径不能越出项目目录") from exc
    return candidate


@dataclass(frozen=True)
class ProjectSummary:
    id: str
    name: str
    root: str
    schema_version: int
    created_at: str
    updated_at: str


class ProjectStore:
    """Durable project metadata with media kept as ordinary relative-path files."""

    def __init__(self, root: str | Path, *, recover_interrupted: bool = False):
        self.root = normalize_project_root(root)
        self.database_path = self.root / "project.sqlite"
        self.manifest_path = self.root / "project.json"
        if not self.database_path.is_file() or not self.manifest_path.is_file():
            raise WorkerProtocolError(f"不是有效的 FireRedAudio 项目：{self.root}")
        migrated = self._migrate()
        if migrated:
            self._write_manifest()
        if recover_interrupted:
            self.recover_interrupted_jobs()

    @classmethod
    def create(cls, root: str | Path, name: str | None = None) -> "ProjectStore":
        target = normalize_project_root(root, create_suffix=True)
        if target.exists() and any(target.iterdir()):
            raise WorkerProtocolError(f"项目目录不是空目录：{target}")
        target.mkdir(parents=True, exist_ok=True)
        for directory in PROJECT_DIRECTORIES:
            (target / directory).mkdir(exist_ok=True)
        project_id = str(uuid.uuid4())
        created = utc_now()
        project_name = str(name or target.stem).strip()[:120] or "未命名项目"
        database = target / "project.sqlite"
        connection = sqlite3.connect(database)
        try:
            _initialize_schema(connection)
            connection.execute(
                "INSERT INTO project(id,name,schema_version,created_at,updated_at) "
                "VALUES(?,?,?,?,?)",
                (project_id, project_name, PROJECT_SCHEMA_VERSION, created, created),
            )
            connection.commit()
        finally:
            connection.close()
        (target / "project.json").write_text(
            json.dumps(
                {
                    "id": project_id,
                    "name": project_name,
                    "root": str(target),
                    "schema_version": PROJECT_SCHEMA_VERSION,
                    "created_at": created,
                    "updated_at": created,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        store = cls(target)
        store._write_manifest()
        return store

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def summary(self) -> ProjectSummary:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM project LIMIT 1").fetchone()
        if row is None:
            raise WorkerProtocolError("项目数据库缺少 project 记录")
        return ProjectSummary(
            id=row["id"],
            name=row["name"],
            root=str(self.root),
            schema_version=int(row["schema_version"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def snapshot(self) -> dict[str, Any]:
        summary = asdict(self.summary())
        with self.connection() as connection:
            counts = {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in (
                    "assets",
                    "voice_profiles",
                    "script_lines",
                    "script_revisions",
                    "casting_roles",
                    "pronunciation_entries",
                    "production_clips",
                    "takes",
                    "jobs",
                    "timeline_clips",
                    "markers",
                    "renders",
                    "transcript_segments",
                )
            }
            counts["archived_script_lines"] = int(
                connection.execute("SELECT COUNT(*) FROM script_lines WHERE archived=1").fetchone()[0]
            )
            counts["script_lines"] = int(
                connection.execute("SELECT COUNT(*) FROM script_lines WHERE archived=0").fetchone()[0]
            )
            job_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
        summary["counts"] = counts
        summary["job_counts"] = {row["status"]: int(row["count"]) for row in job_rows}
        summary["queue_control"] = self.queue_control()
        return summary

    def import_asset(
        self,
        source: str | Path,
        *,
        kind: str = "audio",
        copy_into_project: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_path = Path(source).expanduser().resolve()
        if not source_path.is_file():
            raise WorkerProtocolError(f"素材不存在：{source_path}")
        digest = content_sha256(source_path)
        with self.connection() as connection:
            existing = connection.execute(
                "SELECT * FROM assets WHERE sha256=? AND kind=? LIMIT 1", (digest, kind)
            ).fetchone()
            if existing is not None:
                return _asset_dict(existing)
        asset_id = str(uuid.uuid4())
        if copy_into_project:
            suffix = source_path.suffix.lower()[:16]
            filename = f"{digest[:16]}-{_clean_name(source_path.stem, 'asset')}{suffix}"
            target = safe_project_path(self.root, Path("assets") / filename)
            if not target.exists():
                temporary = target.with_suffix(target.suffix + ".tmp")
                shutil.copy2(source_path, temporary)
                if content_sha256(temporary) != digest:
                    temporary.unlink(missing_ok=True)
                    raise WorkerProtocolError("素材复制后哈希不一致")
                _replace_with_retry(temporary, target)
            relative = target.relative_to(self.root).as_posix()
        else:
            relative = source_path.as_posix()
        created = utc_now()
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO assets(id,kind,rel_path,external,sha256,source_name,metadata_json,created_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    asset_id,
                    str(kind or "asset")[:40],
                    relative,
                    0 if copy_into_project else 1,
                    digest,
                    source_path.name,
                    _json(metadata or {}),
                    created,
                ),
            )
            row = connection.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
        self._touch("asset", asset_id, "imported", {"kind": kind, "sha256": digest})
        return _asset_dict(row)

    def get_asset(self, asset_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
        if row is None:
            raise WorkerProtocolError(f"素材不存在：{asset_id}")
        return _asset_dict(row)

    def resolve_asset_path(self, asset_id: str, *, verify_hash: bool = True) -> Path:
        asset = self.get_asset(asset_id)
        if asset["external"]:
            path = Path(str(asset["rel_path"])).expanduser().resolve()
        else:
            path = safe_project_path(self.root, asset["rel_path"])
        if not path.is_file():
            raise WorkerProtocolError(f"素材文件不存在：{path}")
        if verify_hash and content_sha256(path) != asset["sha256"]:
            raise WorkerProtocolError(f"素材文件哈希已变更：{path}")
        return path

    def upsert_voice_profile(self, value: dict[str, Any]) -> dict[str, Any]:
        profile_id = str(value.get("id") or "").strip()
        name = str(value.get("name") or "").strip()[:120]
        if not name:
            raise WorkerProtocolError("角色音色缺少 name")
        reference_asset_id = str(value.get("reference_asset_id") or "").strip() or None
        now = utc_now()
        with self.connection() as connection:
            # Serialize the read-before-insert decision.  Without an immediate
            # write lock concurrent desktop requests can all observe no match
            # and create duplicate profiles for the same name.
            connection.execute("BEGIN IMMEDIATE")
            if not profile_id:
                same_name = connection.execute(
                    "SELECT id FROM voice_profiles WHERE name=? COLLATE NOCASE ORDER BY updated_at DESC LIMIT 1",
                    (name,),
                ).fetchone()
                profile_id = str(same_name["id"]) if same_name is not None else str(uuid.uuid4())
            existing = connection.execute(
                "SELECT created_at,reference_asset_id,transcript,language FROM voice_profiles WHERE id=?",
                (profile_id,),
            ).fetchone()
            created = existing["created_at"] if existing else now
            voice_changed = bool(existing is not None and (
                str(existing["reference_asset_id"] or "") != str(reference_asset_id or "")
                or str(existing["transcript"] or "") != str(value.get("transcript") or "")[:20_000]
                or str(existing["language"] or "") != str(value.get("language") or "zh")[:16]
            ))
            if voice_changed:
                active = connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE line_id IN "
                    "(SELECT id FROM script_lines WHERE archived=0 AND voice_profile_id=?) "
                    "AND status IN ('validating','running')",
                    (profile_id,),
                ).fetchone()[0]
                if int(active):
                    raise WorkerProtocolError("该音色仍有正在生成的任务，请等待或取消后再更新参考")
                locked = connection.execute(
                    "SELECT COUNT(*) FROM takes WHERE locked=1 AND id IN "
                    "(SELECT current_take_id FROM script_lines WHERE archived=0 AND voice_profile_id=?)",
                    (profile_id,),
                ).fetchone()[0]
                if int(locked):
                    raise WorkerProtocolError("该音色仍被锁定成片使用，请先解锁对应 take 再更新参考")
            connection.execute(
                """
                INSERT INTO voice_profiles(
                    id,name,version,reference_asset_id,transcript,language,tags_json,
                    authorization_notes,quality_json,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name, version=excluded.version,
                    reference_asset_id=excluded.reference_asset_id,
                    transcript=excluded.transcript, language=excluded.language,
                    tags_json=excluded.tags_json,
                    authorization_notes=excluded.authorization_notes,
                    quality_json=excluded.quality_json, updated_at=excluded.updated_at
                """,
                (
                    profile_id,
                    name,
                    int(value.get("version") or 1),
                    reference_asset_id,
                    str(value.get("transcript") or "")[:20_000],
                    str(value.get("language") or "zh")[:16],
                    _json(value.get("tags") or []),
                    str(value.get("authorization_notes") or "")[:2_000],
                    _json(value.get("quality") or {}),
                    created,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM voice_profiles WHERE id=?", (profile_id,)
            ).fetchone()
            if voice_changed:
                connection.execute(
                    "UPDATE casting_roles SET status='pending',representative_take_id=NULL,confirmed_at=NULL,updated_at=? "
                    "WHERE voice_profile_id=?",
                    (now, profile_id),
                )
                connection.execute(
                    "UPDATE takes SET status='candidate',locked=0 WHERE line_id IN "
                    "(SELECT id FROM script_lines WHERE archived=0 AND voice_profile_id=?)",
                    (profile_id,),
                )
                connection.execute(
                    "UPDATE jobs SET status='cancelled',error='角色音色参考已更新',updated_at=? "
                    "WHERE line_id IN (SELECT id FROM script_lines WHERE archived=0 AND voice_profile_id=?) "
                    "AND status IN ('queued','validating','running','interrupted')",
                    (now, profile_id),
                )
                connection.execute(
                    "UPDATE script_lines SET current_take_id=NULL,status='draft',dirty=1,updated_at=? "
                    "WHERE archived=0 AND voice_profile_id=?",
                    (now, profile_id),
                )
        self._touch("voice_profile", profile_id, "upserted", {"name": name})
        return _voice_dict(row)

    def list_voice_profiles(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM voice_profiles ORDER BY name COLLATE NOCASE, updated_at DESC"
            ).fetchall()
        return [_voice_dict(row) for row in rows]

    def delete_voice_profile(self, profile_id: str) -> dict[str, Any]:
        """Delete a project voice without deleting its imported source asset.

        Generated audio is kept as recoverable project media, but any active line
        mapping is cleared because it can no longer be reproduced from the deleted
        voice profile.
        """
        voice_id = str(profile_id or "").strip()
        if not voice_id:
            raise WorkerProtocolError("缺少要删除的项目声音")
        # Voice deletion clears reproducibility metadata and any timeline clips
        # that reference its adopted takes.  Preserve the complete database and
        # manifest first so hand-edited timing/gain/fades remain recoverable.
        self.get_voice_profile(voice_id)
        backup = self.create_script_backup(reason="before_voice_profile_delete")
        now = utc_now()
        with self.connection() as connection:
            # Serialize against queue claims.  Either deletion cancels the
            # still-queued job first, or a claimed validating/running job makes
            # deletion fail safely; a cancelled job can never be resurrected.
            connection.execute("BEGIN IMMEDIATE")
            voice = connection.execute(
                "SELECT * FROM voice_profiles WHERE id=?", (voice_id,)
            ).fetchone()
            if voice is None:
                raise WorkerProtocolError(f"角色音色不存在：{voice_id}")
            line_rows = connection.execute(
                "SELECT id,speaker FROM script_lines WHERE archived=0 AND voice_profile_id=?",
                (voice_id,),
            ).fetchall()
            active = connection.execute(
                "SELECT COUNT(*) FROM jobs WHERE line_id IN "
                "(SELECT id FROM script_lines WHERE archived=0 AND voice_profile_id=?) "
                "AND status IN ('validating','running')",
                (voice_id,),
            ).fetchone()[0]
            if int(active):
                raise WorkerProtocolError("该声音仍有正在生成的任务，请等待或取消后再删除")
            locked = connection.execute(
                "SELECT COUNT(*) FROM takes WHERE locked=1 AND id IN "
                "(SELECT current_take_id FROM script_lines WHERE archived=0 AND voice_profile_id=?)",
                (voice_id,),
            ).fetchone()[0]
            if int(locked):
                raise WorkerProtocolError("该声音仍被锁定成片使用，请先解锁对应采用版本再删除")
            connection.execute(
                "UPDATE jobs SET status='cancelled',error='项目声音已删除',updated_at=? "
                "WHERE line_id IN (SELECT id FROM script_lines WHERE archived=0 AND voice_profile_id=?) "
                "AND status IN ('queued','interrupted')",
                (now, voice_id),
            )
            connection.execute(
                "UPDATE takes SET status='candidate',locked=0 WHERE line_id IN "
                "(SELECT id FROM script_lines WHERE archived=0 AND voice_profile_id=?)",
                (voice_id,),
            )
            removed_timeline_clips = int(connection.execute(
                "DELETE FROM timeline_clips WHERE line_id IN "
                "(SELECT id FROM script_lines WHERE archived=0 AND voice_profile_id=?)",
                (voice_id,),
            ).rowcount)
            connection.execute(
                "UPDATE script_lines SET voice_profile_id=NULL,current_take_id=NULL,status='draft',dirty=1,updated_at=? "
                "WHERE archived=0 AND voice_profile_id=?",
                (now, voice_id),
            )
            connection.execute(
                "UPDATE casting_roles SET voice_profile_id=NULL,voice_signature='',representative_take_id=NULL,"
                "status='pending',confirmed_at=NULL,updated_at=? WHERE voice_profile_id=?",
                (now, voice_id),
            )
            connection.execute("DELETE FROM voice_profiles WHERE id=?", (voice_id,))
        affected_speakers = sorted({str(row["speaker"] or "旁白") for row in line_rows})
        result = {
            "id": voice_id,
            "name": str(voice["name"]),
            "affected_lines": len(line_rows),
            "affected_speakers": affected_speakers,
            "removed_timeline_clips": removed_timeline_clips,
            "source_asset_preserved": bool(voice["reference_asset_id"]),
            "backup_id": backup["id"],
            "backup_path": backup["path"],
        }
        self._touch("voice_profile", voice_id, "deleted", result)
        return result

    def get_voice_profile(self, profile_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM voice_profiles WHERE id=?", (profile_id,)
            ).fetchone()
        if row is None:
            raise WorkerProtocolError(f"角色音色不存在：{profile_id}")
        return _voice_dict(row)

    def voice_profile_readiness(
        self, profile_id: str, *, verify_hash: bool = False
    ) -> dict[str, Any]:
        """Check that a mapped voice has every input required by generation."""
        voice = self.get_voice_profile(profile_id)
        issues: list[str] = []
        asset_id = str(voice.get("reference_asset_id") or "").strip()
        if not asset_id:
            issues.append("缺少参考音频")
        if not str(voice.get("transcript") or "").strip():
            issues.append("缺少参考音频逐字稿")
        reference_path = ""
        if asset_id:
            try:
                reference_path = str(
                    self.resolve_asset_path(asset_id, verify_hash=verify_hash)
                )
            except WorkerProtocolError as exc:
                issues.append(str(exc))
        return {
            "id": str(voice["id"]),
            "name": str(voice["name"]),
            "ready": not issues,
            "issues": list(dict.fromkeys(issues)),
            "reference_path": reference_path,
        }

    def list_casting_roles(self) -> list[dict[str, Any]]:
        """Return the active script role matrix and its persisted director approval."""
        with self.connection() as connection:
            lines = connection.execute(
                "SELECT * FROM script_lines WHERE archived=0 ORDER BY order_index,id"
            ).fetchall()
            casting = {
                str(row["speaker"]): row
                for row in connection.execute("SELECT * FROM casting_roles").fetchall()
            }
            voices = {
                str(row["id"]): row
                for row in connection.execute("SELECT * FROM voice_profiles").fetchall()
            }
            takes = connection.execute(
                "SELECT * FROM takes ORDER BY created_at DESC,id DESC"
            ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for line in lines:
            grouped.setdefault(str(line["speaker"] or "旁白"), []).append(line)
        takes_by_line: dict[str, list[sqlite3.Row]] = {}
        takes_by_id = {str(row["id"]): row for row in takes}
        for take in takes:
            takes_by_line.setdefault(str(take["line_id"]), []).append(take)
        roles: list[dict[str, Any]] = []
        for speaker, role_lines in grouped.items():
            mapped_ids = {str(row["voice_profile_id"] or "") for row in role_lines}
            mapped_voice_id = next(iter(mapped_ids)) if len(mapped_ids) == 1 else ""
            mapped_voice = voices.get(mapped_voice_id)
            voice_readiness = (
                self.voice_profile_readiness(mapped_voice_id, verify_hash=False)
                if mapped_voice is not None
                else {"ready": False, "issues": []}
            )
            saved = casting.get(speaker)
            active_ids = {str(row["id"]) for row in role_lines}
            representative_line_id = (
                str(saved["representative_line_id"] or "")
                if saved is not None and str(saved["representative_line_id"] or "") in active_ids
                else str(role_lines[0]["id"])
            )
            representative_line = next(
                row for row in role_lines if str(row["id"]) == representative_line_id
            )
            saved_take = (
                takes_by_id.get(str(saved["representative_take_id"] or ""))
                if saved is not None else None
            )
            representative_take = saved_take
            if representative_take is None or str(representative_take["line_id"]) != representative_line_id:
                representative_take = next(
                    (
                        take for take in takes_by_line.get(representative_line_id, [])
                        if str(take["status"]) != "rejected"
                        and (
                            not mapped_voice_id
                            or str(take["voice_profile_id"] or "") == mapped_voice_id
                        )
                        and (
                            mapped_voice is None
                            or str(take["created_at"]) >= str(mapped_voice["updated_at"])
                        )
                    ),
                    None,
                )
            signature = _casting_voice_signature(mapped_voice)
            confirmed = bool(
                saved is not None
                and str(saved["status"]) == "confirmed"
                and mapped_voice_id
                and str(saved["voice_profile_id"] or "") == mapped_voice_id
                and str(saved["voice_signature"] or "") == signature
                and mapped_voice is not None
                and bool(voice_readiness["ready"])
            )
            issues: list[str] = []
            if len(mapped_ids) != 1 or not mapped_voice_id:
                issues.append("该角色存在未映射或不一致的音色")
            elif mapped_voice is None:
                issues.append("映射的音色档案不存在")
            elif not voice_readiness["ready"]:
                issues.extend(str(value) for value in voice_readiness["issues"])
            if not confirmed:
                issues.append("导演尚未确认")
            take_value = None
            if representative_take is not None:
                take_value = _take_dict(representative_take)
                take_path = safe_project_path(self.root, representative_take["rel_path"])
                take_value["path"] = str(take_path)
                take_value["exists"] = take_path.is_file()
            roles.append(
                {
                    "speaker": speaker,
                    "line_count": len(role_lines),
                    "scenes": sorted({str(row["scene"] or "") for row in role_lines if row["scene"]}),
                    "voice_profile_id": mapped_voice_id or None,
                    "voice_name": None if mapped_voice is None else str(mapped_voice["name"]),
                    "voice_quality": {} if mapped_voice is None else _read_json(mapped_voice["quality_json"], {}),
                    "voice_ready": bool(voice_readiness["ready"]),
                    "representative_line_id": representative_line_id,
                    "representative_text": str(representative_line["text"]),
                    "representative_take": take_value,
                    "representative_optional": True,
                    "confirmed": confirmed,
                    "status": "confirmed" if confirmed else "pending",
                    "notes": "" if saved is None else str(saved["notes"] or ""),
                    "issues": list(dict.fromkeys(issues)),
                }
            )
        return roles

    def casting_readiness(self, speakers: Iterable[str] | None = None) -> dict[str, Any]:
        requested = {str(value) for value in (speakers or []) if str(value)}
        roles = [
            role for role in self.list_casting_roles()
            if not requested or role["speaker"] in requested
        ]
        issues = [
            {"speaker": role["speaker"], "issues": role["issues"]}
            for role in roles if not role["confirmed"]
        ]
        return {
            "ok": bool(roles) and not issues,
            "role_count": len(roles),
            "confirmed_count": sum(bool(role["confirmed"]) for role in roles),
            "issues": issues,
            "roles": roles,
        }

    def map_casting_role(self, speaker: str, voice_profile_id: str) -> dict[str, Any]:
        role = str(speaker or "").strip()[:120]
        voice_id = str(voice_profile_id or "").strip()
        if not role or not voice_id:
            raise WorkerProtocolError("角色和音色档案不能为空")
        self.get_voice_profile(voice_id)
        now = utc_now()
        with self.connection() as connection:
            lines = connection.execute(
                "SELECT * FROM script_lines WHERE archived=0 AND speaker=? ORDER BY order_index,id",
                (role,),
            ).fetchall()
            if not lines:
                raise WorkerProtocolError(f"脚本中没有角色：{role}")
            line_ids = [str(line["id"]) for line in lines]
            placeholders = ",".join("?" for _ in line_ids)
            active = connection.execute(
                f"SELECT COUNT(*) FROM jobs WHERE line_id IN ({placeholders}) "
                "AND status IN ('validating','running')",
                line_ids,
            ).fetchone()[0]
            if int(active):
                raise WorkerProtocolError("该角色仍有正在生成的任务，请等待当前任务结束后再更换音色")
            locked = connection.execute(
                f"SELECT COUNT(*) FROM takes WHERE id IN (SELECT current_take_id FROM script_lines "
                f"WHERE id IN ({placeholders})) AND locked=1",
                line_ids,
            ).fetchone()[0]
            if int(locked):
                raise WorkerProtocolError("该角色存在锁定的采用 take，请先解锁再更换音色")
            connection.execute(
                f"UPDATE takes SET status='candidate',locked=0 WHERE line_id IN ({placeholders})",
                line_ids,
            )
            connection.execute(
                f"UPDATE jobs SET status='cancelled',error='角色音色映射已变化',updated_at=? "
                f"WHERE line_id IN ({placeholders}) AND status IN ('queued','validating','running','interrupted')",
                [now, *line_ids],
            )
            connection.execute(
                f"UPDATE script_lines SET voice_profile_id=?,current_take_id=NULL,status='draft',dirty=1,updated_at=? "
                f"WHERE id IN ({placeholders})",
                [voice_id, now, *line_ids],
            )
            connection.execute(
                """
                INSERT INTO casting_roles(
                    speaker,voice_profile_id,voice_signature,representative_line_id,
                    representative_take_id,status,notes,confirmed_at,updated_at
                ) VALUES(?,?,?,?,?,'pending','',NULL,?)
                ON CONFLICT(speaker) DO UPDATE SET
                    voice_profile_id=excluded.voice_profile_id,
                    voice_signature='',representative_line_id=excluded.representative_line_id,
                    representative_take_id=NULL,status='pending',confirmed_at=NULL,
                    updated_at=excluded.updated_at
                """,
                (role, voice_id, "", line_ids[0], None, now),
            )
        self._touch("casting_role", role, "mapped", {"voice_profile_id": voice_id})
        return next(value for value in self.list_casting_roles() if value["speaker"] == role)

    def confirm_casting_role(
        self,
        speaker: str,
        *,
        representative_take_id: str | None = None,
        confirmed: bool = True,
        notes: str = "",
    ) -> dict[str, Any]:
        role = str(speaker or "").strip()[:120]
        if not role:
            raise WorkerProtocolError("角色不能为空")
        if not confirmed:
            with self.connection() as connection:
                active = connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE line_id IN "
                    "(SELECT id FROM script_lines WHERE archived=0 AND speaker=?) "
                    "AND status IN ('validating','running')",
                    (role,),
                ).fetchone()[0]
                if int(active):
                    raise WorkerProtocolError("该角色仍有正在生成的任务，暂不能撤销选角确认")
                connection.execute(
                    "UPDATE casting_roles SET status='pending',representative_take_id=NULL,notes=?,confirmed_at=NULL,updated_at=? WHERE speaker=?",
                    (str(notes)[:2_000], utc_now(), role),
                )
                connection.execute(
                    "UPDATE jobs SET status='cancelled',error='角色选角确认已撤销',updated_at=? "
                    "WHERE line_id IN (SELECT id FROM script_lines WHERE archived=0 AND speaker=?) "
                    "AND status IN ('queued','interrupted')",
                    (utc_now(), role),
                )
            self._touch("casting_role", role, "revoked", {})
            return next(value for value in self.list_casting_roles() if value["speaker"] == role)
        now = utc_now()
        take_id = str(representative_take_id or "").strip()
        with self.connection() as connection:
            lines = connection.execute(
                "SELECT * FROM script_lines WHERE archived=0 AND speaker=? ORDER BY order_index,id",
                (role,),
            ).fetchall()
            if not lines:
                raise WorkerProtocolError(f"脚本中没有角色：{role}")
            voice_ids = {str(line["voice_profile_id"] or "") for line in lines}
            if len(voice_ids) != 1 or not next(iter(voice_ids)):
                raise WorkerProtocolError("该角色的台词尚未统一映射音色")
            voice_id = next(iter(voice_ids))
            voice = connection.execute(
                "SELECT * FROM voice_profiles WHERE id=?", (voice_id,)
            ).fetchone()
            if voice is None:
                raise WorkerProtocolError("映射的音色档案不存在")
            readiness = self.voice_profile_readiness(voice_id, verify_hash=True)
            if not readiness["ready"]:
                raise WorkerProtocolError(
                    f"声音“{readiness['name']}”尚未准备好："
                    + "；".join(readiness["issues"])
                )
            representative_line_id = str(lines[0]["id"])
            if take_id:
                take = connection.execute("SELECT * FROM takes WHERE id=?", (take_id,)).fetchone()
                if take is None:
                    raise WorkerProtocolError("所选代表句不存在")
                line_by_id = {str(line["id"]): line for line in lines}
                if str(take["line_id"]) not in line_by_id:
                    raise WorkerProtocolError("代表句不属于该角色")
                if str(take["voice_profile_id"] or "") != voice_id:
                    raise WorkerProtocolError("代表句不是由当前映射声音生成")
                if str(take["created_at"]) < str(voice["updated_at"]):
                    raise WorkerProtocolError("声音参考已更新，请重新生成代表句后再使用该试听版本")
                if str(take["status"]) == "rejected":
                    raise WorkerProtocolError("已拒绝的代表句不能用于选角确认")
                self._adopt_take_in_connection(connection, take_id)
                representative_line_id = str(take["line_id"])
            signature = _casting_voice_signature(voice)
            connection.execute(
                """
                INSERT INTO casting_roles(
                    speaker,voice_profile_id,voice_signature,representative_line_id,
                    representative_take_id,status,notes,confirmed_at,updated_at
                ) VALUES(?,?,?,?,?,'confirmed',?,?,?)
                ON CONFLICT(speaker) DO UPDATE SET
                    voice_profile_id=excluded.voice_profile_id,
                    voice_signature=excluded.voice_signature,
                    representative_line_id=excluded.representative_line_id,
                    representative_take_id=excluded.representative_take_id,
                    status='confirmed',notes=excluded.notes,
                    confirmed_at=excluded.confirmed_at,updated_at=excluded.updated_at
                """,
                (
                    role,
                    voice_id,
                    signature,
                    representative_line_id,
                    take_id or None,
                    str(notes)[:2_000],
                    now,
                    now,
                ),
            )
        self._touch(
            "casting_role",
            role,
            "confirmed",
            {"take_id": take_id or None, "confirmation_basis": "voice_selection"},
        )
        return next(value for value in self.list_casting_roles() if value["speaker"] == role)

    def replace_script_lines(self, lines: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = [_normalize_line(index, value) for index, value in enumerate(lines)]
        now = utc_now()
        with self.connection() as connection:
            connection.execute("DELETE FROM casting_roles")
            connection.execute("DELETE FROM jobs")
            connection.execute("DELETE FROM timeline_clips")
            connection.execute("DELETE FROM takes")
            connection.execute("DELETE FROM script_lines")
            for line in normalized:
                connection.execute(
                    """
                    INSERT INTO script_lines(
                        id,order_index,scene,speaker,voice_profile_id,text,language,instruction,director_notes,
                        seed,preset,target_start,target_end,status,current_take_id,dirty,
                        archived,source_key,revision_id,metadata_json,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        line["id"],
                        line["order_index"],
                        line["scene"],
                        line["speaker"],
                        line["voice_profile_id"],
                        line["text"],
                        line["language"],
                        line["instruction"],
                        line["director_notes"],
                        line["seed"],
                        line["preset"],
                        line["target_start"],
                        line["target_end"],
                        line["status"],
                        None,
                        1,
                        0,
                        _script_source_key(line),
                        None,
                        _json(line["metadata"]),
                        now,
                        now,
                    ),
                )
        self._touch("script", None, "replaced", {"line_count": len(normalized)})
        return self.list_script_lines()

    def create_script_backup(self, *, reason: str = "manual") -> dict[str, Any]:
        """Create a portable SQLite/manifest snapshot before a destructive script action."""
        created = utc_now()
        backup_id = (
            f"script-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        target = safe_project_path(self.root, Path("backups") / backup_id)
        target.mkdir(parents=True, exist_ok=False)
        database = target / "project.sqlite"
        manifest = target / "project.json"
        metadata_path = target / "backup.json"
        try:
            with self.connection() as connection:
                backup_connection = sqlite3.connect(database)
                try:
                    connection.backup(backup_connection)
                    backup_connection.commit()
                finally:
                    backup_connection.close()
            shutil.copy2(self.manifest_path, manifest)
            snapshot = self.snapshot()
            metadata = {
                "id": backup_id,
                "kind": "script-safety-backup",
                "reason": str(reason or "manual")[:120],
                "created_at": created,
                "project_id": snapshot["id"],
                "project_name": snapshot["name"],
                "schema_version": snapshot["schema_version"],
                "counts": snapshot.get("counts", {}),
                "database_sha256": content_sha256(database),
                "manifest_sha256": content_sha256(manifest),
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        self._touch("project_backup", backup_id, "created", {"reason": reason})
        return {**metadata, "path": str(target)}

    def list_script_backups(self) -> list[dict[str, Any]]:
        root = safe_project_path(self.root, "backups")
        if not root.is_dir():
            return []
        backups: list[dict[str, Any]] = []
        for metadata_path in root.glob("script-*/backup.json"):
            try:
                value = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if not isinstance(value, dict):
                continue
            database = metadata_path.parent / "project.sqlite"
            manifest = metadata_path.parent / "project.json"
            value["path"] = str(metadata_path.parent)
            value["valid"] = database.is_file() and manifest.is_file()
            backups.append(value)
        return sorted(backups, key=lambda value: str(value.get("created_at") or ""), reverse=True)

    def replace_script_lines_safe(
        self,
        lines: Iterable[dict[str, Any]],
        *,
        confirmation: str,
        source_format: str = "",
        source_name: str = "",
    ) -> dict[str, Any]:
        """Fully replace a script only after an explicit project-bound confirmation and backup."""
        summary = self.summary()
        expected = f"REPLACE:{summary.id}"
        if str(confirmation or "") != expected:
            raise WorkerProtocolError("完整替换需要二次确认；请重新从项目界面发起")
        normalized = [_normalize_line(index, value) for index, value in enumerate(lines)]
        before = self.list_script_lines()
        backup = self.create_script_backup(reason="before_full_script_replace")
        after = self.replace_script_lines(normalized)
        revision_id = str(uuid.uuid4())
        now = utc_now()
        diff = {
            "mode": "replace",
            "counts": {
                "unchanged": 0,
                "modified": 0,
                "added": len(after),
                "archived": len(before),
            },
            "existing_count": len(before),
            "incoming_count": len(after),
            "backup_id": backup["id"],
            "safe": True,
        }
        with self.connection() as connection:
            connection.execute(
                "UPDATE script_lines SET revision_id=?,updated_at=? WHERE archived=0",
                (revision_id, now),
            )
            connection.execute(
                """
                INSERT INTO script_revisions(
                    id,source_format,source_name,mode,diff_json,before_json,after_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    revision_id,
                    str(source_format or "")[:40],
                    str(source_name or "")[:500],
                    "replace",
                    _json(diff),
                    _json(before),
                    _json(after),
                    now,
                ),
            )
        self._touch(
            "script_revision",
            revision_id,
            "fully_replaced",
            {"backup_id": backup["id"], "line_count": len(after)},
        )
        return {
            "revision_id": revision_id,
            "diff": diff,
            "backup": backup,
            "lines": self.list_script_lines(),
        }

    def list_script_lines(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        with self.connection() as connection:
            if include_archived:
                rows = connection.execute(
                    "SELECT * FROM script_lines ORDER BY archived,order_index,id"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM script_lines WHERE archived=0 ORDER BY order_index,id"
                ).fetchall()
        return [_line_dict(row) for row in rows]

    def restore_archived_lines(self, line_ids: Iterable[str]) -> dict[str, Any]:
        ids = list(dict.fromkeys(str(value) for value in line_ids if str(value)))
        if not ids:
            raise WorkerProtocolError("没有选择要恢复的归档台词")
        if len(ids) > 500:
            raise WorkerProtocolError("单次最多恢复 500 条归档台词")
        placeholders = ",".join("?" for _ in ids)
        revision_id = str(uuid.uuid4())
        now = utc_now()
        before = self.list_script_lines()
        restored: list[str] = []
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM script_lines WHERE archived=1 AND id IN ({placeholders})",
                ids,
            ).fetchall()
            found = {str(row["id"]): row for row in rows}
            missing = [line_id for line_id in ids if line_id not in found]
            if missing:
                raise WorkerProtocolError(f"归档台词不存在：{', '.join(missing[:5])}")
            for line_id in ids:
                row = found[line_id]
                take_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM takes WHERE line_id=?", (line_id,)
                    ).fetchone()[0]
                )
                adopted = None
                if row["current_take_id"]:
                    adopted = connection.execute(
                        "SELECT status FROM takes WHERE id=?", (row["current_take_id"],)
                    ).fetchone()
                status = (
                    "completed"
                    if adopted is not None and str(adopted["status"]) == "adopted"
                    else "review"
                    if take_count
                    else "draft"
                )
                connection.execute(
                    "UPDATE script_lines SET archived=0,status=?,dirty=1,revision_id=?,updated_at=? "
                    "WHERE id=?",
                    (status, revision_id, now, line_id),
                )
                restored.append(line_id)
            # Archived clips keep their old sequence value.  After the active
            # timeline has been reordered that value may collide with another
            # clip; normalize inside the same transaction before exposing it.
            _normalize_timeline_sequence(connection)
            after_rows = connection.execute(
                "SELECT * FROM script_lines WHERE archived=0 ORDER BY order_index,id"
            ).fetchall()
            after = [_line_dict(row) for row in after_rows]
            diff = {
                "mode": "restore",
                "counts": {"restored": len(restored)},
                "line_ids": restored,
                "safe": True,
            }
            connection.execute(
                """
                INSERT INTO script_revisions(
                    id,source_format,source_name,mode,diff_json,before_json,after_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (revision_id, "archive", "archive-restore", "restore", _json(diff), _json(before), _json(after), now),
            )
        self._touch("script_revision", revision_id, "archive_restored", {"line_ids": restored})
        return {
            "revision_id": revision_id,
            "restored": len(restored),
            "line_ids": restored,
            "lines": self.list_script_lines(),
        }

    def list_pronunciation_entries(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM pronunciation_entries ORDER BY length(display_text) DESC,display_text,id"
            ).fetchall()
        return [_pronunciation_dict(row) for row in rows]

    def upsert_pronunciation_entry(self, value: dict[str, Any]) -> dict[str, Any]:
        entry_id = str(value.get("id") or uuid.uuid4())
        display_text = str(value.get("display_text") or "").strip()[:500]
        spoken_text = str(value.get("spoken_text") or "").strip()[:1_000]
        if not display_text or not spoken_text:
            raise WorkerProtocolError("发音词表的显示文本和朗读文本不能为空")
        language = str(value.get("language") or "all").strip().lower()[:16] or "all"
        if language not in {"all", "zh", "en"}:
            raise WorkerProtocolError("发音词表语言必须是 all/zh/en")
        case_sensitive = 1 if value.get("case_sensitive") else 0
        notes = str(value.get("notes") or "")[:2_000]
        now = utc_now()
        with self.connection() as connection:
            existing = connection.execute(
                "SELECT id FROM pronunciation_entries WHERE display_text=? AND language=? AND id<>?",
                (display_text, language, entry_id),
            ).fetchone()
            if existing is not None:
                entry_id = str(existing["id"])
            connection.execute(
                """
                INSERT INTO pronunciation_entries(
                    id,display_text,spoken_text,language,case_sensitive,notes,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    display_text=excluded.display_text,spoken_text=excluded.spoken_text,
                    language=excluded.language,case_sensitive=excluded.case_sensitive,
                    notes=excluded.notes,updated_at=excluded.updated_at
                """,
                (entry_id, display_text, spoken_text, language, case_sensitive, notes, now, now),
            )
            row = connection.execute(
                "SELECT * FROM pronunciation_entries WHERE id=?", (entry_id,)
            ).fetchone()
        self._touch("pronunciation_entry", entry_id, "upserted", {"display_text": display_text})
        return _pronunciation_dict(row)

    def delete_pronunciation_entry(self, entry_id: str) -> bool:
        target = str(entry_id or "").strip()
        with self.connection() as connection:
            cursor = connection.execute("DELETE FROM pronunciation_entries WHERE id=?", (target,))
            deleted = cursor.rowcount > 0
        if deleted:
            self._touch("pronunciation_entry", target, "deleted", {})
        return deleted

    def pronunciation_preview(self, text: str, *, language: str = "zh") -> dict[str, Any]:
        return _apply_pronunciation_entries(
            str(text or ""),
            str(language or "zh"),
            self.list_pronunciation_entries(),
        )

    def upsert_production_clip(self, value: dict[str, Any]) -> dict[str, Any]:
        clip_id = str(value.get("id") or uuid.uuid4())
        asset_id = str(value.get("asset_id") or "").strip()
        if not asset_id:
            raise WorkerProtocolError("制作轨片段缺少素材")
        kind = str(value.get("kind") or "music").strip().lower()
        if kind not in {"music", "ambience", "room_tone"}:
            raise WorkerProtocolError("制作轨类型必须是 music/ambience/room_tone")
        position = max(0.0, float(value.get("position") or 0.0))
        duration = max(0.0, float(value.get("duration") or 0.0))
        gain_db = max(-60.0, min(12.0, float(value.get("gain_db") or 0.0)))
        fade_in = max(0.0, float(value.get("fade_in") or 0.0))
        fade_out = max(0.0, float(value.get("fade_out") or 0.0))
        ducking_db = max(-30.0, min(0.0, float(value.get("ducking_db") or 0.0)))
        loop = 1 if value.get("loop") else 0
        fill_gaps = 1 if value.get("fill_gaps") else 0
        muted = 1 if value.get("muted") else 0
        if fill_gaps and kind != "room_tone":
            raise WorkerProtocolError("自动补空隙只适用于 room_tone")
        if fill_gaps:
            loop = 1
        if loop and duration <= 0:
            raise WorkerProtocolError("循环制作轨必须填写目标时长")
        now = utc_now()
        with self.connection() as connection:
            if connection.execute("SELECT 1 FROM assets WHERE id=?", (asset_id,)).fetchone() is None:
                raise WorkerProtocolError(f"制作轨素材不存在：{asset_id}")
            connection.execute(
                """
                INSERT INTO production_clips(
                    id,asset_id,kind,position,duration,gain_db,fade_in,fade_out,
                    ducking_db,loop,fill_gaps,muted,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    asset_id=excluded.asset_id,kind=excluded.kind,position=excluded.position,
                    duration=excluded.duration,gain_db=excluded.gain_db,
                    fade_in=excluded.fade_in,fade_out=excluded.fade_out,
                    ducking_db=excluded.ducking_db,loop=excluded.loop,
                    fill_gaps=excluded.fill_gaps,muted=excluded.muted,
                    updated_at=excluded.updated_at
                """,
                (
                    clip_id, asset_id, kind, position, duration, gain_db, fade_in, fade_out,
                    ducking_db, loop, fill_gaps, muted, now, now,
                ),
            )
            row = connection.execute("SELECT * FROM production_clips WHERE id=?", (clip_id,)).fetchone()
        self._touch("production_clip", clip_id, "upserted", {"kind": kind})
        return self._production_clip_dict(row)

    def list_production_clips(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM production_clips ORDER BY position,kind,created_at,id"
            ).fetchall()
        return [self._production_clip_dict(row) for row in rows]

    def _production_clip_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["loop"] = bool(value.get("loop", 0))
        value["fill_gaps"] = bool(value.get("fill_gaps", 0))
        value["muted"] = bool(value.get("muted", 0))
        try:
            value["path"] = str(self.resolve_asset_path(str(value["asset_id"])))
            value["exists"] = True
        except WorkerProtocolError:
            value["path"] = ""
            value["exists"] = False
        return value

    def delete_production_clip(self, clip_id: str) -> bool:
        target = str(clip_id or "").strip()
        with self.connection() as connection:
            cursor = connection.execute("DELETE FROM production_clips WHERE id=?", (target,))
            deleted = cursor.rowcount > 0
        if deleted:
            self._touch("production_clip", target, "deleted", {})
        return deleted

    def production_render_inputs(self) -> list[dict[str, Any]]:
        inputs: list[dict[str, Any]] = []
        for index, clip in enumerate(self.list_production_clips()):
            if clip["muted"]:
                continue
            if not clip["exists"]:
                raise WorkerProtocolError(f"制作轨素材缺失：{clip['id']}")
            inputs.append(
                {
                    "id": f"production-{clip['id']}",
                    "production_clip_id": clip["id"],
                    "order_index": 1_000_000 + index,
                    "path": clip["path"],
                    "track": clip["kind"],
                    "kind": clip["kind"],
                    "position": clip["position"],
                    "duration": clip["duration"],
                    "in_offset": 0.0,
                    "out_offset": 0.0,
                    "gain_db": clip["gain_db"],
                    "fade_in": clip["fade_in"],
                    "fade_out": clip["fade_out"],
                    "ducking_db": clip["ducking_db"],
                    "loop": clip["loop"],
                    "fill_gaps": clip["fill_gaps"],
                    "muted": False,
                }
            )
        return inputs

    def preview_script_merge(self, lines: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Describe a deterministic, non-destructive script merge without writing."""
        normalized = [_normalize_line(index, value) for index, value in enumerate(lines)]
        existing = self.list_script_lines()
        matches = _match_script_lines(existing, normalized)
        existing_by_id = {str(line["id"]): line for line in existing}
        matched_existing = set(matches.values())
        changes: list[dict[str, Any]] = []
        counts = {"unchanged": 0, "modified": 0, "added": 0, "archived": 0}
        compare_fields = (
            "order_index",
            "scene",
            "speaker",
            "text",
            "language",
            "instruction",
            "seed",
            "preset",
            "target_start",
            "target_end",
        )
        for index, incoming in enumerate(normalized):
            matched_id = matches.get(index)
            if not matched_id:
                counts["added"] += 1
                changes.append(
                    {
                        "kind": "added",
                        "incoming_index": index,
                        "line_id": incoming["id"],
                        "speaker": incoming["speaker"],
                        "text": incoming["text"],
                        "fields": [],
                    }
                )
                continue
            current = existing_by_id[matched_id]
            fields = [field for field in compare_fields if current.get(field) != incoming.get(field)]
            kind = "modified" if fields else "unchanged"
            counts[kind] += 1
            changes.append(
                {
                    "kind": kind,
                    "incoming_index": index,
                    "line_id": matched_id,
                    "speaker": incoming["speaker"],
                    "text": incoming["text"],
                    "fields": fields,
                }
            )
        for current in existing:
            if current["id"] in matched_existing:
                continue
            counts["archived"] += 1
            changes.append(
                {
                    "kind": "archived",
                    "incoming_index": None,
                    "line_id": current["id"],
                    "speaker": current["speaker"],
                    "text": current["text"],
                    "fields": [],
                }
            )
        return {
            "mode": "merge",
            "counts": counts,
            "changes": changes,
            "safe": True,
            "destructive_deletes": 0,
            "existing_count": len(existing),
            "incoming_count": len(normalized),
        }

    def preview_script_replace(self, lines: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Describe the explicitly destructive full-replacement path without writing."""
        normalized = [_normalize_line(index, value) for index, value in enumerate(lines)]
        existing = self.list_script_lines()
        changes = [
            {
                "kind": "archived",
                "incoming_index": None,
                "line_id": line["id"],
                "speaker": line["speaker"],
                "text": line["text"],
                "fields": [],
            }
            for line in existing
        ]
        changes.extend(
            {
                "kind": "added",
                "incoming_index": index,
                "line_id": line["id"],
                "speaker": line["speaker"],
                "text": line["text"],
                "fields": [],
            }
            for index, line in enumerate(normalized)
        )
        return {
            "mode": "replace",
            "counts": {
                "unchanged": 0,
                "modified": 0,
                "added": len(normalized),
                "archived": len(existing),
            },
            "changes": changes,
            "safe": True,
            "requires_backup": True,
            "requires_confirmation": True,
            "destructive_deletes": len(existing),
            "existing_count": len(existing),
            "incoming_count": len(normalized),
        }

    def merge_script_lines(
        self,
        lines: Iterable[dict[str, Any]],
        *,
        source_format: str = "",
        source_name: str = "",
    ) -> dict[str, Any]:
        """Merge a script revision while preserving all prior takes and timeline edits."""
        normalized = [_normalize_line(index, value) for index, value in enumerate(lines)]
        preview = self.preview_script_merge(normalized)
        matches = {
            int(change["incoming_index"]): str(change["line_id"])
            for change in preview["changes"]
            if change["kind"] in {"unchanged", "modified"}
            and change["incoming_index"] is not None
        }
        revision_id = str(uuid.uuid4())
        now = utc_now()
        generation_fields = {
            "speaker",
            "voice_profile_id",
            "text",
            "language",
            "instruction",
            "seed",
            "preset",
        }
        with self.connection() as connection:
            before_rows = connection.execute(
                "SELECT * FROM script_lines WHERE archived=0 ORDER BY order_index,id"
            ).fetchall()
            before = [_line_dict(row) for row in before_rows]
            existing_by_id = {str(line["id"]): line for line in before}
            voice_rows = connection.execute("SELECT id,name FROM voice_profiles").fetchall()
            voices_by_name = {str(row["name"]).casefold(): str(row["id"]) for row in voice_rows}
            matched_existing: set[str] = set()
            for index, incoming in enumerate(normalized):
                matched_id = matches.get(index)
                if matched_id:
                    current = existing_by_id[matched_id]
                    matched_existing.add(matched_id)
                    voice_profile_id = incoming.get("voice_profile_id") or current.get(
                        "voice_profile_id"
                    )
                    if incoming["speaker"] != current["speaker"]:
                        voice_profile_id = voices_by_name.get(incoming["speaker"].casefold())
                    values = {**incoming, "id": matched_id, "voice_profile_id": voice_profile_id}
                    changed = {
                        field
                        for field in (
                            "order_index",
                            "scene",
                            "speaker",
                            "voice_profile_id",
                            "text",
                            "language",
                            "instruction",
                            "seed",
                            "preset",
                            "target_start",
                            "target_end",
                        )
                        if current.get(field) != values.get(field)
                    }
                    generation_changed = bool(changed.intersection(generation_fields))
                    timing_changed = bool(changed.intersection({"target_start", "target_end"}))
                    status = "draft" if generation_changed else current["status"]
                    current_take_id = None if generation_changed else current.get("current_take_id")
                    dirty = 1 if changed else int(bool(current.get("dirty")))
                    connection.execute(
                        """
                        UPDATE script_lines SET
                            order_index=?,scene=?,speaker=?,voice_profile_id=?,text=?,language=?,
                            instruction=?,seed=?,preset=?,target_start=?,target_end=?,status=?,
                            current_take_id=?,dirty=?,archived=0,source_key=?,revision_id=?,
                            metadata_json=?,updated_at=? WHERE id=?
                        """,
                        (
                            values["order_index"],
                            values["scene"],
                            values["speaker"],
                            values["voice_profile_id"],
                            values["text"],
                            values["language"],
                            values["instruction"],
                            values["seed"],
                            values["preset"],
                            values["target_start"],
                            values["target_end"],
                            status,
                            current_take_id,
                            dirty,
                            _script_source_key(values),
                            revision_id,
                            _json(values["metadata"]),
                            now,
                            matched_id,
                        ),
                    )
                    if generation_changed:
                        connection.execute(
                            "UPDATE takes SET status='candidate' WHERE line_id=?",
                            (matched_id,),
                        )
                        connection.execute(
                            "UPDATE jobs SET status='cancelled',error='脚本版本已变化',updated_at=? "
                            "WHERE line_id=? AND status IN ('queued','validating','running','interrupted')",
                            (now, matched_id),
                        )
                    elif timing_changed and values["target_start"] is not None:
                        connection.execute(
                            "UPDATE timeline_clips SET position=?,updated_at=? WHERE line_id=?",
                            (float(values["target_start"]), now, matched_id),
                        )
                    continue
                line_id = str(incoming["id"] or uuid.uuid4())
                voice_profile_id = incoming.get("voice_profile_id") or voices_by_name.get(
                    incoming["speaker"].casefold()
                )
                connection.execute(
                    """
                    INSERT INTO script_lines(
                        id,order_index,scene,speaker,voice_profile_id,text,language,instruction,director_notes,
                        seed,preset,target_start,target_end,status,current_take_id,dirty,
                        archived,source_key,revision_id,metadata_json,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        line_id,
                        incoming["order_index"],
                        incoming["scene"],
                        incoming["speaker"],
                        voice_profile_id,
                        incoming["text"],
                        incoming["language"],
                        incoming["instruction"],
                        incoming["director_notes"],
                        incoming["seed"],
                        incoming["preset"],
                        incoming["target_start"],
                        incoming["target_end"],
                        "draft",
                        None,
                        1,
                        0,
                        _script_source_key(incoming),
                        revision_id,
                        _json(incoming["metadata"]),
                        now,
                        now,
                    ),
                )
            for current in before:
                line_id = str(current["id"])
                if line_id in matched_existing:
                    continue
                connection.execute(
                    "UPDATE script_lines SET archived=1,status='archived',revision_id=?,updated_at=? "
                    "WHERE id=?",
                    (revision_id, now, line_id),
                )
                connection.execute(
                    "UPDATE jobs SET status='cancelled',error='台词已在新脚本版本中归档',updated_at=? "
                    "WHERE line_id=? AND status IN ('queued','validating','running','interrupted')",
                    (now, line_id),
                )
            after_rows = connection.execute(
                "SELECT * FROM script_lines WHERE archived=0 ORDER BY order_index,id"
            ).fetchall()
            after = [_line_dict(row) for row in after_rows]
            connection.execute(
                """
                INSERT INTO script_revisions(
                    id,source_format,source_name,mode,diff_json,before_json,after_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    revision_id,
                    str(source_format or "")[:40],
                    str(source_name or "")[:500],
                    "merge",
                    _json(preview),
                    _json(before),
                    _json(after),
                    now,
                ),
            )
        self._touch(
            "script_revision",
            revision_id,
            "merged",
            {"counts": preview["counts"], "source_format": source_format},
        )
        return {"revision_id": revision_id, "diff": preview, "lines": self.list_script_lines()}

    def patch_script_line(self, line_id: str, values: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "scene",
            "speaker",
            "voice_profile_id",
            "text",
            "language",
            "instruction",
            "director_notes",
            "seed",
            "preset",
            "target_start",
            "target_end",
            "status",
        }
        updates = {key: values[key] for key in allowed if key in values}
        if not updates:
            raise WorkerProtocolError("没有可更新的台词字段")
        if "text" in updates and not str(updates["text"] or "").strip():
            raise WorkerProtocolError("台词文本不能为空")
        generation_fields = {
            "speaker",
            "voice_profile_id",
            "text",
            "language",
            "instruction",
            "seed",
            "preset",
        }
        with self.connection() as connection:
            existing = connection.execute(
                "SELECT * FROM script_lines WHERE id=?", (line_id,)
            ).fetchone()
            if existing is None:
                raise WorkerProtocolError(f"台词不存在：{line_id}")
            changed_generation = {
                key for key in generation_fields.intersection(updates)
                if existing[key] != updates[key]
            }
            if changed_generation:
                active = connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE line_id=? AND status IN ('validating','running')",
                    (line_id,),
                ).fetchone()[0]
                if int(active):
                    raise WorkerProtocolError("该句正在生成，请等待当前任务结束后再修改")
                locked = connection.execute(
                    "SELECT locked FROM takes WHERE id=?", (existing["current_take_id"],)
                ).fetchone()
                if locked is not None and bool(locked["locked"]):
                    raise WorkerProtocolError("该句当前 take 已锁定，请先解锁再修改生成内容")
                updates["status"] = "draft"
            assignments = ",".join(f"{key}=?" for key in updates)
            params = list(updates.values()) + [1, utc_now(), line_id]
            cursor = connection.execute(
                f"UPDATE script_lines SET {assignments},dirty=?,updated_at=? WHERE id=?",
                params,
            )
            if changed_generation:
                connection.execute(
                    "UPDATE takes SET status='candidate',locked=0 WHERE line_id=?",
                    (line_id,),
                )
                connection.execute(
                    "UPDATE jobs SET status='cancelled',error='台词生成内容已修改',updated_at=? "
                    "WHERE line_id=? AND status IN ('queued','validating','running','interrupted')",
                    (utc_now(), line_id),
                )
                connection.execute(
                    "UPDATE script_lines SET current_take_id=NULL WHERE id=?", (line_id,)
                )
                speakers = {str(existing["speaker"] or ""), str(updates.get("speaker", existing["speaker"]) or "")}
                for speaker in speakers:
                    if speaker:
                        connection.execute(
                            "UPDATE casting_roles SET status='pending',confirmed_at=NULL,updated_at=? WHERE speaker=?",
                            (utc_now(), speaker),
                        )
            row = connection.execute(
                "SELECT * FROM script_lines WHERE id=?", (line_id,)
            ).fetchone()
        self._touch("script_line", line_id, "updated", {"fields": sorted(updates)})
        return _line_dict(row)

    def enqueue_lines(
        self,
        line_ids: Iterable[str] | None = None,
        *,
        force: bool = False,
        require_casting: bool = False,
    ) -> list[dict[str, Any]]:
        requested = [str(value) for value in (line_ids or []) if str(value)]
        with self.connection() as connection:
            # The active-job check and insert must be one serialized decision;
            # HTTP requests can arrive concurrently even when the UI is guarded.
            connection.execute("BEGIN IMMEDIATE")
            if requested:
                placeholders = ",".join("?" for _ in requested)
                rows = connection.execute(
                    f"SELECT * FROM script_lines WHERE archived=0 AND id IN ({placeholders}) "
                    "ORDER BY order_index",
                    requested,
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM script_lines WHERE archived=0 ORDER BY order_index"
                ).fetchall()
            if require_casting and rows:
                readiness = self.casting_readiness({str(row["speaker"] or "旁白") for row in rows})
                if not readiness["ok"]:
                    names = "、".join(issue["speaker"] for issue in readiness["issues"][:8])
                    raise WorkerProtocolError(f"角色选角尚未确认：{names}")
            now = utc_now()
            pronunciation_entries = [
                _pronunciation_dict(row)
                for row in connection.execute(
                    "SELECT * FROM pronunciation_entries ORDER BY length(display_text) DESC,display_text,id"
                ).fetchall()
            ]
            jobs: list[dict[str, Any]] = []
            for priority, line in enumerate(rows):
                existing_active = connection.execute(
                    "SELECT id FROM jobs WHERE line_id=? "
                    "AND status IN ('queued','validating','running','interrupted') "
                    "ORDER BY updated_at DESC,created_at DESC LIMIT 1",
                    (line["id"],),
                ).fetchone()
                if existing_active is not None:
                    continue
                current_locked = False
                if line["current_take_id"]:
                    take_lock = connection.execute(
                        "SELECT locked FROM takes WHERE id=?", (line["current_take_id"],)
                    ).fetchone()
                    current_locked = take_lock is not None and bool(take_lock["locked"])
                if current_locked:
                    continue
                if not force and line["status"] == "completed" and line["current_take_id"]:
                    continue
                if not force and line["status"] == "review" and connection.execute(
                    "SELECT 1 FROM takes WHERE line_id=? LIMIT 1", (line["id"],)
                ).fetchone() is not None:
                    continue
                job_id = str(uuid.uuid4())
                pronunciation = _apply_pronunciation_entries(
                    str(line["text"]), str(line["language"] or "zh"), pronunciation_entries
                )
                payload = {
                    "line_id": line["id"],
                    "order_index": line["order_index"],
                    "text": line["text"],
                    "display_text": line["text"],
                    "spoken_text": pronunciation["spoken_text"],
                    "pronunciation_matches": pronunciation["matches"],
                    "speaker": line["speaker"],
                    "voice_profile_id": line["voice_profile_id"],
                    "language": line["language"],
                    "instruction": line["instruction"],
                    "seed": line["seed"],
                    "quality_preset": line["preset"],
                    "target_start": line["target_start"],
                    "target_end": line["target_end"],
                }
                connection.execute(
                    """
                    INSERT INTO jobs(
                        id,line_id,take_id,stage,status,payload_json,checkpoint_rel_path,
                        attempts,error,priority,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        job_id,
                        line["id"],
                        None,
                        "generate",
                        "queued",
                        _json(payload),
                        None,
                        0,
                        None,
                        priority,
                        now,
                        now,
                    ),
                )
                connection.execute(
                    "UPDATE script_lines SET status='queued',updated_at=? WHERE id=?",
                    (now, line["id"]),
                )
                jobs.append({"id": job_id, "status": "queued", "payload": payload})
        self._touch("queue", None, "enqueued", {"count": len(jobs)})
        return jobs

    def list_jobs(self, statuses: Iterable[str] | None = None) -> list[dict[str, Any]]:
        requested = [str(value) for value in (statuses or [])]
        with self.connection() as connection:
            if requested:
                placeholders = ",".join("?" for _ in requested)
                rows = connection.execute(
                    f"SELECT * FROM jobs WHERE status IN ({placeholders}) "
                    "ORDER BY priority,created_at",
                    requested,
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM jobs ORDER BY priority,created_at"
                ).fetchall()
        return [_job_dict(row) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise WorkerProtocolError(f"任务不存在：{job_id}")
        return _job_dict(row)

    def claim_queued_job(self, job_id: str) -> dict[str, Any] | None:
        """Atomically move one queued job into validation, or return ``None``."""
        target = str(job_id or "").strip()
        if not target:
            raise WorkerProtocolError("缺少要领取的任务")
        now = utc_now()
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE jobs SET status='validating',stage='generate',error='',updated_at=? "
                "WHERE id=? AND status='queued'",
                (now, target),
            )
            if cursor.rowcount != 1:
                return None
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (target,)).fetchone()
            if row is None:
                raise WorkerProtocolError(f"任务不存在：{target}")
            if row["line_id"]:
                connection.execute(
                    "UPDATE script_lines SET status='validating',updated_at=? WHERE id=?",
                    (now, row["line_id"]),
                )
        return _job_dict(row)

    def set_job_priority(self, job_id: str, priority: int) -> dict[str, Any]:
        normalized = max(-1_000_000, min(1_000_000, int(priority)))
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET priority=?,updated_at=? WHERE id=?",
                (normalized, utc_now(), job_id),
            )
            if cursor.rowcount != 1:
                raise WorkerProtocolError(f"任务不存在：{job_id}")
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        self._touch("job", job_id, "priority_changed", {"priority": normalized})
        return _job_dict(row)

    def cancel_pending_jobs(self) -> int:
        now = utc_now()
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT line_id FROM jobs WHERE status='queued' AND line_id IS NOT NULL"
            ).fetchall()
            cursor = connection.execute(
                "UPDATE jobs SET status='cancelled',error='用户取消未开始任务',updated_at=? "
                "WHERE status='queued'",
                (now,),
            )
            for row in rows:
                connection.execute(
                    "UPDATE script_lines SET status='cancelled',updated_at=? WHERE id=?",
                    (now, row["line_id"]),
                )
            count = int(cursor.rowcount)
        if count:
            self._touch("queue", None, "pending_cancelled", {"count": count})
        return count

    def request_job_cancellation(self, job_id: str) -> dict[str, Any]:
        """Cancel one queued job or mark one active job for cooperative runtime cancellation."""
        target = str(job_id or "").strip()
        if not target:
            raise WorkerProtocolError("缺少要取消的任务")
        now = utc_now()
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (target,)).fetchone()
            if row is None:
                raise WorkerProtocolError(f"任务不存在：{target}")
            status = str(row["status"])
            if status == "queued":
                connection.execute(
                    "UPDATE jobs SET status='cancelled',error='用户取消单句任务',updated_at=? WHERE id=?",
                    (now, target),
                )
                if row["line_id"]:
                    connection.execute(
                        "UPDATE script_lines SET status='cancelled',updated_at=? WHERE id=?",
                        (now, row["line_id"]),
                    )
                result = {
                    "job_id": target,
                    "status": "cancelled",
                    "cancelled": True,
                    "needs_runtime_cancel": False,
                }
            elif status in {"validating", "running"}:
                requested = self.requested_job_cancellations()
                requested.add(target)
                self._write_job_cancellations(requested)
                result = {
                    "job_id": target,
                    "status": status,
                    "cancelled": False,
                    "needs_runtime_cancel": True,
                }
            else:
                result = {
                    "job_id": target,
                    "status": status,
                    "cancelled": False,
                    "needs_runtime_cancel": False,
                    "reason": f"任务当前状态为 {status}，无需取消",
                }
        self._touch("job", target, "cancellation_requested", {"status": status})
        return result

    def requested_job_cancellations(self) -> set[str]:
        path = safe_project_path(self.root, Path("cache") / "job-cancellations.json")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return set()
        values = value.get("job_ids") if isinstance(value, dict) else None
        if not isinstance(values, list):
            return set()
        return {str(item) for item in values if str(item)}

    def clear_job_cancellations(self, job_ids: Iterable[str]) -> None:
        requested = self.requested_job_cancellations()
        requested.difference_update(str(value) for value in job_ids)
        self._write_job_cancellations(requested)

    def _write_job_cancellations(self, job_ids: set[str]) -> None:
        path = safe_project_path(self.root, Path("cache") / "job-cancellations.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"job_ids": sorted(job_ids), "updated_at": utc_now()}
        temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        _replace_with_retry(temporary, path)

    def queue_control(self) -> dict[str, Any]:
        path = safe_project_path(self.root, Path("cache") / "queue-control.json")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            value = {}
        return {
            "paused": bool(value.get("paused", False)),
            "updated_at": value.get("updated_at"),
        }

    def set_queue_paused(self, paused: bool) -> dict[str, Any]:
        path = safe_project_path(self.root, Path("cache") / "queue-control.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"paused": bool(paused), "updated_at": utc_now()}
        temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        _replace_with_retry(temporary, path)
        self._touch("queue", None, "paused" if paused else "continued", {})
        return value

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        checkpoint_rel_path: str | None = None,
        error: str | None = None,
        take_id: str | None = None,
    ) -> dict[str, Any]:
        if status is not None and status not in JOB_STATES:
            raise WorkerProtocolError(f"未知任务状态：{status}")
        if stage is not None and stage not in JOB_STAGES:
            raise WorkerProtocolError(f"未知任务阶段：{stage}")
        if checkpoint_rel_path:
            safe_project_path(self.root, checkpoint_rel_path)
        values: dict[str, Any] = {"updated_at": utc_now()}
        for key, value in {
            "status": status,
            "stage": stage,
            "checkpoint_rel_path": checkpoint_rel_path,
            "error": error,
            "take_id": take_id,
        }.items():
            if value is not None:
                values[key] = value
        if status == "running":
            values["attempts"] = "__increment__"
        assignments: list[str] = []
        params: list[Any] = []
        for key, value in values.items():
            if value == "__increment__":
                assignments.append(f"{key}={key}+1")
            else:
                assignments.append(f"{key}=?")
                params.append(value)
        params.append(job_id)
        with self.connection() as connection:
            cursor = connection.execute(
                f"UPDATE jobs SET {','.join(assignments)} WHERE id=?", params
            )
            if cursor.rowcount != 1:
                raise WorkerProtocolError(f"任务不存在：{job_id}")
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row["line_id"]:
                line_status = str(row["status"])
                if line_status == "completed" and row["take_id"]:
                    take_row = connection.execute(
                        "SELECT status FROM takes WHERE id=?", (row["take_id"],)
                    ).fetchone()
                    if take_row is not None and str(take_row["status"]) != "adopted":
                        line_status = "review"
                connection.execute(
                    "UPDATE script_lines SET status=?,updated_at=? WHERE id=?",
                    (line_status, utc_now(), row["line_id"]),
                )
        return _job_dict(row)

    def recover_interrupted_jobs(self) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status='interrupted',updated_at=? "
                "WHERE status IN ('validating','running')",
                (utc_now(),),
            )
            count = int(cursor.rowcount)
        return count

    def resume_interrupted_jobs(self) -> int:
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET status='queued',error=NULL,updated_at=? "
                "WHERE status IN ('interrupted','failed')",
                (utc_now(),),
            )
            count = int(cursor.rowcount)
        if count:
            self._touch("queue", None, "resumed", {"count": count})
        return count

    def add_take(
        self,
        line_id: str,
        audio_path: str | Path,
        *,
        parent_take_id: str | None = None,
        seed: int | None = None,
        preset: str = "balanced",
        model_revision: str | None = None,
        quality: dict[str, Any] | None = None,
        adopt: bool = False,
    ) -> dict[str, Any]:
        source = Path(audio_path).expanduser().resolve()
        if not source.is_file():
            raise WorkerProtocolError(f"take 音频不存在：{source}")
        take_id = str(uuid.uuid4())
        target_dir = safe_project_path(self.root, Path("segments") / line_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower() or ".wav"
        target = target_dir / f"{take_id}{suffix}"
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(source, temporary)
        digest = content_sha256(temporary)
        _replace_with_retry(temporary, target)
        now = utc_now()
        with self.connection() as connection:
            line = connection.execute(
                "SELECT voice_profile_id FROM script_lines WHERE id=?", (line_id,)
            ).fetchone()
            if line is None:
                target.unlink(missing_ok=True)
                raise WorkerProtocolError(f"台词不存在：{line_id}")
            connection.execute(
                """
                INSERT INTO takes(
                    id,line_id,parent_take_id,voice_profile_id,rel_path,sha256,seed,preset,
                    model_revision,quality_json,status,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    take_id,
                    line_id,
                    parent_take_id,
                    line["voice_profile_id"],
                    target.relative_to(self.root).as_posix(),
                    digest,
                    seed,
                    preset,
                    model_revision,
                    _json(quality or {}),
                    "candidate",
                    now,
                ),
            )
            if adopt:
                self._adopt_take_in_connection(connection, take_id)
            else:
                connection.execute(
                    "UPDATE script_lines SET status='review',dirty=1,updated_at=? WHERE id=?",
                    (now, line_id),
                )
            row = connection.execute("SELECT * FROM takes WHERE id=?", (take_id,)).fetchone()
        self._touch("take", take_id, "created", {"line_id": line_id, "adopted": adopt})
        return _take_dict(row)

    def adopt_take(
        self,
        take_id: str,
        *,
        timeline_clip: dict[str, Any] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        with self.connection() as connection:
            updated = self._adopt_take_in_connection(
                connection, take_id, timeline_clip=timeline_clip, force=force
            )
        self._touch("take", take_id, "adopted", {"line_id": updated["line_id"]})
        return _take_dict(updated)

    def _adopt_take_in_connection(
        self,
        connection: sqlite3.Connection,
        take_id: str,
        *,
        timeline_clip: dict[str, Any] | None = None,
        force: bool = False,
    ) -> sqlite3.Row:
        now = utc_now()
        row = connection.execute(
            """
            SELECT t.*,l.speaker,l.target_start,l.order_index,l.archived
            FROM takes AS t JOIN script_lines AS l ON l.id=t.line_id
            WHERE t.id=?
            """,
            (take_id,),
        ).fetchone()
        if row is None:
            raise WorkerProtocolError(f"take 不存在：{take_id}")
        if bool(row["archived"]):
            raise WorkerProtocolError("归档台词的 take 不能直接采用，请先恢复台词")
        line_id = str(row["line_id"])
        current = connection.execute(
            "SELECT t.id,t.locked FROM script_lines AS l LEFT JOIN takes AS t "
            "ON t.id=l.current_take_id WHERE l.id=?",
            (line_id,),
        ).fetchone()
        if (
            current is not None
            and current["id"]
            and str(current["id"]) != take_id
            and bool(current["locked"])
            and not force
        ):
            raise WorkerProtocolError("当前采用 take 已锁定；请先解锁再切换版本")
        connection.execute(
            "UPDATE takes SET status='candidate',locked=0 WHERE line_id=? AND id<>?",
            (line_id, take_id),
        )
        connection.execute(
            "UPDATE takes SET status='adopted',reviewed_at=? WHERE id=?", (now, take_id)
        )
        connection.execute(
            "UPDATE script_lines SET current_take_id=?,status='completed',dirty=1,updated_at=? "
            "WHERE id=?",
            (take_id, now, line_id),
        )
        current_clip = connection.execute(
            "SELECT * FROM timeline_clips WHERE line_id=? ORDER BY id LIMIT 1", (line_id,)
        ).fetchone()
        quality = _read_json(row["quality_json"], {})
        signal = quality.get("signal") if isinstance(quality.get("signal"), dict) else {}
        duration = max(0.0, float(signal.get("duration_seconds") or 0.0))
        creating_clip = current_clip is None
        insertion_index = (
            _prepare_timeline_insertion(
                connection,
                line_id=line_id,
                line_order=int(row["order_index"] or 0),
            )
            if creating_clip
            else None
        )
        if timeline_clip is not None:
            clip = dict(timeline_clip)
            clip["line_id"] = line_id
            clip["take_id"] = take_id
        elif current_clip is not None:
            clip = dict(current_clip)
            clip["take_id"] = take_id
            if duration > 0:
                clip["duration"] = duration
        else:
            clip = {
                "id": f"line-{line_id}",
                "line_id": line_id,
                "take_id": take_id,
                "track": str(row["speaker"] or "dialogue"),
                "sequence_index": insertion_index,
                "position": float(row["target_start"] or 0.0),
                "duration": duration,
            }
        if insertion_index is not None:
            # Insert a newly adopted middle line beside its nearest script-order
            # neighbour without disturbing custom ordering already made by the user.
            clip["sequence_index"] = insertion_index
        elif "sequence_index" not in clip:
            clip["sequence_index"] = int(row["order_index"] or 0)
        fields = _timeline_clip_fields(clip)
        connection.execute(
            """
            INSERT INTO timeline_clips(
                id,line_id,take_id,track,sequence_index,position,duration,in_offset,out_offset,
                gain_db,fade_in,fade_out,muted,version,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                line_id=excluded.line_id,take_id=excluded.take_id,track=excluded.track,
                sequence_index=excluded.sequence_index,position=excluded.position,duration=excluded.duration,
                in_offset=excluded.in_offset,out_offset=excluded.out_offset,
                gain_db=excluded.gain_db,fade_in=excluded.fade_in,
                fade_out=excluded.fade_out,muted=excluded.muted,
                version=timeline_clips.version+1,updated_at=excluded.updated_at
            """,
            fields,
        )
        _normalize_timeline_sequence(connection)
        updated = connection.execute("SELECT * FROM takes WHERE id=?", (take_id,)).fetchone()
        if updated is None:
            raise WorkerProtocolError(f"take 采用失败：{take_id}")
        return updated

    def review_take(
        self,
        take_id: str,
        *,
        action: str = "note",
        note: str | None = None,
        locked: bool | None = None,
    ) -> dict[str, Any]:
        """Persist a take review decision without deleting any audio file."""
        decision = str(action or "note").strip().lower()
        if decision not in {"note", "reject", "restore", "lock", "unlock"}:
            raise WorkerProtocolError(f"未知 take 审核动作：{decision}")
        now = utc_now()
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM takes WHERE id=?", (take_id,)).fetchone()
            if row is None:
                raise WorkerProtocolError(f"take 不存在：{take_id}")
            status = str(row["status"])
            next_status = status
            next_locked = bool(row["locked"]) if locked is None else bool(locked)
            if decision == "reject":
                if status == "adopted":
                    raise WorkerProtocolError("当前采用 take 不能直接拒绝，请先采用其他版本")
                next_status = "rejected"
                next_locked = False
            elif decision == "restore":
                if status == "adopted":
                    raise WorkerProtocolError("当前 take 已经是采用版本")
                next_status = "candidate"
            elif decision == "lock":
                if status != "adopted":
                    raise WorkerProtocolError("只有当前采用 take 可以锁定")
                next_locked = True
            elif decision == "unlock":
                next_locked = False
            next_note = str(row["review_note"] or "") if note is None else str(note)[:2_000]
            connection.execute(
                "UPDATE takes SET status=?,locked=?,review_note=?,reviewed_at=? WHERE id=?",
                (next_status, 1 if next_locked else 0, next_note, now, take_id),
            )
            updated = connection.execute("SELECT * FROM takes WHERE id=?", (take_id,)).fetchone()
        self._touch(
            "take",
            take_id,
            f"review_{decision}",
            {"status": next_status, "locked": next_locked, "note": next_note},
        )
        return _take_dict(updated)

    def list_takes(self, line_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM takes WHERE line_id=? ORDER BY created_at DESC", (line_id,)
            ).fetchall()
        return [_take_dict(row) for row in rows]

    def prepare_take_comparison(
        self,
        take_a_id: str,
        take_b_id: str,
        *,
        target_lufs: float = -20.0,
        sync_onset: bool = True,
        match_loudness: bool = True,
    ) -> dict[str, Any]:
        """Build source-preserving A/B previews for two takes of the same line."""
        first_id = str(take_a_id or "").strip()
        second_id = str(take_b_id or "").strip()
        if not first_id or not second_id:
            raise WorkerProtocolError("A/B 对比需要两个 take")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM takes WHERE id IN (?,?)",
                (first_id, second_id),
            ).fetchall()
        by_id = {str(row["id"]): row for row in rows}
        if first_id not in by_id or second_id not in by_id:
            raise WorkerProtocolError("A/B 对比 take 不存在")
        if str(by_id[first_id]["line_id"]) != str(by_id[second_id]["line_id"]):
            raise WorkerProtocolError("A/B 对比必须来自同一条台词")
        source_a = safe_project_path(self.root, str(by_id[first_id]["rel_path"]))
        source_b = safe_project_path(self.root, str(by_id[second_id]["rel_path"]))
        result = prepare_synchronized_ab(
            source_a,
            source_b,
            safe_project_path(self.root, "previews"),
            target_lufs=float(target_lufs),
            sync_onset=bool(sync_onset),
            match_loudness=bool(match_loudness),
        )
        result.update(
            {
                "line_id": str(by_id[first_id]["line_id"]),
                "take_a_id": first_id,
                "take_b_id": second_id,
                "take_a": _take_dict(by_id[first_id]),
                "take_b": _take_dict(by_id[second_id]),
            }
        )
        self._touch(
            "take_comparison",
            result["session_id"],
            "prepared",
            {
                "line_id": result["line_id"],
                "take_a_id": first_id,
                "take_b_id": second_id,
                "target_lufs": float(target_lufs),
            },
        )
        return result

    def list_project_artifacts(self) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        with self.connection() as connection:
            takes = connection.execute(
                """
                SELECT t.*,l.speaker,l.text,l.order_index
                FROM takes AS t JOIN script_lines AS l ON l.id=t.line_id
                WHERE l.archived=0
                ORDER BY t.created_at DESC
                """
            ).fetchall()
            renders = connection.execute(
                "SELECT * FROM renders ORDER BY created_at DESC"
            ).fetchall()
        for row in takes:
            value = dict(row)
            relative = value["rel_path"]
            audio = safe_project_path(self.root, relative)
            values.append(
                {
                    "kind": "take",
                    "id": value["id"],
                    "line_id": value["line_id"],
                    "speaker": value["speaker"],
                    "text": value["text"],
                    "order_index": value["order_index"],
                    "status": value["status"],
                    "parent_take_id": value["parent_take_id"],
                    "voice_profile_id": value.get("voice_profile_id"),
                    "locked": bool(value.get("locked", 0)),
                    "review_note": str(value.get("review_note") or ""),
                    "reviewed_at": value.get("reviewed_at"),
                    "seed": value["seed"],
                    "preset": value["preset"],
                    "quality": _read_json(value.get("quality_json"), {}),
                    "path": str(audio),
                    "sidecar_path": str(audio.with_suffix(audio.suffix + ".json")),
                    "sha256": value["sha256"],
                    "created_at": value["created_at"],
                    "exists": audio.is_file(),
                }
            )
        for row in renders:
            value = dict(row)
            audio = safe_project_path(self.root, value["rel_path"])
            manifest = (
                safe_project_path(self.root, value["manifest_rel_path"])
                if value["manifest_rel_path"]
                else None
            )
            values.append(
                {
                    "kind": "render",
                    "id": value["id"],
                    "status": "completed",
                    "delivery_mode": value.get("delivery_mode", "draft"),
                    "path": str(audio),
                    "sidecar_path": str(manifest) if manifest else None,
                    "sha256": value["sha256"],
                    "created_at": value["created_at"],
                    "exists": audio.is_file(),
                }
            )
        return sorted(values, key=lambda value: value["created_at"], reverse=True)

    def replace_transcript_segments(
        self, asset_id: str, segments: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.get_asset(asset_id)
        normalized: list[dict[str, Any]] = []
        previous_end = 0.0
        for index, raw in enumerate(segments):
            if not isinstance(raw, dict):
                raise WorkerProtocolError(f"第 {index + 1} 个校对分段不是对象")
            start = float(raw.get("start_seconds", raw.get("start", 0.0)))
            end = float(raw.get("end_seconds", raw.get("end", 0.0)))
            text = str(raw.get("text") or "").strip()
            if start < 0 or end <= start:
                raise WorkerProtocolError(f"第 {index + 1} 个校对分段时间范围无效")
            if not text:
                raise WorkerProtocolError(f"第 {index + 1} 个校对分段文本为空")
            normalized.append(
                {
                    "id": str(raw.get("id") or uuid.uuid4()),
                    "order_index": int(raw.get("index", raw.get("order_index", index))),
                    "start_seconds": start,
                    "end_seconds": end,
                    "text": text,
                    "original_text": str(raw.get("original_text") or text),
                    "status": str(raw.get("status") or "review"),
                    "speaker": str(raw.get("speaker") or "")[:120],
                    "overlap_warning": start < previous_end,
                }
            )
            previous_end = max(previous_end, end)
        now = utc_now()
        with self.connection() as connection:
            connection.execute("DELETE FROM transcript_segments WHERE asset_id=?", (asset_id,))
            for item in normalized:
                connection.execute(
                    """
                    INSERT INTO transcript_segments(
                        id,asset_id,order_index,start_seconds,end_seconds,text,original_text,
                        status,speaker,overlap_warning,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        item["id"], asset_id, item["order_index"], item["start_seconds"],
                        item["end_seconds"], item["text"], item["original_text"],
                        item["status"], item["speaker"], 1 if item["overlap_warning"] else 0,
                        now, now,
                    ),
                )
        self._touch("transcript", asset_id, "replaced", {"segment_count": len(normalized)})
        return self.list_transcript_segments(asset_id)

    def list_transcript_segments(self, asset_id: str | None = None) -> list[dict[str, Any]]:
        with self.connection() as connection:
            if asset_id:
                rows = connection.execute(
                    "SELECT * FROM transcript_segments WHERE asset_id=? ORDER BY order_index,id",
                    (asset_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM transcript_segments ORDER BY asset_id,order_index,id"
                ).fetchall()
        return [_transcript_dict(row) for row in rows]

    def patch_transcript_segment(self, segment_id: str, values: dict[str, Any]) -> dict[str, Any]:
        allowed = {"start_seconds", "end_seconds", "text", "status", "speaker"}
        updates = {key: values[key] for key in allowed if key in values}
        if not updates:
            raise WorkerProtocolError("没有可更新的校对字段")
        if "text" in updates and not str(updates["text"] or "").strip():
            raise WorkerProtocolError("校对文本不能为空")
        if "status" in updates and updates["status"] not in {"review", "confirmed"}:
            raise WorkerProtocolError("校对状态必须是 review/confirmed")
        assignments = ",".join(f"{key}=?" for key in updates)
        with self.connection() as connection:
            existing = connection.execute(
                "SELECT * FROM transcript_segments WHERE id=?", (segment_id,)
            ).fetchone()
            if existing is None:
                raise WorkerProtocolError(f"校对分段不存在：{segment_id}")
            start = float(updates.get("start_seconds", existing["start_seconds"]))
            end = float(updates.get("end_seconds", existing["end_seconds"]))
            if start < 0 or end <= start:
                raise WorkerProtocolError("校对分段时间范围无效")
            params = list(updates.values()) + [utc_now(), segment_id]
            connection.execute(
                f"UPDATE transcript_segments SET {assignments},updated_at=? WHERE id=?", params
            )
            row = connection.execute(
                "SELECT * FROM transcript_segments WHERE id=?", (segment_id,)
            ).fetchone()
        self._touch("transcript_segment", segment_id, "updated", {"fields": sorted(updates)})
        return _transcript_dict(row)

    def replace_transcript_text(
        self, asset_id: str, search: str, replacement: str, *, case_sensitive: bool = True
    ) -> int:
        needle = str(search)
        if not needle:
            raise WorkerProtocolError("搜索文本不能为空")
        changed = 0
        for row in self.list_transcript_segments(asset_id):
            text = row["text"]
            if case_sensitive:
                updated = text.replace(needle, str(replacement))
            else:
                updated = re.sub(
                    re.escape(needle), lambda _match: str(replacement), text, flags=re.IGNORECASE
                )
            if updated != text:
                self.patch_transcript_segment(row["id"], {"text": updated, "status": "review"})
                changed += 1
        return changed

    def upsert_timeline_clip(self, value: dict[str, Any]) -> dict[str, Any]:
        return self.upsert_timeline_clips([value])[0]

    def upsert_timeline_clips(
        self, values: Iterable[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Atomically insert/update a bounded group of non-destructive timeline edits."""
        items = [dict(value) for value in values]
        if not items:
            raise WorkerProtocolError("没有可保存的时间线片段")
        if len(items) > 500:
            raise WorkerProtocolError("单次最多保存 500 个时间线片段")
        result_ids: list[str] = []
        with self.connection() as connection:
            for item in items:
                line_id = str(item.get("line_id") or "").strip()
                take_id = str(item.get("take_id") or "").strip()
                if not line_id or not take_id:
                    raise WorkerProtocolError("时间轴片段缺少 line_id/take_id")
                line = connection.execute(
                    "SELECT current_take_id,archived,order_index FROM script_lines WHERE id=?",
                    (line_id,),
                ).fetchone()
                if line is None or bool(line["archived"]):
                    raise WorkerProtocolError(f"时间轴台词不存在或已归档：{line_id}")
                if str(line["current_take_id"] or "") != take_id:
                    raise WorkerProtocolError(
                        "时间轴只能使用该句已采用的 take；请先通过 takes/adopt 原子切换版本"
                    )
                if "sequence_index" not in item:
                    existing = None
                    if item.get("id"):
                        existing = connection.execute(
                            "SELECT sequence_index FROM timeline_clips WHERE id=?",
                            (str(item["id"]),),
                        ).fetchone()
                    item["sequence_index"] = (
                        int(existing["sequence_index"])
                        if existing is not None
                        else int(line["order_index"] or 0)
                    )
                fields = _timeline_clip_fields(item)
                connection.execute(
                    """
                    INSERT INTO timeline_clips(
                        id,line_id,take_id,track,sequence_index,position,duration,in_offset,out_offset,
                        gain_db,fade_in,fade_out,muted,version,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET
                        line_id=excluded.line_id,take_id=excluded.take_id,track=excluded.track,
                        sequence_index=excluded.sequence_index,position=excluded.position,duration=excluded.duration,
                        in_offset=excluded.in_offset,out_offset=excluded.out_offset,
                        gain_db=excluded.gain_db,fade_in=excluded.fade_in,
                        fade_out=excluded.fade_out,muted=excluded.muted,
                        version=timeline_clips.version+1,updated_at=excluded.updated_at
                    """,
                    fields,
                )
                result_ids.append(str(fields[0]))
            _normalize_timeline_sequence(connection)
            results = []
            for clip_id in result_ids:
                row = connection.execute(
                    "SELECT * FROM timeline_clips WHERE id=?", (clip_id,)
                ).fetchone()
                if row is None:
                    raise WorkerProtocolError(f"时间轴片段保存失败：{clip_id}")
                results.append(dict(row))
        self._touch(
            "timeline_clip",
            None if len(results) > 1 else str(results[0]["id"]),
            "batch_upserted" if len(results) > 1 else "upserted",
            {"count": len(results), "ids": [str(value["id"]) for value in results]},
        )
        return results

    def delete_timeline_clips(self, clip_ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = list(dict.fromkeys(str(value).strip() for value in clip_ids if str(value).strip()))
        if not ids:
            raise WorkerProtocolError("没有选择要删除的时间线片段")
        if len(ids) > 500:
            raise WorkerProtocolError("单次最多删除 500 个时间线片段")
        placeholders = ",".join("?" for _ in ids)
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM timeline_clips WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            found = {str(row["id"]) for row in rows}
            missing = [clip_id for clip_id in ids if clip_id not in found]
            if missing:
                raise WorkerProtocolError(f"时间轴片段不存在：{', '.join(missing[:5])}")
            connection.execute(
                f"DELETE FROM timeline_clips WHERE id IN ({placeholders})",
                ids,
            )
            _normalize_timeline_sequence(connection)
        deleted = [dict(row) for row in rows]
        self._touch(
            "timeline_clip",
            None if len(deleted) > 1 else str(deleted[0]["id"]),
            "batch_deleted" if len(deleted) > 1 else "deleted",
            {"count": len(deleted), "ids": ids},
        )
        return deleted

    def list_timeline_clips(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT c.* FROM timeline_clips AS c
                JOIN script_lines AS l ON l.id=c.line_id
                WHERE l.archived=0
                ORDER BY c.sequence_index,l.order_index,c.id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def upsert_marker(self, value: dict[str, Any]) -> dict[str, Any]:
        marker_id = str(value.get("id") or uuid.uuid4())
        position = max(0.0, float(value.get("position") or 0.0))
        kind = str(value.get("kind") or "note").strip()[:40] or "note"
        label = str(value.get("label") or "").strip()[:500]
        metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO markers(id,position,kind,label,metadata_json) VALUES(?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    position=excluded.position,kind=excluded.kind,
                    label=excluded.label,metadata_json=excluded.metadata_json
                """,
                (marker_id, position, kind, label, _json(metadata)),
            )
            row = connection.execute("SELECT * FROM markers WHERE id=?", (marker_id,)).fetchone()
        self._touch("marker", marker_id, "upserted", {"position": position, "kind": kind})
        result = dict(row)
        result["metadata"] = _read_json(result.pop("metadata_json"), {})
        return result

    def list_markers(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM markers ORDER BY position,kind,id"
            ).fetchall()
        values: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            value["metadata"] = _read_json(value.pop("metadata_json"), {})
            values.append(value)
        return values

    def delete_marker(self, marker_id: str) -> bool:
        with self.connection() as connection:
            cursor = connection.execute("DELETE FROM markers WHERE id=?", (str(marker_id),))
            deleted = cursor.rowcount > 0
        if deleted:
            self._touch("marker", str(marker_id), "deleted", {})
        return deleted

    def import_locator_markers(
        self,
        structured: Any,
        *,
        replace_source: bool = False,
        source_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        candidates = _locator_marker_candidates(structured)
        if replace_source:
            with self.connection() as connection:
                rows = connection.execute("SELECT id,metadata_json FROM markers").fetchall()
                locator_ids = [
                    row["id"]
                    for row in rows
                    if _read_json(row["metadata_json"], {}).get("source") == "long_locator"
                ]
                connection.executemany(
                    "DELETE FROM markers WHERE id=?", ((marker_id,) for marker_id in locator_ids)
                )
        imported = []
        for index, candidate in enumerate(candidates):
            position = _timestamp_seconds(
                candidate.get("start_time")
                or candidate.get("start")
                or candidate.get("start_seconds")
                or candidate.get("time")
                or candidate.get("timestamp")
            )
            if position is None:
                continue
            label = next(
                (
                    str(candidate.get(key)).strip()
                    for key in ("topic", "title", "label", "match", "transcript_or_event", "summary")
                    if candidate.get(key) not in (None, "")
                ),
                f"定位标记 {index + 1}",
            )
            metadata = {
                "source": "long_locator",
                "source_item": candidate,
                **(source_metadata or {}),
            }
            imported.append(
                self.upsert_marker(
                    {
                        "position": position,
                        "kind": "locator",
                        "label": label,
                        "metadata": metadata,
                    }
                )
            )
        return imported

    def timeline_render_inputs(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT c.*,t.rel_path AS take_rel_path
                FROM timeline_clips AS c
                JOIN takes AS t ON t.id=c.take_id
                JOIN script_lines AS l ON l.id=c.line_id
                WHERE l.archived=0
                ORDER BY c.sequence_index,l.order_index,c.id
                """
            ).fetchall()
        values: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            value["path"] = str(safe_project_path(self.root, value.pop("take_rel_path")))
            values.append(value)
        return values

    def delivery_preflight(self) -> dict[str, Any]:
        """Verify that every active script line resolves to one adopted, renderable take."""
        with self.connection() as connection:
            lines = connection.execute(
                "SELECT * FROM script_lines WHERE archived=0 ORDER BY order_index,id"
            ).fetchall()
            takes = connection.execute("SELECT * FROM takes").fetchall()
            clips = connection.execute(
                "SELECT * FROM timeline_clips ORDER BY updated_at DESC,id"
            ).fetchall()
        takes_by_id = {str(row["id"]): row for row in takes}
        clips_by_line: dict[str, sqlite3.Row] = {}
        for row in clips:
            clips_by_line.setdefault(str(row["line_id"]), row)
        issues: list[dict[str, Any]] = []
        deliverable_lines: list[dict[str, Any]] = []
        if not lines:
            issues.append(
                {
                    "severity": "error",
                    "code": "empty_project",
                    "line_id": None,
                    "order_index": None,
                    "message": "项目中没有可交付的台词",
                }
            )
        else:
            casting = self.casting_readiness({str(row["speaker"] or "旁白") for row in lines})
            first_line_by_speaker: dict[str, sqlite3.Row] = {}
            for row in lines:
                first_line_by_speaker.setdefault(str(row["speaker"] or "旁白"), row)
            for casting_issue in casting["issues"]:
                speaker = str(casting_issue["speaker"])
                line = first_line_by_speaker.get(speaker)
                issues.append(
                    {
                        "severity": "error",
                        "code": "casting_unconfirmed",
                        "line_id": None if line is None else str(line["id"]),
                        "order_index": None if line is None else int(line["order_index"]),
                        "message": f"角色 {speaker} 尚未完成代表句选角确认",
                    }
                )
        for row in lines:
            line_id = str(row["id"])
            order = int(row["order_index"]) + 1
            prefix = f"第 {order} 句"
            if not row["voice_profile_id"]:
                issues.append(
                    {
                        "severity": "error",
                        "code": "unmapped_voice",
                        "line_id": line_id,
                        "order_index": int(row["order_index"]),
                        "message": f"{prefix}尚未映射角色音色",
                    }
                )
            take_id = str(row["current_take_id"] or "")
            take = takes_by_id.get(take_id)
            if not take_id or take is None:
                issues.append(
                    {
                        "severity": "error",
                        "code": "missing_adopted_take",
                        "line_id": line_id,
                        "order_index": int(row["order_index"]),
                        "message": f"{prefix}没有已采用 take",
                    }
                )
                continue
            if str(take["status"]) != "adopted":
                issues.append(
                    {
                        "severity": "error",
                        "code": "take_status_mismatch",
                        "line_id": line_id,
                        "order_index": int(row["order_index"]),
                        "message": f"{prefix}当前 take 与 adopted 状态不一致",
                    }
                )
            audio = safe_project_path(self.root, take["rel_path"])
            if not audio.is_file() or content_sha256(audio) != str(take["sha256"]):
                issues.append(
                    {
                        "severity": "error",
                        "code": "missing_or_changed_take",
                        "line_id": line_id,
                        "order_index": int(row["order_index"]),
                        "message": f"{prefix}采用的音频缺失或哈希变化",
                    }
                )
            clip = clips_by_line.get(line_id)
            if clip is None:
                issues.append(
                    {
                        "severity": "error",
                        "code": "missing_timeline_clip",
                        "line_id": line_id,
                        "order_index": int(row["order_index"]),
                        "message": f"{prefix}没有时间线片段",
                    }
                )
            elif str(clip["take_id"]) != take_id:
                issues.append(
                    {
                        "severity": "error",
                        "code": "timeline_take_mismatch",
                        "line_id": line_id,
                        "order_index": int(row["order_index"]),
                        "message": f"{prefix}时间线使用的 take 与采用版不一致",
                    }
                )
            deliverable_lines.append(
                {
                    "line_id": line_id,
                    "order_index": int(row["order_index"]),
                    "speaker": row["speaker"],
                    "take_id": take_id,
                    "take_sha256": take["sha256"],
                    "timeline_clip_id": None if clip is None else clip["id"],
                    "dirty": bool(row["dirty"]),
                }
            )
        error_count = sum(issue["severity"] == "error" for issue in issues)
        return {
            "ok": bool(lines) and error_count == 0,
            "active_lines": len(lines),
            "ready_lines": len(deliverable_lines),
            "error_count": error_count,
            "issues": issues,
            "lines": deliverable_lines,
        }

    def record_render(
        self,
        output_path: str | Path,
        *,
        manifest_path: str | Path | None = None,
        quality_report: dict[str, Any] | None = None,
        delivery_mode: str = "draft",
        rendered_line_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        output = Path(output_path).expanduser().resolve()
        try:
            relative = output.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise WorkerProtocolError("项目渲染文件必须位于项目目录内") from exc
        if not output.is_file():
            raise WorkerProtocolError(f"渲染文件不存在：{output}")
        manifest_relative = None
        if manifest_path is not None:
            manifest = Path(manifest_path).expanduser().resolve()
            try:
                manifest_relative = manifest.relative_to(self.root).as_posix()
            except ValueError as exc:
                raise WorkerProtocolError("渲染 manifest 必须位于项目目录内") from exc
        render_id = str(uuid.uuid4())
        created = utc_now()
        digest = content_sha256(output)
        mode = str(delivery_mode or "draft").lower()
        if mode not in {"draft", "final"}:
            raise WorkerProtocolError("交付模式必须是 draft/final")
        line_ids = list(dict.fromkeys(str(value) for value in (rendered_line_ids or []) if str(value)))
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO renders(id,rel_path,sha256,manifest_rel_path,quality_report_json,"
                "delivery_mode,created_at) VALUES(?,?,?,?,?,?,?)",
                (
                    render_id,
                    relative,
                    digest,
                    manifest_relative,
                    _json(quality_report or {}),
                    mode,
                    created,
                ),
            )
            if mode == "final" and line_ids:
                placeholders = ",".join("?" for _ in line_ids)
                connection.execute(
                    f"UPDATE script_lines SET dirty=0,updated_at=? "
                    f"WHERE archived=0 AND id IN ({placeholders})",
                    [created, *line_ids],
                )
            row = connection.execute("SELECT * FROM renders WHERE id=?", (render_id,)).fetchone()
        self._touch("render", render_id, "created", {"sha256": digest, "delivery_mode": mode})
        result = dict(row)
        result["quality_report"] = _read_json(result.pop("quality_report_json"), {})
        return result

    def _migrate(self) -> bool:
        connection = sqlite3.connect(self.database_path)
        try:
            connection.row_factory = sqlite3.Row
            try:
                row = connection.execute("SELECT schema_version FROM project LIMIT 1").fetchone()
            except sqlite3.DatabaseError as exc:
                raise WorkerProtocolError("项目数据库缺少 project 表") from exc
            if row is None:
                raise WorkerProtocolError("项目数据库缺少版本信息")
            version = int(row["schema_version"])
            if version > PROJECT_SCHEMA_VERSION:
                raise WorkerProtocolError(
                    f"项目格式 v{version} 高于当前支持的 v{PROJECT_SCHEMA_VERSION}"
                )
            migrated = version < PROJECT_SCHEMA_VERSION
            if migrated:
                backup = self.database_path.with_suffix(
                    f".v{version}.{datetime.now().strftime('%Y%m%d%H%M%S')}.bak"
                )
                backup_connection = sqlite3.connect(backup)
                try:
                    connection.backup(backup_connection)
                    backup_connection.commit()
                finally:
                    backup_connection.close()
            _initialize_schema(connection)
            if migrated:
                connection.execute(
                    "UPDATE project SET schema_version=?,updated_at=?",
                    (PROJECT_SCHEMA_VERSION, utc_now()),
                )
            connection.commit()
        finally:
            connection.close()
        return migrated

    def _touch(
        self, entity_type: str, entity_id: str | None, action: str, payload: dict[str, Any]
    ) -> None:
        now = utc_now()
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO events(id,entity_type,entity_id,action,payload_json,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (str(uuid.uuid4()), entity_type, entity_id, action, _json(payload), now),
            )
            connection.execute("UPDATE project SET updated_at=?", (now,))
        self._write_manifest()

    def _write_manifest(self) -> None:
        payload = self.snapshot()
        temporary = self.manifest_path.with_suffix(
            self.manifest_path.suffix + f".{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _replace_with_retry(temporary, self.manifest_path)
        finally:
            temporary.unlink(missing_ok=True)


def _locator_marker_candidates(structured: Any) -> list[dict[str, Any]]:
    if isinstance(structured, list):
        return [value for value in structured if isinstance(value, dict)]
    if not isinstance(structured, dict):
        return []
    timestamp_keys = {"start_time", "start", "start_seconds", "time", "timestamp"}
    if timestamp_keys.intersection(structured):
        return [structured]
    for key in ("timeline", "items", "segments", "matches", "chapters", "events", "results"):
        value = structured.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _timestamp_seconds(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    try:
        if ":" not in text:
            return max(0.0, float(text))
        parts = [float(part) for part in text.split(":")]
        if len(parts) == 2:
            minutes, seconds = parts
            return max(0.0, minutes * 60 + seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return max(0.0, hours * 3600 + minutes * 60 + seconds)
    except ValueError:
        return None
    return None


def _initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS project(
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS assets(
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            rel_path TEXT NOT NULL,
            external INTEGER NOT NULL DEFAULT 0,
            sha256 TEXT NOT NULL,
            source_name TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            UNIQUE(kind,sha256)
        );
        CREATE TABLE IF NOT EXISTS voice_profiles(
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            reference_asset_id TEXT,
            transcript TEXT NOT NULL DEFAULT '',
            language TEXT NOT NULL DEFAULT 'zh',
            tags_json TEXT NOT NULL DEFAULT '[]',
            authorization_notes TEXT NOT NULL DEFAULT '',
            quality_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(reference_asset_id) REFERENCES assets(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS script_lines(
            id TEXT PRIMARY KEY,
            order_index INTEGER NOT NULL,
            scene TEXT NOT NULL DEFAULT '',
            speaker TEXT NOT NULL DEFAULT '',
            voice_profile_id TEXT,
            text TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT 'zh',
            instruction TEXT NOT NULL DEFAULT '',
            director_notes TEXT NOT NULL DEFAULT '',
            seed INTEGER,
            preset TEXT NOT NULL DEFAULT 'balanced',
            target_start REAL,
            target_end REAL,
            status TEXT NOT NULL DEFAULT 'draft',
            current_take_id TEXT,
            dirty INTEGER NOT NULL DEFAULT 1,
            archived INTEGER NOT NULL DEFAULT 0,
            source_key TEXT NOT NULL DEFAULT '',
            revision_id TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(voice_profile_id) REFERENCES voice_profiles(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS script_revisions(
            id TEXT PRIMARY KEY,
            source_format TEXT NOT NULL DEFAULT '',
            source_name TEXT NOT NULL DEFAULT '',
            mode TEXT NOT NULL DEFAULT 'merge',
            diff_json TEXT NOT NULL DEFAULT '{}',
            before_json TEXT NOT NULL DEFAULT '[]',
            after_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS takes(
            id TEXT PRIMARY KEY,
            line_id TEXT NOT NULL,
            parent_take_id TEXT,
            voice_profile_id TEXT,
            rel_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            seed INTEGER,
            preset TEXT NOT NULL DEFAULT 'balanced',
            model_revision TEXT,
            quality_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'candidate',
            locked INTEGER NOT NULL DEFAULT 0,
            review_note TEXT NOT NULL DEFAULT '',
            reviewed_at TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(line_id) REFERENCES script_lines(id) ON DELETE CASCADE,
            FOREIGN KEY(parent_take_id) REFERENCES takes(id) ON DELETE SET NULL,
            FOREIGN KEY(voice_profile_id) REFERENCES voice_profiles(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS casting_roles(
            speaker TEXT PRIMARY KEY,
            voice_profile_id TEXT,
            voice_signature TEXT NOT NULL DEFAULT '',
            representative_line_id TEXT,
            representative_take_id TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            notes TEXT NOT NULL DEFAULT '',
            confirmed_at TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(voice_profile_id) REFERENCES voice_profiles(id) ON DELETE SET NULL,
            FOREIGN KEY(representative_line_id) REFERENCES script_lines(id) ON DELETE SET NULL,
            FOREIGN KEY(representative_take_id) REFERENCES takes(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS pronunciation_entries(
            id TEXT PRIMARY KEY,
            display_text TEXT NOT NULL,
            spoken_text TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT 'all',
            case_sensitive INTEGER NOT NULL DEFAULT 0,
            notes TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(display_text,language)
        );
        CREATE TABLE IF NOT EXISTS production_clips(
            id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'music',
            position REAL NOT NULL DEFAULT 0,
            duration REAL NOT NULL DEFAULT 0,
            gain_db REAL NOT NULL DEFAULT 0,
            fade_in REAL NOT NULL DEFAULT 0,
            fade_out REAL NOT NULL DEFAULT 0,
            ducking_db REAL NOT NULL DEFAULT 0,
            loop INTEGER NOT NULL DEFAULT 0,
            fill_gaps INTEGER NOT NULL DEFAULT 0,
            muted INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY,
            line_id TEXT,
            take_id TEXT,
            stage TEXT NOT NULL DEFAULT 'generate',
            status TEXT NOT NULL DEFAULT 'queued',
            payload_json TEXT NOT NULL DEFAULT '{}',
            checkpoint_rel_path TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            priority INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(line_id) REFERENCES script_lines(id) ON DELETE CASCADE,
            FOREIGN KEY(take_id) REFERENCES takes(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS timeline_clips(
            id TEXT PRIMARY KEY,
            line_id TEXT NOT NULL,
            take_id TEXT NOT NULL,
            track TEXT NOT NULL DEFAULT 'dialogue',
            sequence_index INTEGER NOT NULL DEFAULT 0,
            position REAL NOT NULL DEFAULT 0,
            duration REAL NOT NULL DEFAULT 0,
            in_offset REAL NOT NULL DEFAULT 0,
            out_offset REAL NOT NULL DEFAULT 0,
            gain_db REAL NOT NULL DEFAULT 0,
            fade_in REAL NOT NULL DEFAULT 0,
            fade_out REAL NOT NULL DEFAULT 0,
            muted INTEGER NOT NULL DEFAULT 0,
            version INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(line_id) REFERENCES script_lines(id) ON DELETE CASCADE,
            FOREIGN KEY(take_id) REFERENCES takes(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS markers(
            id TEXT PRIMARY KEY,
            position REAL NOT NULL,
            kind TEXT NOT NULL DEFAULT 'note',
            label TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS transcript_segments(
            id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL,
            order_index INTEGER NOT NULL,
            start_seconds REAL NOT NULL,
            end_seconds REAL NOT NULL,
            text TEXT NOT NULL,
            original_text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'review',
            speaker TEXT NOT NULL DEFAULT '',
            overlap_warning INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(asset_id) REFERENCES assets(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS renders(
            id TEXT PRIMARY KEY,
            rel_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            manifest_rel_path TEXT,
            quality_report_json TEXT NOT NULL DEFAULT '{}',
            delivery_mode TEXT NOT NULL DEFAULT 'draft',
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events(
            id TEXT PRIMARY KEY,
            entity_type TEXT NOT NULL,
            entity_id TEXT,
            action TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_script_order ON script_lines(order_index);
        CREATE INDEX IF NOT EXISTS idx_take_line ON takes(line_id,created_at);
        CREATE INDEX IF NOT EXISTS idx_job_queue ON jobs(status,priority,created_at);
        CREATE INDEX IF NOT EXISTS idx_timeline_position ON timeline_clips(position,track);
        CREATE INDEX IF NOT EXISTS idx_transcript_asset ON transcript_segments(asset_id,order_index);
        """
    )
    _ensure_schema_columns(connection)
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_script_active ON script_lines(archived,order_index)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_script_source ON script_lines(source_key)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_timeline_sequence "
        "ON timeline_clips(sequence_index,position)"
    )


def _ensure_schema_columns(connection: sqlite3.Connection) -> None:
    """Add compatibility columns when opening an older project before bumping its schema."""
    table_columns = {
        table: {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        for table in ("script_lines", "renders", "takes", "production_clips", "timeline_clips")
    }
    for name, declaration in (
        ("archived", "INTEGER NOT NULL DEFAULT 0"),
        ("source_key", "TEXT NOT NULL DEFAULT ''"),
        ("revision_id", "TEXT"),
        ("director_notes", "TEXT NOT NULL DEFAULT ''"),
    ):
        if name not in table_columns["script_lines"]:
            connection.execute(f"ALTER TABLE script_lines ADD COLUMN {name} {declaration}")
    if "delivery_mode" not in table_columns["renders"]:
        connection.execute(
            "ALTER TABLE renders ADD COLUMN delivery_mode TEXT NOT NULL DEFAULT 'draft'"
        )
    for name, declaration in (
        ("voice_profile_id", "TEXT"),
        ("locked", "INTEGER NOT NULL DEFAULT 0"),
        ("review_note", "TEXT NOT NULL DEFAULT ''"),
        ("reviewed_at", "TEXT"),
    ):
        if name not in table_columns["takes"]:
            connection.execute(f"ALTER TABLE takes ADD COLUMN {name} {declaration}")
    if "fill_gaps" not in table_columns["production_clips"]:
        connection.execute(
            "ALTER TABLE production_clips ADD COLUMN fill_gaps INTEGER NOT NULL DEFAULT 0"
        )
    if "sequence_index" not in table_columns["timeline_clips"]:
        connection.execute(
            "ALTER TABLE timeline_clips ADD COLUMN sequence_index INTEGER NOT NULL DEFAULT 0"
        )
        connection.execute(
            "UPDATE timeline_clips SET sequence_index=COALESCE((SELECT order_index FROM script_lines "
            "WHERE script_lines.id=timeline_clips.line_id),0)"
        )
    _normalize_timeline_sequence(connection)
    connection.execute(
        "UPDATE takes SET voice_profile_id=(SELECT voice_profile_id FROM script_lines "
        "WHERE script_lines.id=takes.line_id) WHERE voice_profile_id IS NULL"
    )


def _ordered_timeline_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return editing order without allowing track time to act as a tie-breaker."""
    return connection.execute(
        """
        SELECT c.id,c.line_id,c.sequence_index,l.order_index
        FROM timeline_clips AS c
        JOIN script_lines AS l ON l.id=c.line_id
        WHERE l.archived=0
        ORDER BY c.sequence_index,l.order_index,c.id
        """
    ).fetchall()


def _normalize_timeline_sequence(connection: sqlite3.Connection) -> None:
    """Repair duplicate/gapped legacy sequence values into one dense stable order."""
    for index, row in enumerate(_ordered_timeline_rows(connection)):
        if int(row["sequence_index"] or 0) != index:
            connection.execute(
                "UPDATE timeline_clips SET sequence_index=? WHERE id=?",
                (index, str(row["id"])),
            )


def _prepare_timeline_insertion(
    connection: sqlite3.Connection,
    *,
    line_id: str,
    line_order: int,
) -> int:
    """Open one sequence slot for a newly adopted line while preserving custom order."""
    rows = [row for row in _ordered_timeline_rows(connection) if str(row["line_id"]) != line_id]
    prior_orders = [
        int(row["order_index"] or 0)
        for row in rows
        if int(row["order_index"] or 0) < line_order
    ]
    if prior_orders:
        nearest_prior = max(prior_orders)
        insertion_index = max(
            index
            for index, row in enumerate(rows)
            if int(row["order_index"] or 0) == nearest_prior
        ) + 1
    else:
        following_orders = [
            int(row["order_index"] or 0)
            for row in rows
            if int(row["order_index"] or 0) > line_order
        ]
        if following_orders:
            nearest_following = min(following_orders)
            insertion_index = min(
                index
                for index, row in enumerate(rows)
                if int(row["order_index"] or 0) == nearest_following
            )
        else:
            insertion_index = len(rows)
    for index, existing in enumerate(rows):
        shifted_index = index if index < insertion_index else index + 1
        if int(existing["sequence_index"] or 0) != shifted_index:
            connection.execute(
                "UPDATE timeline_clips SET sequence_index=? WHERE id=?",
                (shifted_index, str(existing["id"])),
            )
    return insertion_index


def _normalize_line(index: int, value: dict[str, Any]) -> dict[str, Any]:
    text = str(value.get("text") or "").strip()
    if not text:
        raise WorkerProtocolError(f"第 {index + 1} 条台词为空")
    start = value.get("target_start")
    end = value.get("target_end")
    start_value = None if start in (None, "") else float(start)
    end_value = None if end in (None, "") else float(end)
    if start_value is not None and start_value < 0:
        raise WorkerProtocolError(f"第 {index + 1} 条台词开始时间不能小于 0")
    if start_value is not None and end_value is not None and end_value <= start_value:
        raise WorkerProtocolError(f"第 {index + 1} 条台词结束时间必须晚于开始时间")
    return {
        "id": str(value.get("id") or uuid.uuid4()),
        "order_index": int(value.get("order_index", index)),
        "scene": str(value.get("scene") or "")[:200],
        "speaker": str(value.get("speaker") or "")[:120],
        "voice_profile_id": str(value.get("voice_profile_id") or "") or None,
        "text": text[:20_000],
        "language": str(value.get("language") or "zh")[:16],
        "instruction": str(value.get("instruction") or "")[:2_000],
        "director_notes": str(value.get("director_notes") or "")[:4_000],
        "seed": None if value.get("seed") in (None, "") else int(value["seed"]),
        "preset": str(value.get("preset") or "balanced")[:32],
        "target_start": start_value,
        "target_end": end_value,
        "status": "draft",
        "metadata": value.get("metadata") or {},
    }


def _asset_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["external"] = bool(result["external"])
    result["metadata"] = _read_json(result.pop("metadata_json"), {})
    return result


def _pronunciation_dict(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["case_sensitive"] = bool(result.get("case_sensitive", 0))
    return result


def _apply_pronunciation_entries(
    text: str, language: str, entries: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    source = str(text or "")
    normalized_language = str(language or "zh").lower()
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    for raw in entries:
        entry = dict(raw)
        entry_language = str(entry.get("language") or "all").lower()
        if entry_language not in {"all", normalized_language}:
            continue
        display = str(entry.get("display_text") or "")
        spoken = str(entry.get("spoken_text") or "")
        if not display or not spoken:
            continue
        flags = 0 if bool(entry.get("case_sensitive")) else re.IGNORECASE
        for match in re.finditer(re.escape(display), source, flags):
            candidates.append((match.start(), match.end(), entry))
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), str(item[2].get("id") or "")))
    selected: list[tuple[int, int, dict[str, Any]]] = []
    cursor = 0
    for start, end, entry in candidates:
        if start < cursor:
            continue
        selected.append((start, end, entry))
        cursor = end
    parts: list[str] = []
    matches: list[dict[str, Any]] = []
    cursor = 0
    for start, end, entry in selected:
        parts.append(source[cursor:start])
        spoken = str(entry["spoken_text"])
        parts.append(spoken)
        matches.append(
            {
                "entry_id": str(entry.get("id") or ""),
                "display_text": source[start:end],
                "spoken_text": spoken,
                "start": start,
                "end": end,
            }
        )
        cursor = end
    parts.append(source[cursor:])
    attention_pattern = re.compile(
        r"\b[A-Z]{2,}\b|\d+(?:[./:\-]\d+)+|\d+(?:\.\d+)?|[一二三四五六七八九十百千万亿〇零]{2,}"
    )
    attention_tokens = list(dict.fromkeys(match.group(0) for match in attention_pattern.finditer(source)))
    spoken_text = "".join(parts)
    return {
        "display_text": source,
        "spoken_text": spoken_text,
        "changed": spoken_text != source,
        "matches": matches,
        "attention_tokens": attention_tokens,
    }


def _timeline_clip_fields(value: dict[str, Any]) -> tuple[Any, ...]:
    clip_id = str(value.get("id") or uuid.uuid4())
    line_id = str(value.get("line_id") or "").strip()
    take_id = str(value.get("take_id") or "").strip()
    if not line_id or not take_id:
        raise WorkerProtocolError("时间轴片段缺少 line_id/take_id")
    duration = max(0.0, float(value.get("duration") or 0.0))
    in_offset = max(0.0, float(value.get("in_offset") or 0.0))
    out_offset = max(0.0, float(value.get("out_offset") or 0.0))
    if out_offset and out_offset <= in_offset:
        raise WorkerProtocolError("时间轴片段的裁剪终点必须大于起点")
    if duration and in_offset >= duration:
        raise WorkerProtocolError("时间轴片段的裁剪起点超出音频时长")
    if duration and out_offset > duration + 0.001:
        raise WorkerProtocolError("时间轴片段的裁剪终点超出音频时长")
    return (
        clip_id,
        line_id,
        take_id,
        str(value.get("track") or "dialogue")[:80],
        max(0, min(1_000_000, int(value.get("sequence_index") or 0))),
        max(0.0, float(value.get("position") or 0.0)),
        duration,
        in_offset,
        out_offset,
        float(value.get("gain_db") or 0.0),
        max(0.0, float(value.get("fade_in") or 0.0)),
        max(0.0, float(value.get("fade_out") or 0.0)),
        1 if value.get("muted") else 0,
        int(value.get("version") or 1),
        utc_now(),
    )


def _voice_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["tags"] = _read_json(result.pop("tags_json"), [])
    result["quality"] = _read_json(result.pop("quality_json"), {})
    return result


def _casting_voice_signature(row: sqlite3.Row | None) -> str:
    if row is None:
        return ""
    payload = "\n".join(
        (
            str(row["id"] or ""),
            str(row["reference_asset_id"] or ""),
            str(row["transcript"] or ""),
            str(row["language"] or ""),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _script_source_key(line: dict[str, Any]) -> str:
    metadata = line.get("metadata") if isinstance(line.get("metadata"), dict) else {}
    for key in ("source_index", "source_block", "source_row"):
        value = metadata.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    start = line.get("target_start")
    end = line.get("target_end")
    if start is not None and end is not None:
        return f"time:{float(start):.3f}:{float(end):.3f}"
    return ""


def _script_fingerprint(line: dict[str, Any]) -> str:
    text = re.sub(r"\s+", " ", str(line.get("text") or "").strip()).casefold()
    speaker = str(line.get("speaker") or "").strip().casefold()
    start = line.get("target_start")
    end = line.get("target_end")
    timing = "" if start is None or end is None else f"|{float(start):.3f}|{float(end):.3f}"
    return f"{speaker}|{text}{timing}"


def _match_script_lines(
    existing: list[dict[str, Any]], incoming: list[dict[str, Any]]
) -> dict[int, str]:
    """Match stable identities first, then exact content, then remaining line order."""
    matches: dict[int, str] = {}
    used: set[str] = set()
    existing_by_id = {str(line["id"]): line for line in existing}
    for index, line in enumerate(incoming):
        line_id = str(line.get("id") or "")
        if line_id in existing_by_id and line_id not in used:
            matches[index] = line_id
            used.add(line_id)

    def unique_map(values: list[dict[str, Any]], key_fn: Any) -> dict[str, str]:
        grouped: dict[str, list[str]] = {}
        for value in values:
            value_id = str(value["id"])
            if value_id in used:
                continue
            key = key_fn(value)
            if key:
                grouped.setdefault(key, []).append(value_id)
        return {key: ids[0] for key, ids in grouped.items() if len(ids) == 1}

    for key_fn in (_script_source_key, _script_fingerprint):
        lookup = unique_map(existing, key_fn)
        incoming_counts: dict[str, int] = {}
        for index, line in enumerate(incoming):
            if index in matches:
                continue
            key = key_fn(line)
            if key:
                incoming_counts[key] = incoming_counts.get(key, 0) + 1
        for index, line in enumerate(incoming):
            if index in matches:
                continue
            key = key_fn(line)
            matched_id = lookup.get(key) if incoming_counts.get(key) == 1 else None
            if matched_id and matched_id not in used:
                matches[index] = matched_id
                used.add(matched_id)

    existing_by_order = {
        int(line["order_index"]): str(line["id"])
        for line in existing
        if str(line["id"]) not in used
    }
    for index, line in enumerate(incoming):
        if index in matches:
            continue
        # A stable source key (for example an SRT block index) is an identity.
        # If that identity did not match, this is a new line rather than an
        # order-based replacement of a different archived line.
        if _script_source_key(line):
            continue
        matched_id = existing_by_order.get(int(line["order_index"]))
        if matched_id and matched_id not in used:
            matches[index] = matched_id
            used.add(matched_id)
    return matches


def _line_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["dirty"] = bool(result["dirty"])
    result["archived"] = bool(result.get("archived", 0))
    result["metadata"] = _read_json(result.pop("metadata_json"), {})
    return result


def _job_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["payload"] = _read_json(result.pop("payload_json"), {})
    return result


def _take_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["locked"] = bool(result.get("locked", 0))
    result["quality"] = _read_json(result.pop("quality_json"), {})
    return result


def _transcript_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["overlap_warning"] = bool(result["overlap_warning"])
    return result
