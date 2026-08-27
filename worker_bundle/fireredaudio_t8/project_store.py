from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import WorkerProtocolError


PROJECT_SCHEMA_VERSION = 2
PROJECT_SUFFIX = ".firered"
PROJECT_DIRECTORIES = (
    "assets",
    "voices",
    "scripts",
    "segments",
    "renders",
    "cache",
    "logs",
)
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
                    "takes",
                    "jobs",
                    "timeline_clips",
                    "markers",
                    "renders",
                    "transcript_segments",
                )
            }
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
                temporary.replace(target)
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
        profile_id = str(value.get("id") or uuid.uuid4())
        name = str(value.get("name") or "").strip()[:120]
        if not name:
            raise WorkerProtocolError("角色音色缺少 name")
        reference_asset_id = str(value.get("reference_asset_id") or "").strip() or None
        now = utc_now()
        with self.connection() as connection:
            existing = connection.execute(
                "SELECT created_at FROM voice_profiles WHERE id=?", (profile_id,)
            ).fetchone()
            created = existing["created_at"] if existing else now
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
        self._touch("voice_profile", profile_id, "upserted", {"name": name})
        return _voice_dict(row)

    def list_voice_profiles(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM voice_profiles ORDER BY name COLLATE NOCASE, updated_at DESC"
            ).fetchall()
        return [_voice_dict(row) for row in rows]

    def get_voice_profile(self, profile_id: str) -> dict[str, Any]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM voice_profiles WHERE id=?", (profile_id,)
            ).fetchone()
        if row is None:
            raise WorkerProtocolError(f"角色音色不存在：{profile_id}")
        return _voice_dict(row)

    def replace_script_lines(self, lines: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = [_normalize_line(index, value) for index, value in enumerate(lines)]
        now = utc_now()
        with self.connection() as connection:
            connection.execute("DELETE FROM jobs")
            connection.execute("DELETE FROM timeline_clips")
            connection.execute("DELETE FROM takes")
            connection.execute("DELETE FROM script_lines")
            for line in normalized:
                connection.execute(
                    """
                    INSERT INTO script_lines(
                        id,order_index,scene,speaker,voice_profile_id,text,language,instruction,
                        seed,preset,target_start,target_end,status,current_take_id,dirty,
                        metadata_json,created_at,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                        line["seed"],
                        line["preset"],
                        line["target_start"],
                        line["target_end"],
                        line["status"],
                        None,
                        1,
                        _json(line["metadata"]),
                        now,
                        now,
                    ),
                )
        self._touch("script", None, "replaced", {"line_count": len(normalized)})
        return self.list_script_lines()

    def list_script_lines(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM script_lines ORDER BY order_index,id"
            ).fetchall()
        return [_line_dict(row) for row in rows]

    def patch_script_line(self, line_id: str, values: dict[str, Any]) -> dict[str, Any]:
        allowed = {
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
            "status",
        }
        updates = {key: values[key] for key in allowed if key in values}
        if not updates:
            raise WorkerProtocolError("没有可更新的台词字段")
        if "text" in updates and not str(updates["text"] or "").strip():
            raise WorkerProtocolError("台词文本不能为空")
        generation_fields = {
            "voice_profile_id",
            "text",
            "language",
            "instruction",
            "seed",
            "preset",
        }
        if generation_fields.intersection(updates) and "status" not in updates:
            updates["status"] = "draft"
        assignments = ",".join(f"{key}=?" for key in updates)
        params = list(updates.values()) + [1, utc_now(), line_id]
        with self.connection() as connection:
            cursor = connection.execute(
                f"UPDATE script_lines SET {assignments},dirty=?,updated_at=? WHERE id=?",
                params,
            )
            if cursor.rowcount != 1:
                raise WorkerProtocolError(f"台词不存在：{line_id}")
            row = connection.execute(
                "SELECT * FROM script_lines WHERE id=?", (line_id,)
            ).fetchone()
        self._touch("script_line", line_id, "updated", {"fields": sorted(updates)})
        return _line_dict(row)

    def enqueue_lines(
        self, line_ids: Iterable[str] | None = None, *, force: bool = False
    ) -> list[dict[str, Any]]:
        requested = [str(value) for value in (line_ids or []) if str(value)]
        with self.connection() as connection:
            if requested:
                placeholders = ",".join("?" for _ in requested)
                rows = connection.execute(
                    f"SELECT * FROM script_lines WHERE id IN ({placeholders}) ORDER BY order_index",
                    requested,
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM script_lines ORDER BY order_index"
                ).fetchall()
            now = utc_now()
            jobs: list[dict[str, Any]] = []
            for priority, line in enumerate(rows):
                if not force and line["status"] == "completed" and line["current_take_id"]:
                    continue
                job_id = str(uuid.uuid4())
                payload = {
                    "line_id": line["id"],
                    "order_index": line["order_index"],
                    "text": line["text"],
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
        temporary.replace(path)
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
                connection.execute(
                    "UPDATE script_lines SET status=?,updated_at=? WHERE id=?",
                    (row["status"], utc_now(), row["line_id"]),
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
        temporary.replace(target)
        now = utc_now()
        with self.connection() as connection:
            if connection.execute(
                "SELECT 1 FROM script_lines WHERE id=?", (line_id,)
            ).fetchone() is None:
                target.unlink(missing_ok=True)
                raise WorkerProtocolError(f"台词不存在：{line_id}")
            connection.execute(
                """
                INSERT INTO takes(
                    id,line_id,parent_take_id,rel_path,sha256,seed,preset,model_revision,
                    quality_json,status,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    take_id,
                    line_id,
                    parent_take_id,
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
                connection.execute(
                    "UPDATE takes SET status='candidate' WHERE line_id=?", (line_id,)
                )
                connection.execute("UPDATE takes SET status='adopted' WHERE id=?", (take_id,))
                connection.execute(
                    "UPDATE script_lines SET current_take_id=?,status='completed',dirty=1,updated_at=? "
                    "WHERE id=?",
                    (take_id, now, line_id),
                )
            row = connection.execute("SELECT * FROM takes WHERE id=?", (take_id,)).fetchone()
        self._touch("take", take_id, "created", {"line_id": line_id, "adopted": adopt})
        return _take_dict(row)

    def adopt_take(self, take_id: str) -> dict[str, Any]:
        now = utc_now()
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM takes WHERE id=?", (take_id,)).fetchone()
            if row is None:
                raise WorkerProtocolError(f"take 不存在：{take_id}")
            connection.execute(
                "UPDATE takes SET status='candidate' WHERE line_id=?", (row["line_id"],)
            )
            connection.execute("UPDATE takes SET status='adopted' WHERE id=?", (take_id,))
            connection.execute(
                "UPDATE script_lines SET current_take_id=?,status='completed',dirty=1,updated_at=? "
                "WHERE id=?",
                (take_id, now, row["line_id"]),
            )
            updated = connection.execute("SELECT * FROM takes WHERE id=?", (take_id,)).fetchone()
        self._touch("take", take_id, "adopted", {"line_id": row["line_id"]})
        return _take_dict(updated)

    def list_takes(self, line_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM takes WHERE line_id=? ORDER BY created_at DESC", (line_id,)
            ).fetchall()
        return [_take_dict(row) for row in rows]

    def list_project_artifacts(self) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        with self.connection() as connection:
            takes = connection.execute(
                """
                SELECT t.*,l.speaker,l.text,l.order_index
                FROM takes AS t JOIN script_lines AS l ON l.id=t.line_id
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
        normalized = [_timeline_clip_fields(value) for value in items]
        results: list[dict[str, Any]] = []
        with self.connection() as connection:
            for fields in normalized:
                connection.execute(
                    """
                    INSERT INTO timeline_clips(
                        id,line_id,take_id,track,position,duration,in_offset,out_offset,
                        gain_db,fade_in,fade_out,muted,version,updated_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(id) DO UPDATE SET
                        line_id=excluded.line_id,take_id=excluded.take_id,track=excluded.track,
                        position=excluded.position,duration=excluded.duration,
                        in_offset=excluded.in_offset,out_offset=excluded.out_offset,
                        gain_db=excluded.gain_db,fade_in=excluded.fade_in,
                        fade_out=excluded.fade_out,muted=excluded.muted,
                        version=timeline_clips.version+1,updated_at=excluded.updated_at
                    """,
                    fields,
                )
                row = connection.execute(
                    "SELECT * FROM timeline_clips WHERE id=?", (fields[0],)
                ).fetchone()
                if row is None:
                    raise WorkerProtocolError(f"时间轴片段保存失败：{fields[0]}")
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
                "SELECT * FROM timeline_clips ORDER BY position,track,id"
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
                ORDER BY c.position,c.track,c.id
                """
            ).fetchall()
        values: list[dict[str, Any]] = []
        for row in rows:
            value = dict(row)
            value["path"] = str(safe_project_path(self.root, value.pop("take_rel_path")))
            values.append(value)
        return values

    def record_render(
        self,
        output_path: str | Path,
        *,
        manifest_path: str | Path | None = None,
        quality_report: dict[str, Any] | None = None,
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
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO renders(id,rel_path,sha256,manifest_rel_path,quality_report_json,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (
                    render_id,
                    relative,
                    digest,
                    manifest_relative,
                    _json(quality_report or {}),
                    created,
                ),
            )
            connection.execute("UPDATE script_lines SET dirty=0,updated_at=?", (created,))
            row = connection.execute("SELECT * FROM renders WHERE id=?", (render_id,)).fetchone()
        self._touch("render", render_id, "created", {"sha256": digest})
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
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.manifest_path)


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
            seed INTEGER,
            preset TEXT NOT NULL DEFAULT 'balanced',
            target_start REAL,
            target_end REAL,
            status TEXT NOT NULL DEFAULT 'draft',
            current_take_id TEXT,
            dirty INTEGER NOT NULL DEFAULT 1,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(voice_profile_id) REFERENCES voice_profiles(id) ON DELETE SET NULL
        );
        CREATE TABLE IF NOT EXISTS takes(
            id TEXT PRIMARY KEY,
            line_id TEXT NOT NULL,
            parent_take_id TEXT,
            rel_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            seed INTEGER,
            preset TEXT NOT NULL DEFAULT 'balanced',
            model_revision TEXT,
            quality_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'candidate',
            created_at TEXT NOT NULL,
            FOREIGN KEY(line_id) REFERENCES script_lines(id) ON DELETE CASCADE,
            FOREIGN KEY(parent_take_id) REFERENCES takes(id) ON DELETE SET NULL
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


def _line_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["dirty"] = bool(result["dirty"])
    result["metadata"] = _read_json(result.pop("metadata_json"), {})
    return result


def _job_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["payload"] = _read_json(result.pop("payload_json"), {})
    return result


def _take_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["quality"] = _read_json(result.pop("quality_json"), {})
    return result


def _transcript_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["overlap_warning"] = bool(result["overlap_warning"])
    return result
