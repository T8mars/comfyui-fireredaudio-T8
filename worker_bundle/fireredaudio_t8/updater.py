from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import WorkerProtocolError


VERSION_RE = re.compile(r"^[0-9]+(?:\.[0-9]+){1,3}(?:[-+][0-9A-Za-z.-]+)?$")
EXECUTABLE_NAME = "T8star-Aix-FireRedAudio.exe"


def inspect_update_manifest(manifest_path: str | Path, package_path: str | Path) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    package = Path(package_path).expanduser().resolve()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkerProtocolError(f"更新 manifest 无法读取：{exc}") from exc
    if not isinstance(manifest, dict):
        raise WorkerProtocolError("更新 manifest 必须是 JSON 对象")
    version = str(manifest.get("version") or "").strip()
    if not VERSION_RE.fullmatch(version):
        raise WorkerProtocolError(f"更新版本号无效：{version!r}")
    channel = str(manifest.get("channel") or "stable")
    if channel not in {"stable", "testing"}:
        raise WorkerProtocolError("更新通道必须是 stable/testing")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise WorkerProtocolError("更新 manifest 缺少 artifacts")
    package_name = package.name.lower()
    artifact = next(
        (
            item
            for item in artifacts
            if isinstance(item, dict)
            and str(item.get("name") or "").lower() == package_name
        ),
        None,
    )
    if artifact is None:
        desktop_artifacts = [
            item
            for item in artifacts
            if isinstance(item, dict) and "win64" in str(item.get("name") or "").lower()
        ]
        if len(desktop_artifacts) == 1:
            artifact = desktop_artifacts[0]
        else:
            raise WorkerProtocolError("更新包未出现在 manifest artifacts 中")
    if not package.is_file():
        raise WorkerProtocolError(f"更新包不存在：{package}")
    expected_size = int(artifact.get("size") or 0)
    actual_size = package.stat().st_size
    if expected_size and expected_size != actual_size:
        raise WorkerProtocolError(
            f"更新包大小不匹配：期望 {expected_size}，实际 {actual_size}"
        )
    expected_hash = str(artifact.get("sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise WorkerProtocolError("更新 manifest 缺少有效 SHA-256")
    actual_hash = file_sha256(package)
    if actual_hash != expected_hash:
        raise WorkerProtocolError(
            f"更新包 SHA-256 不匹配：期望 {expected_hash}，实际 {actual_hash}"
        )
    return {
        "version": version,
        "channel": channel,
        "publisher": str(manifest.get("publisher") or "T8star-Aix"),
        "source": str(manifest.get("source") or "local-release-manifest"),
        "package": str(package),
        "size": actual_size,
        "sha256": actual_hash,
    }


def install_update(
    install_root: str | Path,
    manifest_path: str | Path,
    package_path: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(install_root).expanduser().resolve()
    report = inspect_update_manifest(manifest_path, package_path)
    versions = root / "versions"
    updates = root / ".updates"
    state_path = root / "update-state.json"
    required = int(report["size"] * 2.2) + 64 * 1024 * 1024
    free = shutil.disk_usage(root if root.exists() else root.parent).free
    if free < required:
        raise WorkerProtocolError(
            f"更新磁盘空间不足：需要约 {required} 字节，可用 {free} 字节"
        )
    target = _safe_version_path(versions, report["version"])
    state = load_update_state(root)
    if dry_run:
        return {
            **report,
            "dry_run": True,
            "install_root": str(root),
            "target": str(target),
            "current": state.get("current"),
            "required_free_bytes": required,
            "available_free_bytes": free,
        }
    root.mkdir(parents=True, exist_ok=True)
    versions.mkdir(exist_ok=True)
    updates.mkdir(exist_ok=True)
    staging = updates / f"staging-{report['version']}-{uuid.uuid4().hex}"
    staged_version = staging / "version"
    try:
        staging.mkdir(parents=True)
        extract_root = staging / "extract"
        _safe_extract_zip(Path(report["package"]), extract_root)
        application = _locate_application_directory(extract_root, report["version"])
        shutil.copytree(application, staged_version)
        executable = staged_version / EXECUTABLE_NAME
        if not executable.is_file():
            raise WorkerProtocolError(f"更新包缺少 {EXECUTABLE_NAME}")
        if target.exists():
            existing = target / EXECUTABLE_NAME
            if not existing.is_file():
                raise WorkerProtocolError(f"已存在的版本目录不完整：{target}")
            _remove_tree_checked(staging, updates)
        else:
            staged_version.replace(target)
            _remove_tree_checked(staging, updates)
        previous = state.get("current")
        history = [report["version"]]
        for version in state.get("history", []):
            if version != report["version"] and VERSION_RE.fullmatch(str(version)):
                history.append(str(version))
        new_state = {
            "schema_version": 1,
            "current": report["version"],
            "previous": previous if previous != report["version"] else state.get("previous"),
            "channel": report["channel"],
            "publisher": report["publisher"],
            "source": report["source"],
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "history": history[:2],
        }
        _write_state_atomic(state_path, new_state)
        _prune_versions(versions, set(new_state["history"]))
        return {**report, "installed": True, "state": new_state, "target": str(target)}
    except Exception:
        if staging.exists():
            _remove_tree_checked(staging, updates)
        raise


def rollback_update(install_root: str | Path) -> dict[str, Any]:
    root = Path(install_root).expanduser().resolve()
    state = load_update_state(root)
    current = str(state.get("current") or "")
    previous = str(state.get("previous") or "")
    if not VERSION_RE.fullmatch(previous):
        raise WorkerProtocolError("没有可回滚的上一版本")
    target = _safe_version_path(root / "versions", previous)
    if not (target / EXECUTABLE_NAME).is_file():
        raise WorkerProtocolError(f"回滚版本不完整：{target}")
    new_state = dict(state)
    new_state.update(
        {
            "current": previous,
            "previous": current if VERSION_RE.fullmatch(current) else None,
            "rolled_back_at": datetime.now(timezone.utc).isoformat(),
            "history": [previous, current] if VERSION_RE.fullmatch(current) else [previous],
        }
    )
    _write_state_atomic(root / "update-state.json", new_state)
    return {"rolled_back": True, "state": new_state, "target": str(target)}


def load_update_state(install_root: str | Path) -> dict[str, Any]:
    path = Path(install_root).expanduser().resolve() / "update-state.json"
    if not path.is_file():
        return {"schema_version": 1, "current": None, "previous": None, "history": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkerProtocolError(f"更新状态文件损坏：{exc}") from exc
    if not isinstance(value, dict):
        raise WorkerProtocolError("更新状态文件格式无效")
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            candidate = (base / Path(info.filename)).resolve()
            try:
                candidate.relative_to(base)
            except ValueError as exc:
                raise WorkerProtocolError(f"更新包包含越界路径：{info.filename}") from exc
            candidate.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, candidate.open("wb") as output:
                shutil.copyfileobj(source, output)


def _locate_application_directory(extract_root: Path, version: str) -> Path:
    direct_candidates = [
        path.parent
        for path in extract_root.rglob(EXECUTABLE_NAME)
        if path.is_file()
    ]
    preferred = [path for path in direct_candidates if version in path.parts]
    candidates = preferred or direct_candidates
    if len(candidates) != 1:
        raise WorkerProtocolError(
            f"更新包必须且只能包含一个 {EXECUTABLE_NAME}"
        )
    return candidates[0]


def _safe_version_path(versions_root: Path, version: str) -> Path:
    if not VERSION_RE.fullmatch(str(version)):
        raise WorkerProtocolError(f"版本号无效：{version}")
    root = versions_root.resolve()
    candidate = (root / str(version)).resolve()
    if candidate.parent != root:
        raise WorkerProtocolError("版本目录越界")
    return candidate


def _remove_tree_checked(path: Path, allowed_parent: Path) -> None:
    target = path.resolve()
    parent = allowed_parent.resolve()
    try:
        target.relative_to(parent)
    except ValueError as exc:
        raise WorkerProtocolError("拒绝删除更新目录之外的路径") from exc
    if target == parent:
        raise WorkerProtocolError("拒绝删除整个更新目录")
    shutil.rmtree(target)


def _write_state_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _prune_versions(versions_root: Path, keep: set[str]) -> None:
    if not versions_root.is_dir():
        return
    for candidate in versions_root.iterdir():
        if not candidate.is_dir() or candidate.name in keep:
            continue
        if VERSION_RE.fullmatch(candidate.name):
            _remove_tree_checked(candidate, versions_root)
