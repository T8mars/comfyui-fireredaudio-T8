from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .constants import GENERATION_TASKS
from .errors import ModelValidationError


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    problem: str
    expected: str | int | None = None
    actual: str | int | None = None


@dataclass(frozen=True)
class ValidationReport:
    root: str
    profile: str
    valid: bool
    hashes_verified: bool
    checked_files: int
    required_bytes: int
    issues: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issues"] = [asdict(issue) for issue in self.issues]
        return data

    def require_valid(self) -> "ValidationReport":
        if not self.valid:
            details = "; ".join(
                f"{item.path}: {item.problem}" for item in self.issues[:8]
            )
            raise ModelValidationError(f"模型目录校验失败：{details}")
        return self


def manifest_path() -> Path:
    return Path(__file__).resolve().parents[1] / "manifests" / "model_firered_audio.json"


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else manifest_path()
    return json.loads(target.read_text(encoding="utf-8"))


def normalize_model_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if (root / "FireRedAudio" / "config.json").is_file():
        return root
    if (root / "config.json").is_file() and root.name.lower() == "fireredaudio":
        return root.parent
    return root


def model_paths(root: str | Path) -> tuple[Path, Path]:
    normalized = normalize_model_root(root)
    return normalized / "FireRedAudio", normalized / "RedAE_decoder" / "model.pt"


def required_file_entries(
    manifest: dict[str, Any], profile: str = "full"
) -> Iterable[tuple[str, dict[str, Any]]]:
    normalized_profile = profile.lower().strip()
    if normalized_profile not in {"lite", "full"}:
        raise ValueError("profile 必须是 lite 或 full")
    for relative, metadata in manifest["files"].items():
        if normalized_profile == "lite" and relative.startswith("RedAE_decoder/"):
            continue
        yield relative, metadata


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_dir(
    root: str | Path,
    *,
    profile: str = "full",
    verify_hashes: bool = False,
    manifest: dict[str, Any] | None = None,
) -> ValidationReport:
    normalized = normalize_model_root(root)
    definition = manifest or load_manifest()
    issues: list[ValidationIssue] = []
    checked = 0
    total = 0
    for relative, metadata in required_file_entries(definition, profile):
        checked += 1
        total += int(metadata["size"])
        target = normalized / Path(relative)
        if not target.is_file():
            issues.append(ValidationIssue(relative, "缺少文件"))
            continue
        actual_size = target.stat().st_size
        expected_size = int(metadata["size"])
        if actual_size != expected_size:
            issues.append(
                ValidationIssue(relative, "文件大小不符", expected_size, actual_size)
            )
            continue
        expected_hash = metadata.get("sha256")
        if verify_hashes and expected_hash:
            actual_hash = sha256_file(target)
            if actual_hash.lower() != str(expected_hash).lower():
                issues.append(
                    ValidationIssue(relative, "SHA-256 不符", expected_hash, actual_hash)
                )
    return ValidationReport(
        root=str(normalized),
        profile=profile,
        valid=not issues,
        hashes_verified=bool(verify_hashes),
        checked_files=checked,
        required_bytes=total,
        issues=tuple(issues),
    )


def profile_for_task(task: str) -> str:
    return "full" if task in GENERATION_TASKS else "lite"
