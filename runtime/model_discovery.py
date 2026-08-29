from __future__ import annotations

import hashlib
import json
from pathlib import Path

MISSING_MODEL_OPTION = "未找到模型（请运行 scripts/download_models.py）"
LOCAL_MANIFEST_NAMES = ("fireredaudio-model.json", "model-package.json")


def manifest() -> dict:
    path = Path(__file__).resolve().parents[1] / "manifests" / "model_firered_audio.json"
    return json.loads(path.read_text(encoding="utf-8"))


def local_manifest(root: Path) -> dict | None:
    """Return model-package metadata written by the T8 conversion tools, if present."""
    for name in LOCAL_MANIFEST_NAMES:
        path = root / name
        if not path.is_file():
            continue
        try:
            definition = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"模型清单无法读取：{path.name}：{exc}") from exc
        if not isinstance(definition.get("files"), dict):
            raise ValueError(f"模型清单缺少 files：{path.name}")
        return definition
    return None


def validation_manifest(root: Path) -> dict:
    """Prefer an immutable local package manifest over the upstream BF16 manifest."""
    return local_manifest(root) or manifest()


def register_model_paths() -> None:
    try:
        import folder_paths

        base = Path(folder_paths.models_dir) / "TTS"
        base.mkdir(parents=True, exist_ok=True)
        try:
            folder_paths.add_model_folder_path("TTS", str(base))
        except TypeError:
            folder_paths.add_model_folder_path("TTS", str(base), is_default=True)
    except Exception:
        return


def search_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        import folder_paths

        roots.extend(Path(item) for item in folder_paths.get_folder_paths("TTS"))
    except Exception:
        pass
    return list(dict.fromkeys(root.resolve() for root in roots if root.exists()))


def discover_models() -> dict[str, Path]:
    matches: list[Path] = []
    seen: set[Path] = set()
    for base in sorted(search_roots(), key=lambda item: str(item).lower()):
        candidates = [
            base,
            *sorted(
                (item for item in base.iterdir() if item.is_dir()),
                key=lambda item: item.name.lower(),
            ),
        ]
        for candidate in candidates:
            if (candidate / "FireRedAudio" / "config.json").is_file():
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    matches.append(resolved)
    name_counts: dict[str, int] = {}
    for candidate in matches:
        name_counts[candidate.name] = name_counts.get(candidate.name, 0) + 1
    found: dict[str, Path] = {}
    for candidate in matches:
        label = (
            candidate.name
            if name_counts[candidate.name] == 1
            else f"{candidate.name} · {candidate}"
        )
        found[label] = candidate
    return dict(sorted(found.items()))


def model_options() -> list[str]:
    values = list(discover_models())
    return values or [MISSING_MODEL_OPTION]


def resolve_model(name: str, custom_path: str = "") -> Path:
    if custom_path.strip():
        return Path(custom_path).expanduser().resolve()
    found = discover_models()
    if name not in found:
        raise FileNotFoundError("未找到 FireRedAudio 模型；请填写自定义路径或运行下载脚本")
    return found[name]


def validate_sizes(root: Path, profile: str = "full") -> dict:
    definition = validation_manifest(root)
    issues: list[dict] = []
    checked = 0
    for relative, metadata in definition["files"].items():
        if profile == "lite" and relative.startswith("RedAE_decoder/"):
            continue
        checked += 1
        target = root / relative
        if not target.is_file():
            issues.append({"path": relative, "problem": "missing"})
        elif target.stat().st_size != int(metadata["size"]):
            issues.append({"path": relative, "problem": "size", "expected": metadata["size"], "actual": target.stat().st_size})
    return {
        "valid": not issues,
        "checked_files": checked,
        "issues": issues,
        "model_profile": definition.get("profile", "upstream-bf16"),
        "model_format": definition.get("format", "safetensors-bf16"),
        "stable": bool(definition.get("stable", True)),
    }


def fingerprint(root: Path) -> str:
    definition = validation_manifest(root)
    digest = hashlib.sha256(str(root).encode("utf-8"))
    for relative in definition["files"]:
        target = root / relative
        if target.exists():
            stat = target.stat()
            digest.update(f"{relative}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()
