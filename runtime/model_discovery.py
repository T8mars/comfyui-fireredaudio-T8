from __future__ import annotations

import hashlib
import json
from pathlib import Path

MISSING_MODEL_OPTION = "未找到模型（请运行 scripts/download_models.py）"


def manifest() -> dict:
    path = Path(__file__).resolve().parents[1] / "manifests" / "model_firered_audio.json"
    return json.loads(path.read_text(encoding="utf-8"))


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
    found: dict[str, Path] = {}
    for base in search_roots():
        candidates = [base, *[item for item in base.iterdir() if item.is_dir()]]
        for candidate in candidates:
            if (candidate / "FireRedAudio" / "config.json").is_file():
                found[candidate.name] = candidate.resolve()
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
    definition = manifest()
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
    return {"valid": not issues, "checked_files": checked, "issues": issues}


def fingerprint(root: Path) -> str:
    definition = manifest()
    digest = hashlib.sha256(str(root).encode("utf-8"))
    for relative in definition["files"]:
        target = root / relative
        if target.exists():
            stat = target.stat()
            digest.update(f"{relative}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()
