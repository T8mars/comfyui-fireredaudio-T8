from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fireredaudio_t8.constants import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    MODELSCOPE_MODEL_REPOSITORY,
    MODELSCOPE_MODEL_REVISION,
)
from fireredaudio_t8.model_manager import (
    local_manifest_path,
    load_manifest,
    required_file_entries,
    validate_model_dir,
)


MODEL_VARIANTS = {
    "int8-wo-safe-v1": {
        "revision": "09ac8f5b1e74872a38a67a3f02e4917bf5a9668f",
        "profile": "int8-wo-safe",
        "huggingface_only": True,
    },
    "bf16-slim-v1": {
        "revision": "979e1a8ea463b7559518916d46b851785e11900f",
        "profile": "bf16-slim",
        "huggingface_only": True,
    },
    "int8-wo-extended-v1": {
        "revision": "97540c75dd4438fe4615c9ac74b1adb77854abd5",
        "profile": "int8-wo-extended",
        "huggingface_only": True,
    },
    "int8-convrot-experimental-v1": {
        "revision": "2991b5bbd45e7363bcfb4b4ce3aa2667829ce539",
        "profile": "int8-convrot-experimental",
        "huggingface_only": True,
    },
    "upstream-bf16": {
        "revision": MODEL_REVISION,
        "profile": "upstream-bf16",
        "huggingface_only": False,
    },
}
DEFAULT_MODEL_VARIANT = "int8-wo-safe-v1"


def emit_progress(phase: str, progress: float, message: str, **extra: object) -> None:
    payload = {"phase": phase, "progress": progress, "message": message, **extra}
    print("FIREREDAUDIO_PROGRESS " + json.dumps(payload, ensure_ascii=False), flush=True)


def check_disk_space(
    target: Path,
    profile: str,
    manifest: dict | None = None,
) -> tuple[int, int]:
    manifest = manifest or load_manifest()
    remaining = 0
    for relative, metadata in required_file_entries(manifest, profile):
        expected = int(metadata["size"])
        file_path = target / relative
        present = min(file_path.stat().st_size, expected) if file_path.is_file() else 0
        remaining += expected - present
    free = shutil.disk_usage(target).free
    reserve = 2 * 1024**3
    if free < remaining + reserve:
        raise RuntimeError(
            f"磁盘空间不足：至少还需 {remaining / 1024**3:.2f} GiB，"
            f"并预留 2 GiB；当前可用 {free / 1024**3:.2f} GiB"
        )
    return remaining, free


def download_progress_snapshot(
    target: Path,
    profile: str,
    manifest: dict | None = None,
) -> tuple[int, int]:
    manifest = manifest or _target_manifest(target)
    total = 0
    complete = 0
    for relative, metadata in required_file_entries(manifest, profile):
        expected = int(metadata["size"])
        total += expected
        path = target / relative
        if path.is_file():
            complete += min(path.stat().st_size, expected)
    partial = 0
    for pattern in ("*.incomplete", "*.partial", "*.part"):
        partial += sum(
            path.stat().st_size for path in target.rglob(pattern) if path.is_file()
        )
    return min(total, complete + partial), total


def run_with_progress(
    target: Path,
    profile: str,
    source: str,
    manifest: dict,
    action,
) -> None:
    stop = threading.Event()

    def monitor() -> None:
        previous = -1
        while not stop.wait(1.0):
            downloaded, total = download_progress_snapshot(target, profile, manifest)
            if downloaded == previous or total <= 0:
                continue
            previous = downloaded
            ratio = downloaded / total
            emit_progress(
                "download",
                0.08 + 0.82 * ratio,
                f"{source} 下载中：{downloaded / 1024**3:.2f} / {total / 1024**3:.2f} GiB",
                source=source,
                downloaded_bytes=downloaded,
                total_bytes=total,
                remaining_bytes=max(0, total - downloaded),
            )

    thread = threading.Thread(target=monitor, name="model-download-progress", daemon=True)
    thread.start()
    try:
        action()
    finally:
        stop.set()
        thread.join(timeout=2.0)
        downloaded, total = download_progress_snapshot(target, profile, manifest)
        emit_progress(
            "download",
            0.9 if total and downloaded >= total else 0.08 + 0.82 * (downloaded / total if total else 0),
            f"{source} 下载阶段结束",
            source=source,
            downloaded_bytes=downloaded,
            total_bytes=total,
            remaining_bytes=max(0, total - downloaded),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并校验 FireRedAudio 固定版本模型")
    parser.add_argument("--target", required=True, help="模型根目录")
    parser.add_argument("--profile", choices=["lite", "full"], default="full")
    parser.add_argument(
        "--variant",
        choices=list(MODEL_VARIANTS),
        default=DEFAULT_MODEL_VARIANT,
        help="Hugging Face 模型 revision；默认使用稳定 INT8",
    )
    parser.add_argument(
        "--source", choices=["auto", "modelscope", "huggingface"], default="auto"
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--full-hash", action="store_true")
    return parser.parse_args()


def patterns(profile: str, variant: str) -> list[str]:
    selected = ["FireRedAudio/*"]
    if profile == "full":
        selected.append("RedAE_decoder/*")
    if variant != "upstream-bf16":
        selected.extend(["fireredaudio-model.json", "README.md"])
    return selected


def download_huggingface(target: Path, profile: str, variant: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("缺少 huggingface_hub；请先安装隔离运行时依赖") from exc
    snapshot_download(
        repo_id=MODEL_REPOSITORY,
        revision=str(MODEL_VARIANTS[variant]["revision"]),
        local_dir=str(target),
        allow_patterns=patterns(profile, variant),
    )


def download_modelscope(target: Path, profile: str, variant: str) -> None:
    if variant != "upstream-bf16":
        raise RuntimeError("量化/瘦身模型仅发布在 T8star-Aix Hugging Face 仓库")
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError("缺少 modelscope；请安装 modelscope 后重试") from exc
    snapshot_download(
        MODELSCOPE_MODEL_REPOSITORY,
        revision=MODELSCOPE_MODEL_REVISION,
        local_dir=str(target),
        allow_patterns=patterns(profile, variant),
    )


def _target_manifest(target: Path) -> dict:
    local = local_manifest_path(target)
    return load_manifest(local) if local else load_manifest()


def prepare_download_manifest(target: Path, source: str, variant: str) -> dict:
    """Fetch the tiny immutable package manifest before disk preflight."""
    expected_profile = str(MODEL_VARIANTS[variant]["profile"])
    existing = local_manifest_path(target)
    if existing is not None:
        existing_profile = str(load_manifest(existing).get("profile") or "upstream-bf16")
        if existing_profile != expected_profile:
            raise RuntimeError(
                "目标目录已有其他模型版本 "
                f"{existing_profile}；切换到 {expected_profile} 时请选择新的空目录，"
                "避免混合权重分片"
            )
    elif variant != "upstream-bf16" and (
        target / "FireRedAudio" / "config.json"
    ).is_file():
        raise RuntimeError(
            "目标目录已有未带本地清单的旧 BF16 模型；下载量化版时请选择新的空目录，"
            "避免混合权重分片"
        )
    if variant == "upstream-bf16":
        return load_manifest()
    if source != "huggingface":
        raise RuntimeError("所选 INT8/瘦身版本只能从 Hugging Face 下载")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("缺少 huggingface_hub；请先安装隔离运行时依赖") from exc
    snapshot_download(
        repo_id=MODEL_REPOSITORY,
        revision=str(MODEL_VARIANTS[variant]["revision"]),
        local_dir=str(target),
        allow_patterns=["fireredaudio-model.json"],
    )
    local = local_manifest_path(target)
    if local is None:
        raise RuntimeError(f"模型版本 {variant} 缺少 fireredaudio-model.json")
    definition = load_manifest(local)
    if str(definition.get("profile") or "") != expected_profile:
        raise RuntimeError(
            "模型清单与所选版本不一致："
            f"{definition.get('profile')} != {expected_profile}"
        )
    return definition


def main() -> int:
    args = parse_args()
    target = Path(args.target).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        errors: list[str] = []
        sources = (
            [args.source]
            if args.source != "auto"
            else (
                ["huggingface"]
                if MODEL_VARIANTS[args.variant]["huggingface_only"]
                else ["huggingface", "modelscope"]
            )
        )
        for source in sources:
            try:
                emit_progress(
                    "preflight",
                    0.02,
                    f"正在读取 {args.variant} 固定清单并检查磁盘空间",
                    source=source,
                    variant=args.variant,
                )
                definition = prepare_download_manifest(
                    target, source, args.variant
                )
                remaining, free = check_disk_space(
                    target, args.profile, definition
                )
                emit_progress(
                    "preflight",
                    0.05,
                    "磁盘空间检查通过",
                    remaining_bytes=remaining,
                    free_bytes=free,
                    variant=args.variant,
                )
                emit_progress(
                    "download",
                    0.1,
                    f"正在通过 {source} 下载 {args.variant}",
                    source=source,
                    variant=args.variant,
                )
                if source == "modelscope":
                    run_with_progress(
                        target,
                        args.profile,
                        source,
                        definition,
                        lambda: download_modelscope(
                            target, args.profile, args.variant
                        ),
                    )
                else:
                    run_with_progress(
                        target,
                        args.profile,
                        source,
                        definition,
                        lambda: download_huggingface(
                            target, args.profile, args.variant
                        ),
                    )
                break
            except Exception as exc:
                emit_progress("source_fallback", 0.12, f"{source} 不可用，准备切换下载源", source=source)
                errors.append(f"{source}: {exc}")
        else:
            raise RuntimeError("所有模型源均下载失败：" + " | ".join(errors))

    emit_progress("validate", 0.92, "下载结束，正在校验模型文件")
    report = validate_model_dir(
        target,
        profile=args.profile,
        verify_hashes=args.full_hash,
        manifest=load_manifest(),
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    emit_progress(
        "complete" if report.valid else "failed",
        1.0 if report.valid else 0.96,
        "模型校验通过" if report.valid else "模型校验失败",
        valid=report.valid,
    )
    return 0 if report.valid else 2


if __name__ == "__main__":
    sys.exit(main())
