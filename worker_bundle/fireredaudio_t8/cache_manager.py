from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any


def cache_root() -> Path:
    return (Path(tempfile.gettempdir()) / "fireredaudio-t8" / "decoded").resolve()


def cache_status() -> dict[str, Any]:
    root = cache_root()
    files = [path for path in root.glob("*.wav") if path.is_file()] if root.exists() else []
    total = sum(path.stat().st_size for path in files)
    oldest = min((path.stat().st_mtime for path in files), default=None)
    return {
        "path": str(root),
        "file_count": len(files),
        "total_bytes": total,
        "total_mib": round(total / 1024**2, 2),
        "oldest_age_hours": (
            round((time.time() - oldest) / 3600, 2) if oldest is not None else None
        ),
    }


def cleanup_cache(
    *, max_age_hours: float = 72.0, max_size_mib: float = 2048.0, clear_all: bool = False
) -> dict[str, Any]:
    root = cache_root()
    if not root.exists():
        return {"removed_files": 0, "removed_bytes": 0, "status": cache_status()}
    now = time.time()
    files = sorted(
        (path for path in root.glob("*.wav") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
    )
    removed_files = 0
    removed_bytes = 0
    retained: list[Path] = []
    for path in files:
        stat = path.stat()
        expired = (now - stat.st_mtime) / 3600 > max(0.0, max_age_hours)
        if clear_all or expired:
            removed_bytes += stat.st_size
            removed_files += 1
            path.unlink(missing_ok=True)
        else:
            retained.append(path)

    limit = max(0, int(max_size_mib * 1024**2))
    retained_size = sum(path.stat().st_size for path in retained if path.exists())
    for path in retained:
        if retained_size <= limit:
            break
        if not path.exists():
            continue
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        retained_size -= size
        removed_bytes += size
        removed_files += 1
    return {
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
        "status": cache_status(),
    }
