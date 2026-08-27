from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .production import file_digest, stable_digest, write_manifest

ASR_CACHE_SCHEMA_VERSION = 1


def build_asr_cache_descriptor(
    audio_path: str | Path,
    *,
    model_revision: str,
    model_fingerprint: str,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Build the complete identity for one deterministic ASR transcript."""

    source = Path(audio_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ASR 缓存源音频不存在：{source}")
    identity = {
        "schema_version": ASR_CACHE_SCHEMA_VERSION,
        "audio_sha256": file_digest(source),
        "model_revision": str(model_revision),
        "model_fingerprint": str(model_fingerprint),
        "prompt": str(prompt),
        "max_new_tokens": int(max_new_tokens),
    }
    return {**identity, "cache_key": stable_digest(identity)}


def cache_path(cache_root: str | Path, cache_key: str) -> Path:
    key = str(cache_key).lower()
    if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
        raise ValueError("ASR 缓存键必须是 64 位十六进制 SHA-256")
    root = Path(cache_root).resolve()
    target = (root / key[:2] / f"{key}.json").resolve()
    target.relative_to(root)
    return target


def load_cached_transcript(
    cache_root: str | Path, descriptor: dict[str, Any]
) -> dict[str, Any] | None:
    """Return a validated cached transcript, or None for any stale/corrupt entry."""

    key = str(descriptor.get("cache_key") or "")
    try:
        target = cache_path(cache_root, key)
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != ASR_CACHE_SCHEMA_VERSION:
        return None
    if payload.get("cache_key") != key or payload.get("descriptor") != descriptor:
        return None
    transcript = payload.get("transcript")
    if not isinstance(transcript, str):
        return None
    return {
        "transcript": transcript,
        "cache_path": str(target),
        "created_at": payload.get("created_at"),
    }


def store_cached_transcript(
    cache_root: str | Path,
    descriptor: dict[str, Any],
    transcript: str,
) -> Path:
    key = str(descriptor.get("cache_key") or "")
    target = cache_path(cache_root, key)
    write_manifest(
        target,
        {
            "schema_version": ASR_CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "descriptor": descriptor,
            "transcript": str(transcript),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return target


__all__ = [
    "ASR_CACHE_SCHEMA_VERSION",
    "build_asr_cache_descriptor",
    "cache_path",
    "load_cached_transcript",
    "store_cached_transcript",
]
