from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fireredaudio_model_discovery", ROOT / "runtime" / "model_discovery.py"
)
MODEL_DISCOVERY = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODEL_DISCOVERY
SPEC.loader.exec_module(MODEL_DISCOVERY)

fingerprint = MODEL_DISCOVERY.fingerprint
local_manifest = MODEL_DISCOVERY.local_manifest
validate_sizes = MODEL_DISCOVERY.validate_sizes


def write_manifest(root: Path, files: dict, *, profile: str = "int8-wo-safe") -> None:
    (root / "fireredaudio-model.json").write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "profile": profile,
                "format": "torchao-int8-weight-only",
                "stable": True,
                "files": files,
            }
        ),
        encoding="utf-8",
    )


def test_local_manifest_accepts_slim_decoder(tmp_path: Path) -> None:
    main = tmp_path / "FireRedAudio" / "model.safetensors"
    decoder = tmp_path / "RedAE_decoder" / "model.safetensors"
    main.parent.mkdir(parents=True)
    decoder.parent.mkdir(parents=True)
    main.write_bytes(b"main-int8")
    decoder.write_bytes(b"decoder-only")
    files = {
        "FireRedAudio/model.safetensors": {"size": main.stat().st_size},
        "RedAE_decoder/model.safetensors": {"size": decoder.stat().st_size},
    }
    write_manifest(tmp_path, files)

    definition = local_manifest(tmp_path)
    report = validate_sizes(tmp_path, "full")

    assert definition is not None
    assert definition["profile"] == "int8-wo-safe"
    assert report["valid"] is True
    assert report["checked_files"] == 2
    assert report["model_format"] == "torchao-int8-weight-only"


def test_lite_validation_skips_decoder_from_local_manifest(tmp_path: Path) -> None:
    main = tmp_path / "FireRedAudio" / "model.safetensors"
    main.parent.mkdir(parents=True)
    main.write_bytes(b"main")
    write_manifest(
        tmp_path,
        {
            "FireRedAudio/model.safetensors": {"size": main.stat().st_size},
            "RedAE_decoder/model.safetensors": {"size": 12345},
        },
    )

    assert validate_sizes(tmp_path, "lite")["valid"] is True
    assert validate_sizes(tmp_path, "full")["issues"] == [
        {"path": "RedAE_decoder/model.safetensors", "problem": "missing"}
    ]


def test_fingerprint_changes_when_local_quantized_file_changes(tmp_path: Path) -> None:
    weight = tmp_path / "FireRedAudio" / "model.safetensors"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"first")
    write_manifest(
        tmp_path,
        {"FireRedAudio/model.safetensors": {"size": weight.stat().st_size}},
    )
    before = fingerprint(tmp_path)
    weight.write_bytes(b"second-version")
    after = fingerprint(tmp_path)

    assert before != after


def test_invalid_local_manifest_fails_explicitly(tmp_path: Path) -> None:
    (tmp_path / "fireredaudio-model.json").write_text("{broken", encoding="utf-8")
    try:
        validate_sizes(tmp_path)
    except ValueError as exc:
        assert "模型清单无法读取" in str(exc)
    else:
        raise AssertionError("invalid local manifests must not silently fall back")
