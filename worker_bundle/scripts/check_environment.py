from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fireredaudio_t8.model_manager import validate_model_dir


EXPECTED = {
    "torch": "2.8.0",
    "torchaudio": "2.8.0",
    "transformers": "5.8.0",
    "numpy": "2.2.6",
    "einops": "0.8.2",
}

OPTIONAL_PLATFORM = {"torchcodec": "0.11.1"}


def installed(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect(model_root: str | None = None) -> dict[str, Any]:
    packages = {name: installed(name) for name in EXPECTED | OPTIONAL_PLATFORM}
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "python_ok": sys.version_info[:2] == (3, 10),
        "platform": platform.platform(),
        "packages": packages,
        "package_compatibility": {
            name: version == expected or bool(version and version.startswith(expected + "+"))
            for name, expected in EXPECTED.items()
            if (version := packages[name]) is not None
        },
        "optional_platform_packages": OPTIONAL_PLATFORM,
    }
    try:
        import torch

        result["cuda"] = {
            "available": torch.cuda.is_available(),
            "runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "total_memory": torch.cuda.get_device_properties(index).total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        result["cuda"] = {"available": False, "error": str(exc)}
    if model_root:
        root = Path(model_root).expanduser().resolve()
        result["model_lite"] = validate_model_dir(root, profile="lite").to_dict()
        result["model_full"] = validate_model_dir(root, profile="full").to_dict()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="FireRedAudio T8 环境检查")
    parser.add_argument("model_root", nargs="?")
    args = parser.parse_args()
    report = collect(args.model_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    required_ok = report["python_ok"] and all(
        report["package_compatibility"].get(name, False) for name in EXPECTED
    )
    return 0 if required_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
