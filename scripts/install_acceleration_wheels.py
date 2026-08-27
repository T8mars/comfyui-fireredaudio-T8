"""Install the audited Windows acceleration wheels without compiling anything."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.request
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "desktop_acceleration_manifest.json"
CACHE = ROOT / ".acceleration-wheels"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(entry: dict[str, str]) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / entry["filename"]
    expected = entry["sha256"].lower()
    if target.exists() and sha256(target) == expected:
        print(f"Using verified cache: {target.name}")
        return target
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        partial.unlink()
    print(f"Downloading: {entry['url']}")
    request = urllib.request.Request(entry["url"], headers={"User-Agent": "FireRedAudio-T8"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    actual = sha256(partial)
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {entry['filename']}: expected {expected}, got {actual}"
        )
    os.replace(partial, target)
    return target


def verify_target(manifest: dict) -> None:
    target = manifest["target"]
    if os.name != "nt" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("Acceleration wheels only support Windows x64")
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if python_tag != target["pythonTag"]:
        raise RuntimeError(f"Expected {target['pythonTag']}, got {python_tag}")
    torch_version = metadata.version("torch")
    if not torch_version.startswith("2.8.0"):
        raise RuntimeError(f"Expected torch 2.8.0+cu128, got {torch_version}")


def install(uv: str, wheels: list[Path]) -> None:
    command = [
        uv,
        "pip",
        "install",
        "--python",
        sys.executable,
        "--no-deps",
        "--reinstall",
        *map(str, wheels),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def patch_deepspeed_windows_probe() -> None:
    """Prevent DeepSpeed's Linux libaio probe from invoking a compiler on Windows."""
    site_packages = Path(metadata.distribution("deepspeed").locate_file(""))
    source = site_packages / "deepspeed" / "ops" / "op_builder" / "async_io.py"
    text = source.read_text(encoding="utf-8")
    marker = "# FireRedAudio Windows: libaio is Linux-only; never compile-probe it."
    needle = "    def is_compatible(self, verbose=False):\n"
    replacement = (
        needle
        + f"        {marker}\n"
        + "        if os.name == 'nt':\n"
        + "            return False\n"
    )
    if marker not in text:
        if needle not in text:
            raise RuntimeError("Unsupported DeepSpeed AsyncIOBuilder layout; refusing an unsafe patch")
        source.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
        print(f"Patched Linux-only DeepSpeed libaio probe: {source}")

    gds_source = site_packages / "deepspeed" / "ops" / "op_builder" / "gds.py"
    gds_text = gds_source.read_text(encoding="utf-8")
    gds_marker = "# FireRedAudio Windows: GPUDirect Storage probe requires cufile.lib."
    if gds_marker not in gds_text:
        gds_needle = "    def is_compatible(self, verbose=False):\n"
        gds_replacement = (
            gds_needle
            + f"        {gds_marker}\n"
            + "        if os.name == 'nt':\n"
            + "            return False\n"
        )
        if gds_needle not in gds_text:
            raise RuntimeError("Unsupported DeepSpeed GDSBuilder layout; refusing an unsafe patch")
        gds_source.write_text(
            gds_text.replace(gds_needle, gds_replacement, 1), encoding="utf-8"
        )
        print(f"Patched Linux-only DeepSpeed GDS probe: {gds_source}")


def verify_installed(manifest: dict) -> None:
    failures: list[str] = []
    for entry in manifest["packages"]:
        try:
            actual = metadata.version(entry["distribution"])
        except metadata.PackageNotFoundError:
            failures.append(f"{entry['distribution']}: missing")
            continue
        if not actual.startswith(entry["version"]):
            failures.append(
                f"{entry['distribution']}: expected {entry['version']}, got {actual}"
            )
    if failures:
        raise RuntimeError("Acceleration install validation failed: " + "; ".join(failures))
    print("Acceleration metadata validated. Runtime kernels are validated by check_acceleration.py.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    verify_target(manifest)
    wheels = [download(entry) for entry in manifest["packages"]]
    install(args.uv, wheels)
    patch_deepspeed_windows_probe()
    verify_installed(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
