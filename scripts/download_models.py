from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="使用隔离运行时下载 FireRedAudio 模型")
    parser.add_argument("--target", required=True)
    parser.add_argument("--profile", choices=["lite", "full"], default="full")
    parser.add_argument("--source", choices=["auto", "modelscope", "huggingface"], default="auto")
    parser.add_argument("--full-hash", action="store_true")
    parser.add_argument("--python", default="")
    args = parser.parse_args()
    node_root = Path(__file__).resolve().parents[1]
    python = Path(args.python).expanduser().resolve() if args.python else node_root / ".runtime" / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise SystemExit("未找到隔离运行时；请先运行 scripts/setup_runtime.py")
    source = node_root / "worker_bundle"
    script = source / "scripts" / "download_models.py"
    if not script.is_file():
        raise SystemExit("worker_bundle 不完整，请重新安装节点发行包")
    command = [str(python), str(script), "--target", args.target, "--profile", args.profile, "--source", args.source]
    if args.full_hash:
        command.append("--full-hash")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(source), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return subprocess.run(command, cwd=source, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
