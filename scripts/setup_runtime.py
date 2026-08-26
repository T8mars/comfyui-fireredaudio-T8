from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="准备不污染 ComfyUI 的 FireRedAudio 隔离运行时")
    parser.add_argument("--target", default="", help="默认安装到节点 .runtime")
    parser.add_argument("--uv", default="", help="uv 可执行文件；默认优先使用节点随附版本")
    args = parser.parse_args()
    node_root = Path(__file__).resolve().parents[1]
    target = Path(args.target).expanduser().resolve() if args.target else node_root / ".runtime"
    venv = target / ".venv"
    bundled_uv = node_root / "tools" / ("uv.exe" if os.name == "nt" else "uv")
    uv = args.uv or (str(bundled_uv) if bundled_uv.is_file() else "uv")
    target.mkdir(parents=True, exist_ok=True)
    run([uv, "python", "install", "3.10"])
    run([uv, "venv", "--python", "3.10", str(venv)])
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run([uv, "pip", "install", "--python", str(python), "-r", str(node_root / "runtime-requirements.txt")])
    check = subprocess.run(
        [str(python), "-c", "import torch, transformers; print(torch.__version__, transformers.__version__)"],
        check=True,
        capture_output=True,
        text=True,
    )
    print("隔离运行时就绪：", python)
    print(check.stdout.strip())
    print("不会修改 ComfyUI Python：", sys.executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
