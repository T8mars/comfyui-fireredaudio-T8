from __future__ import annotations

import atexit
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from .types import RuntimeHandle
from .worker_client import WorkerClient


class WorkerManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._log_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._client: WorkerClient | None = None
        self._signature: tuple[str, ...] | None = None
        self._log_lines: list[str] = []
        self._log_thread: threading.Thread | None = None

    def client_for(self, handle: RuntimeHandle) -> WorkerClient:
        if handle.worker_url:
            if not handle.worker_token:
                raise RuntimeError("连接外部 Worker 时必须填写 Worker token")
            client = WorkerClient(handle.worker_url, handle.worker_token)
            client.health()
            return client
        with self._lock:
            python = self._resolve_python(handle.runtime_python)
            source = self._resolve_source()
            signature = (str(python), str(source))
            if self._client is not None and self._signature == signature:
                try:
                    self._client.health()
                    return self._client
                except Exception:
                    self._stop_locked()
            self._start_locked(python, source, signature)
            assert self._client is not None
            return self._client

    def status(self) -> dict:
        with self._lock:
            with self._log_lock:
                recent_logs = list(self._log_lines[-30:])
            result = {
                "managed_process": self._process is not None,
                "pid": self._process.pid if self._process else None,
                "recent_logs": recent_logs,
            }
            if self._client:
                try:
                    result["health"] = self._client.health()
                except Exception as exc:
                    result["health_error"] = str(exc)
            return result

    def unload(self, handle: RuntimeHandle) -> dict:
        return self.client_for(handle).unload()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _resolve_python(self, configured: str) -> Path:
        candidates = [
            configured,
            os.environ.get("FIREREDAUDIO_RUNTIME_PYTHON", ""),
            str(Path(__file__).resolve().parents[1] / ".runtime" / ".venv" / "Scripts" / "python.exe"),
            str(Path(os.environ.get("LOCALAPPDATA", "")) / "T8star-Aix" / "FireRedAudio" / "runtime" / ".venv" / "Scripts" / "python.exe"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return Path(candidate).resolve()
        raise RuntimeError(
            "未找到隔离 Python 3.10。请运行 comfyui-fireredaudio-T8/scripts/setup_runtime.py，"
            "或在模型加载器中填写 runtime_python。节点不会使用/修改 ComfyUI 自身 Python。"
        )

    def _resolve_source(self) -> Path:
        override = os.environ.get("FIREREDAUDIO_WORKER_SOURCE", "")
        roots = [
            Path(override) if override else None,
            Path(__file__).resolve().parents[1] / "worker_bundle",
            Path(__file__).resolve().parents[2],
        ]
        for root in roots:
            if root and (root / "fireredaudio_t8" / "worker.py").is_file() and (root / "inference.py").is_file():
                return root.resolve()
        raise RuntimeError("节点缺少 worker_bundle；请重新安装完整发行包")

    def _start_locked(self, python: Path, source: Path, signature: tuple[str, ...]) -> None:
        port = _free_port()
        token = secrets.token_hex(32)
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1",
                "FIREREDAUDIO_WORKER_TOKEN": token,
                "PYTHONPATH": os.pathsep.join(
                    [str(source), env.get("PYTHONPATH", "")]
                ).rstrip(os.pathsep),
            }
        )
        bundled_ffmpeg = Path(__file__).resolve().parents[1] / "tools" / "ffmpeg.exe"
        if bundled_ffmpeg.is_file():
            env["FIREREDAUDIO_FFMPEG"] = str(bundled_ffmpeg)
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self._process = subprocess.Popen(
            [str(python), "-m", "fireredaudio_t8.worker", "--host", "127.0.0.1", "--port", str(port)],
            cwd=str(source),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
        )
        self._client = WorkerClient(f"http://127.0.0.1:{port}", token)
        self._signature = signature
        self._log_thread = threading.Thread(target=self._capture_logs, daemon=True)
        self._log_thread.start()
        last_error: Exception | None = None
        for _ in range(240):
            if self._process.poll() is not None:
                # stdout reaches EOF when the process exits. Briefly join the reader
                # so the exception reliably includes the final startup diagnostics.
                if self._log_thread is not None:
                    self._log_thread.join(timeout=0.5)
                with self._log_lock:
                    recent = "".join(self._log_lines[-12:])
                raise RuntimeError("隔离 Worker 提前退出：" + recent)
            try:
                self._client.health()
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.25)
        self._stop_locked()
        raise RuntimeError(f"隔离 Worker 启动超时：{last_error}")

    def _capture_logs(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            # Log capture must not wait for the lifecycle lock: _start_locked holds
            # it while polling health, and a noisy failing worker could otherwise
            # fill the stdout pipe before its diagnostics become readable.
            with self._log_lock:
                self._log_lines.append(line)
                if len(self._log_lines) > 500:
                    del self._log_lines[:100]

    def _stop_locked(self) -> None:
        if self._client:
            try:
                self._client.request("shutdown", {}, timeout=3.0)
            except Exception:
                pass
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        self._client = None
        self._signature = None
        self._log_thread = None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


WORKER_MANAGER = WorkerManager()
atexit.register(WORKER_MANAGER.stop)
