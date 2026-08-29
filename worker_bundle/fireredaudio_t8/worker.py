from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .constants import DEFAULT_HOST, PROTOCOL_VERSION
from .errors import FireRedAudioT8Error, TaskCancelledError
from .model_manager import validate_model_dir
from .runtime import FireRedAudioRuntime
from .audio_quality import analyze_audio
from .audio_post import prepare_reference_audio
from .production_quality import analyze_production_audio, text_diff_metrics
from .system_info import runtime_readiness

logger = logging.getLogger("fireredaudio_t8.worker")


class WorkerServer(ThreadingHTTPServer):
    daemon_threads = True
    # The creator workspace refreshes several independent project panels in
    # parallel.  HTTPServer's default backlog is only 5, which can make a
    # healthy Worker refuse part of that burst on Windows immediately after a
    # reconnect.  Keep enough headroom for UI refreshes and queued retries.
    request_queue_size = 64

    def __init__(self, address: tuple[str, int], token: str, debug: bool = False):
        super().__init__(address, WorkerRequestHandler)
        self.token = token
        self.debug = debug
        self.runtime = FireRedAudioRuntime()


class WorkerRequestHandler(BaseHTTPRequestHandler):
    server: WorkerServer

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        if self.path.rstrip("/") == "/health":
            self._send_ok(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "status": self.server.runtime.status(),
                }
            )
            return
        self._send_error(HTTPStatus.NOT_FOUND, "未知接口")

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        try:
            payload = self._read_json()
            route = self.path.rstrip("/")
            if route == "/v1/infer":
                result = self.server.runtime.infer(payload)
            elif route == "/v1/infer/tts-batch":
                requests = payload.get("requests")
                if not isinstance(requests, list):
                    raise ValueError("批量 TTS requests 必须是数组")
                result = self.server.runtime.infer_tts_batch(requests)
            elif route == "/v1/model/load":
                result = self.server.runtime.load(
                    payload["model_root"],
                    device=str(payload.get("device") or "auto"),
                    profile=str(payload.get("profile") or "full"),
                    memory_mode=str(payload.get("memory_mode") or "auto"),
                    acceleration_mode=str(payload.get("acceleration_mode") or "auto_safe"),
                )
            elif route == "/v1/model/unload":
                result = self.server.runtime.unload()
            elif route == "/v1/model/validate":
                result = validate_model_dir(
                    payload["model_root"],
                    profile=str(payload.get("profile") or "full"),
                    verify_hashes=bool(payload.get("verify_hashes", False)),
                ).to_dict()
            elif route == "/v1/system/info":
                result = runtime_readiness()
            elif route == "/v1/audio/analyze":
                result = analyze_audio(payload["audio_path"])
            elif route == "/v1/audio/prepare-reference":
                result = prepare_reference_audio(
                    payload["audio_path"],
                    payload["output_path"],
                    trim_silence=bool(payload.get("trim_silence", True)),
                    normalize_loudness=bool(payload.get("normalize_loudness", False)),
                    target_lufs=float(payload.get("target_lufs", -23.0)),
                    highpass_hz=(
                        None
                        if payload.get("highpass_hz") in (None, "", 0, 0.0)
                        else float(payload["highpass_hz"])
                    ),
                )
            elif route == "/v1/audio/production-qa":
                result = analyze_production_audio(
                    payload["audio_path"],
                    target_lufs=float(payload.get("target_lufs", -16.0)),
                    tolerance_lu=float(payload.get("tolerance_lu", 2.0)),
                    true_peak_ceiling_dbfs=float(
                        payload.get("true_peak_ceiling_dbfs", -1.0)
                    ),
                )
                if "reference_text" in payload or "hypothesis_text" in payload:
                    result["transcript_comparison"] = text_diff_metrics(
                        str(payload.get("reference_text") or ""),
                        str(payload.get("hypothesis_text") or ""),
                        language=str(payload.get("language") or "zh"),
                    )
            elif route == "/v1/cache/status":
                result = self.server.runtime.cache_status()
            elif route == "/v1/cache/cleanup":
                result = self.server.runtime.cleanup_cache(
                    max_age_hours=float(payload.get("max_age_hours", 72.0)),
                    max_size_mib=float(payload.get("max_size_mib", 2048.0)),
                    clear_all=bool(payload.get("clear_all", False)),
                )
            elif route == "/v1/task/cancel":
                result = self.server.runtime.cancel(payload.get("task_id"))
            elif route.startswith("/v1/project/"):
                from .project_api import handle_project_request

                result = handle_project_request(route, payload, runtime=self.server.runtime)
            elif route == "/shutdown":
                result = {"shutting_down": True}
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self._send_error(HTTPStatus.NOT_FOUND, "未知接口")
                return
            self._send_ok(result)
        except TaskCancelledError as exc:
            self._send_error(HTTPStatus.CONFLICT, str(exc))
        except (KeyError, ValueError, FireRedAudioT8Error) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            logger.exception("Worker request failed")
            detail = traceback.format_exc() if self.server.debug else str(exc)
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, detail)

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.token}"
        if not secrets.compare_digest(supplied, expected):
            self._send_error(HTTPStatus.UNAUTHORIZED, "Worker token 无效")
            return False
        return True

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Content-Length 无效") from exc
        if length <= 0 or length > 8 * 1024 * 1024:
            raise ValueError("请求体为空或超过 8 MiB")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    def _send_ok(self, result: dict[str, Any]) -> None:
        self._send_json(HTTPStatus.OK, {"ok": True, "result": result})

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(status, {"ok": False, "error": message})

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FireRedAudio T8 isolated worker")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("出于安全原因，Worker 只能绑定回环地址")
    token = os.environ.get("FIREREDAUDIO_WORKER_TOKEN", "")
    if len(token) < 24:
        raise SystemExit("FIREREDAUDIO_WORKER_TOKEN 缺失或过短")
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server = WorkerServer((args.host, args.port), token=token, debug=args.debug)
    ready = {
        "host": server.server_address[0],
        "port": server.server_address[1],
        "protocol_version": PROTOCOL_VERSION,
        "pid": os.getpid(),
    }
    print("FIREREDAUDIO_WORKER_READY " + json.dumps(ready), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.runtime.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
