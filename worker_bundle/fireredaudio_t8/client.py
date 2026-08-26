from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .errors import WorkerProtocolError


@dataclass(frozen=True)
class WorkerClient:
    base_url: str
    token: str
    timeout: float = 30.0

    def request(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            method="GET" if payload is None else "POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("error")
            except Exception:
                detail = str(exc)
            raise WorkerProtocolError(str(detail)) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise WorkerProtocolError(f"无法连接 FireRedAudio Worker：{exc}") from exc
        if not body.get("ok", False):
            raise WorkerProtocolError(str(body.get("error") or "Worker 请求失败"))
        return body.get("result", {})

    def health(self) -> dict[str, Any]:
        return self.request("health")

    def infer(self, payload: dict[str, Any], timeout: float = 3600.0) -> dict[str, Any]:
        return self.request("v1/infer", payload, timeout=timeout)

    def load(self, payload: dict[str, Any], timeout: float = 1800.0) -> dict[str, Any]:
        return self.request("v1/model/load", payload, timeout=timeout)

    def unload(self) -> dict[str, Any]:
        return self.request("v1/model/unload", {})

    def cancel(self, task_id: str | None = None) -> dict[str, Any]:
        return self.request("v1/task/cancel", {"task_id": task_id})

    def system_info(self) -> dict[str, Any]:
        return self.request("v1/system/info", {})

    def analyze_audio(self, audio_path: str) -> dict[str, Any]:
        return self.request("v1/audio/analyze", {"audio_path": audio_path}, timeout=120.0)

    def cache_status(self) -> dict[str, Any]:
        return self.request("v1/cache/status", {})

    def cleanup_cache(self, clear_all: bool = False) -> dict[str, Any]:
        return self.request(
            "v1/cache/cleanup",
            {"clear_all": clear_all, "max_age_hours": 72.0, "max_size_mib": 2048.0},
            timeout=120.0,
        )
