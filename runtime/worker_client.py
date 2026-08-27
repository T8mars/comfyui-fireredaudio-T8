from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkerClient:
    base_url: str
    token: str

    def request(
        self,
        route: str,
        payload: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/{route.lstrip('/')}",
            data=data,
            method="GET" if data is None else "POST",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("error")
            except Exception:
                detail = str(exc)
            raise RuntimeError(str(detail)) from exc
        except Exception as exc:
            raise RuntimeError(f"无法连接隔离 FireRedAudio Worker：{exc}") from exc
        if not body.get("ok"):
            raise RuntimeError(str(body.get("error") or "Worker 请求失败"))
        return body.get("result", {})

    def health(self) -> dict[str, Any]:
        return self.request("health")

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("v1/infer", payload, timeout=3600.0)

    def infer_tts_batch(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        return self.request(
            "v1/infer/tts-batch", {"requests": requests}, timeout=7200.0
        )

    def validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        timeout = 3600.0 if payload.get("verify_hashes") else 60.0
        return self.request("v1/model/validate", payload, timeout=timeout)

    def unload(self) -> dict[str, Any]:
        return self.request("v1/model/unload", {}, timeout=120.0)

    def cancel(self, task_id: str | None = None) -> dict[str, Any]:
        return self.request("v1/task/cancel", {"task_id": task_id}, timeout=5.0)

    def system_info(self) -> dict[str, Any]:
        return self.request("v1/system/info", {}, timeout=30.0)

    def analyze_audio(self, audio_path: str) -> dict[str, Any]:
        return self.request(
            "v1/audio/analyze", {"audio_path": audio_path}, timeout=120.0
        )

    def prepare_reference(
        self,
        audio_path: str,
        output_path: str,
        *,
        trim_silence: bool = True,
        normalize_loudness: bool = False,
        target_lufs: float = -23.0,
        highpass_hz: float | None = 60.0,
    ) -> dict[str, Any]:
        return self.request(
            "v1/audio/prepare-reference",
            {
                "audio_path": audio_path,
                "output_path": output_path,
                "trim_silence": trim_silence,
                "normalize_loudness": normalize_loudness,
                "target_lufs": target_lufs,
                "highpass_hz": highpass_hz,
            },
            timeout=600.0,
        )

    def cache_status(self) -> dict[str, Any]:
        return self.request("v1/cache/status", {}, timeout=30.0)

    def cleanup_cache(self, clear_all: bool = False) -> dict[str, Any]:
        return self.request(
            "v1/cache/cleanup",
            {"clear_all": clear_all, "max_age_hours": 72.0, "max_size_mib": 2048.0},
            timeout=120.0,
        )
