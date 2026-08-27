from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "model_firered_audio.json"


def load_pins(path: str | Path = DEFAULT_MANIFEST) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = ("codeRepository", "codeRevision", "modelRepository", "modelRevision")
    missing = [key for key in required if not str(payload.get(key) or "").strip()]
    if missing:
        raise ValueError("模型清单缺少字段：" + ", ".join(missing))
    return {key: str(payload[key]).strip() for key in required}


def compare_revisions(pins: dict[str, str], latest_code: str, latest_model: str) -> dict[str, Any]:
    code_changed = latest_code.strip() != pins["codeRevision"]
    model_changed = latest_model.strip() != pins["modelRevision"]
    return {
        "status": "update_available" if code_changed or model_changed else "current",
        "updates_available": code_changed or model_changed,
        "code": {
            "repository": pins["codeRepository"],
            "pinned": pins["codeRevision"],
            "latest": latest_code.strip(),
            "changed": code_changed,
            "compare_url": (
                f"https://github.com/{pins['codeRepository']}/compare/"
                f"{pins['codeRevision']}...{latest_code.strip()}"
            ),
        },
        "model": {
            "repository": pins["modelRepository"],
            "pinned": pins["modelRevision"],
            "latest": latest_model.strip(),
            "changed": model_changed,
            "commits_url": f"https://huggingface.co/{pins['modelRepository']}/commits/main",
        },
    }


def fetch_latest_code(repository: str, token: str = "") -> str:
    url = f"https://api.github.com/repos/{repository}/commits?per_page=1"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "comfyui-fireredaudio-T8-upstream-watch"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = _get_json(url, headers)
    if not isinstance(payload, list) or not payload or not payload[0].get("sha"):
        raise RuntimeError("GitHub commits API 未返回最新 commit")
    return str(payload[0]["sha"])


def fetch_latest_model(repository: str) -> str:
    encoded = urllib.parse.quote(repository, safe="/")
    payload = _get_json(
        f"https://huggingface.co/api/models/{encoded}",
        {"Accept": "application/json", "User-Agent": "comfyui-fireredaudio-T8-upstream-watch"},
    )
    if not isinstance(payload, dict) or not payload.get("sha"):
        raise RuntimeError("Hugging Face API 未返回最新模型 revision")
    return str(payload["sha"])


def _get_json(url: str, headers: dict[str, str]) -> Any:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"上游查询失败：{url}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Check pinned FireRedAudio code/model revisions")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default="")
    parser.add_argument("--github-sha", default="", help="offline/test override")
    parser.add_argument("--model-sha", default="", help="offline/test override")
    parser.add_argument("--fail-on-update", action="store_true")
    args = parser.parse_args()
    pins = load_pins(args.manifest)
    latest_code = args.github_sha or fetch_latest_code(
        pins["codeRepository"], os.environ.get("GITHUB_TOKEN", "")
    )
    latest_model = args.model_sha or fetch_latest_model(pins["modelRepository"])
    report = compare_revisions(pins, latest_code, latest_model)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    return 2 if args.fail_on_update and report["updates_available"] else 0


if __name__ == "__main__":
    sys.exit(main())
