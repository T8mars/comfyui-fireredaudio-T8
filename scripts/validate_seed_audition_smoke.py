from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = ROOT / "worker_bundle"
sys.path.insert(0, str(WORKER_ROOT))

from fireredaudio_t8.runtime import FireRedAudioRuntime


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-seed true-model latent batch smoke test")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--target-text", default="这是多种子试音的真实模型验收。")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=4200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-mode", default="sequential")
    parser.add_argument("--acceleration-mode", default="auto_safe")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = FireRedAudioRuntime()
    report_path = output / "seed-audition-smoke.json"
    report: dict = {"ok": False, "outputs": []}
    try:
        requests = []
        for offset in range(2):
            requests.append(
                {
                    "task": "tts",
                    "task_id": f"seed-audition-smoke-{offset + 1}",
                    "model_root": str(Path(args.model_root).resolve()),
                    "prompt_audio": str(Path(args.reference_audio).resolve()),
                    "prompt_text": args.prompt_text,
                    "target_text": args.target_text,
                    "language": "zh",
                    "seed": args.seed + offset,
                    "device": args.device,
                    "memory_mode": args.memory_mode,
                    "acceleration_mode": args.acceleration_mode,
                    "quality_preset": "fast",
                    "output_path": str(output / f"take-seed-{args.seed + offset}.wav"),
                    "release_after": False,
                }
            )
        result = runtime.infer_tts_batch(requests)
        for outcome in result.get("outcomes", []):
            if not outcome.get("ok"):
                raise RuntimeError(str(outcome.get("error") or "batch outcome failed"))
            path = Path(outcome["result"]["output_path"])
            with wave.open(str(path), "rb") as reader:
                duration = reader.getnframes() / reader.getframerate()
                evidence = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "duration_seconds": round(duration, 3),
                    "sample_rate": reader.getframerate(),
                    "channels": reader.getnchannels(),
                }
            if evidence["bytes"] <= 44 or duration <= 0:
                raise RuntimeError(f"无效试音输出：{path}")
            report["outputs"].append(evidence)
        report.update(
            ok=len(report["outputs"]) == 2,
            completed=result.get("completed"),
            failed=result.get("failed"),
            performance=result.get("performance"),
        )
        if not report["ok"]:
            raise RuntimeError("批量端点未返回两个有效候选")
        return 0
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        runtime.unload()
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
