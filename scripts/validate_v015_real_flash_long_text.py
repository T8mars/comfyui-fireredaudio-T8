from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = ROOT / "worker_bundle"
if str(WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKER_ROOT))

from fireredaudio_t8.runtime import FireRedAudioRuntime


LONG_TEXT = (
    "在漫长的制作夜里，导演先确认每个角色的情绪、节奏和发音，再让工程师逐句检查停顿、响度与时间槽。"
    "如果某一句不够自然，系统只返修这一句，并保留已经通过的版本、生成参数和审核记录。"
    "最终交付前，创作者仍会完整试听前后对比，确认声音清晰、语义准确、衔接顺畅，然后再导出对白分轨、字幕和成品文件。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long-text 10-run real FlashAttention median acceptance"
    )
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--prompt-text", default="同时，他强调微调要科学有序。")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 10:
        raise SystemExit("发布性能验收要求 --runs 至少为 10")
    if args.warmup_runs < 1:
        raise SystemExit("至少需要 1 次暖机")
    model_root = Path(args.model_root).resolve()
    reference = Path(args.reference_audio).resolve()
    output_root = Path(args.output_dir).resolve()
    if not model_root.is_dir() or not reference.is_file():
        raise FileNotFoundError("真模型或参考音频不存在")
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "v015-real-flash-long-text-report.json"
    runtime = FireRedAudioRuntime()
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_root": str(model_root),
        "reference_audio": str(reference),
        "reference_sha256": _sha256(reference),
        "prompt_text": args.prompt_text,
        "target_text": LONG_TEXT,
        "target_characters": len(LONG_TEXT),
        "requested_mode": "flash_attention",
        "memory_mode": "sequential",
        "quality_preset": "balanced",
        "warmup_runs": args.warmup_runs,
        "measure_runs": args.runs,
        "runs": [],
        "ok": False,
    }
    try:
        total = args.warmup_runs + args.runs
        for iteration in range(total):
            warmup = iteration < args.warmup_runs
            formal_index = iteration - args.warmup_runs + 1
            label = f"warmup-{iteration + 1}" if warmup else f"run-{formal_index:02d}"
            target = output_root / f"flash-long-{label}.wav"
            started = time.perf_counter()
            result = runtime.infer(
                {
                    "task": "tts",
                    "task_id": f"v015-flash-long-{label}",
                    "model_root": str(model_root),
                    "device": "cuda:0",
                    "memory_mode": "sequential",
                    "acceleration_mode": "flash_attention",
                    "quality_preset": "balanced",
                    "prompt_audio": str(reference),
                    "prompt_text": args.prompt_text,
                    "target_text": LONG_TEXT,
                    "language": "zh",
                    "seed": 20260828,
                    "output_path": str(target),
                    "release_after": False,
                }
            )
            wall_seconds = round(time.perf_counter() - started, 3)
            evidence = _wav_evidence(target)
            status = runtime.status()
            selection = ((status.get("acceleration") or {}).get("selection") or {})
            entry = {
                "run": 0 if warmup else formal_index,
                "warmup": warmup,
                "wall_seconds": wall_seconds,
                "performance": result.get("performance") or {},
                "acceleration_selection": selection,
                **evidence,
            }
            if warmup:
                report["warmup"] = entry
                target.unlink(missing_ok=True)
                Path(f"{target}.json").unlink(missing_ok=True)
            else:
                report["runs"].append(entry)
            _write_json_atomic(report_path, report)
            print(
                "FLASH_LONG_RUN "
                + json.dumps(
                    {
                        "run": entry["run"],
                        "warmup": warmup,
                        "wall_seconds": wall_seconds,
                        "audio_duration_seconds": entry["wav_duration_seconds"],
                        "effective_mode": selection.get("effective"),
                        "fallback": selection.get("effective") != "flash_attention",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        runs = list(report["runs"])
        totals = [
            float((item.get("performance") or {}).get("total_seconds") or item["wall_seconds"])
            for item in runs
        ]
        rtfs = [
            float((item.get("performance") or {}).get("rtf") or 0.0)
            for item in runs
        ]
        hashes = [str(item["output_sha256"]) for item in runs]
        selections = [item.get("acceleration_selection") or {} for item in runs]
        report["summary"] = {
            "runs_completed": len(runs),
            "all_outputs_valid": all(item["output_bytes"] > 44 for item in runs),
            "all_flash_attention_effective": all(
                item.get("effective") == "flash_attention" and item.get("available", True)
                for item in selections
            ),
            "fallback_count": sum(
                item.get("effective") != "flash_attention" for item in selections
            ),
            "median_total_seconds": round(statistics.median(totals), 3),
            "minimum_total_seconds": round(min(totals), 3),
            "maximum_total_seconds": round(max(totals), 3),
            "median_rtf": round(statistics.median(rtfs), 3),
            "median_audio_duration_seconds": round(
                statistics.median(float(item["wav_duration_seconds"]) for item in runs), 3
            ),
            "peak_vram_bytes": max(
                int((item.get("performance") or {}).get("gpu_peak_allocated_bytes") or 0)
                for item in runs
            ),
            "reproducible_hash": len(set(hashes)) == 1,
            "unique_output_hashes": len(set(hashes)),
        }
        report["ok"] = bool(
            len(runs) == args.runs
            and report["summary"]["all_outputs_valid"]
            and report["summary"]["all_flash_attention_effective"]
            and report["summary"]["reproducible_hash"]
        )
        if not report["ok"]:
            raise AssertionError(f"长文本 FlashAttention 验收失败：{report['summary']}")
        return 0
    finally:
        try:
            runtime.unload()
        finally:
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            _write_json_atomic(report_path, report)
            print(
                json.dumps(
                    {
                        "ok": report.get("ok"),
                        "target_characters": report.get("target_characters"),
                        "summary": report.get("summary"),
                        "report_path": str(report_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )


def _wav_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 44:
        raise RuntimeError(f"无效的长文本 TTS 输出：{path}")
    with wave.open(str(path), "rb") as reader:
        frames = reader.getnframes()
        sample_rate = reader.getframerate()
        channels = reader.getnchannels()
    if frames <= 0 or sample_rate <= 0:
        raise RuntimeError(f"空的长文本 TTS 输出：{path}")
    return {
        "output_path": str(path),
        "output_bytes": path.stat().st_size,
        "output_sha256": _sha256(path),
        "wav_duration_seconds": frames / sample_rate,
        "wav_sample_rate": sample_rate,
        "wav_channels": channels,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
