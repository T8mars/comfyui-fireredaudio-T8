from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NODE_ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = NODE_ROOT / "worker_bundle"
if str(WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKER_ROOT))

from fireredaudio_t8.runtime import FireRedAudioRuntime

TEXTS = (
    "这是第一轮真实模型稳定性验证。",
    "连续生成可以暴露显存没有释放的问题。",
    "这一轮检查参考音频缓存是否能够复用。",
    "声音应当清晰自然并且不是空文件。",
    "现在记录生成耗时和输出音频长度。",
    "稳定的工作流不应随着轮次持续占用更多显存。",
    "这条语音用于检查长时间运行后的模型状态。",
    "每一轮都使用不同种子生成独立的音频。",
    "节点运行时保持与宿主依赖完全隔离。",
    "自动安全加速模式必须保留明确的回退信息。",
    "第十一轮开始进入后半程稳定性检查。",
    "所有结果都会写入可复核的本地报告。",
    "显存漂移将比较预热后的前五轮与最后五轮。",
    "如果任何一轮失败，验收报告会明确标记失败。",
    "输出文件会校验大小、时长和哈希。",
    "这一轮继续验证连续语音合成能力。",
    "测试不会下载模型，也不会把模型打进节点包。",
    "真实模型输出不能由模拟数据替代。",
    "倒数第二轮检查运行时仍然可以正常解码。",
    "第二十轮完成，连续生成稳定性验证结束。",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ComfyUI FireRedAudio true-model TTS long-run and VRAM stability audit"
    )
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--prompt-text", default="同时，他强调微调要科学有序。")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-mode", default="sequential")
    parser.add_argument("--acceleration-mode", default="auto_safe")
    parser.add_argument("--quality-preset", default="fast")
    parser.add_argument("--seed", type=int, default=88000)
    parser.add_argument(
        "--output-dir",
        default=str(NODE_ROOT / "validation" / "real-model-long-run"),
    )
    parser.add_argument("--cancel-file", default="")
    parser.add_argument(
        "--max-median-drift-mib",
        type=float,
        default=512.0,
        help="allowed nvidia-smi used-memory median drift from warm first five to last five",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds < 20:
        raise SystemExit("--rounds must be at least 20 for the release stability gate")
    model_root = Path(args.model_root).resolve()
    reference = Path(args.reference_audio).resolve()
    if not model_root.is_dir():
        raise SystemExit(f"model root does not exist: {model_root}")
    if not reference.is_file():
        raise SystemExit(f"reference audio does not exist: {reference}")
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "real-model-long-run-report.json"
    runtime = FireRedAudioRuntime()
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "node_repository": "comfyui-fireredaudio-T8",
        "worker_root": str(WORKER_ROOT),
        "model_root": str(model_root),
        "reference_audio": str(reference),
        "reference_sha256": _sha256(reference),
        "rounds_requested": args.rounds,
        "device": args.device,
        "memory_mode": args.memory_mode,
        "acceleration_mode": args.acceleration_mode,
        "quality_preset": args.quality_preset,
        "entries": [],
        "ok": False,
    }
    exit_code = 1
    try:
        for index in range(1, args.rounds + 1):
            if args.cancel_file and Path(args.cancel_file).is_file():
                raise RuntimeError("validation cancelled by sentinel file")
            target = output_root / f"round-{index:02d}.wav"
            before = _gpu_memory(args.device)
            started = time.perf_counter()
            result = runtime.infer(
                {
                    "task": "tts",
                    "task_id": f"comfy-long-run-{index:02d}",
                    "model_root": str(model_root),
                    "device": args.device,
                    "memory_mode": args.memory_mode,
                    "acceleration_mode": args.acceleration_mode,
                    "quality_preset": args.quality_preset,
                    "prompt_audio": str(reference),
                    "prompt_text": args.prompt_text,
                    "target_text": TEXTS[(index - 1) % len(TEXTS)],
                    "language": "zh",
                    "seed": args.seed + index,
                    "output_path": str(target),
                    "release_after": False,
                }
            )
            wall_seconds = time.perf_counter() - started
            after = _gpu_memory(args.device)
            wav = _wav_evidence(target)
            entry = {
                "round": index,
                "task_id": f"comfy-long-run-{index:02d}",
                "seed": args.seed + index,
                "text": TEXTS[(index - 1) % len(TEXTS)],
                "wall_seconds": round(wall_seconds, 3),
                "runtime_elapsed_seconds": result.get("elapsed_seconds"),
                "audio_duration_seconds": result.get(
                    "audio_duration_seconds", result.get("duration_seconds")
                ),
                "performance": result.get("performance") or {},
                "worker_acceleration": result.get("acceleration"),
                "gpu_before": before,
                "gpu_after": after,
                **wav,
            }
            report["entries"].append(entry)
            _write_json_atomic(report_path, report)
            print("LONG_RUN_ROUND " + json.dumps(entry, ensure_ascii=False), flush=True)

        entries = list(report["entries"])
        settled = [
            float(entry["gpu_after"]["used_mib"])
            for entry in entries[1:]
            if entry.get("gpu_after") and entry["gpu_after"].get("used_mib") is not None
        ]
        first_window = settled[:5]
        last_window = settled[-5:]
        median_drift = (
            statistics.median(last_window) - statistics.median(first_window)
            if first_window and last_window
            else None
        )
        performance = [entry.get("performance") or {} for entry in entries]
        report["summary"] = {
            "rounds_completed": len(entries),
            "all_outputs_valid": all(
                entry.get("output_bytes", 0) > 44 and entry.get("wav_duration_seconds", 0) > 0
                for entry in entries
            ),
            "median_wall_seconds": round(
                statistics.median(float(entry["wall_seconds"]) for entry in entries), 3
            ),
            "max_wall_seconds": max(float(entry["wall_seconds"]) for entry in entries),
            "max_gpu_peak_allocated_bytes": max(
                (int(value.get("gpu_peak_allocated_bytes") or 0) for value in performance),
                default=0,
            ),
            "max_gpu_peak_reserved_bytes": max(
                (int(value.get("gpu_peak_reserved_bytes") or 0) for value in performance),
                default=0,
            ),
            "settled_used_mib_min": min(settled) if settled else None,
            "settled_used_mib_max": max(settled) if settled else None,
            "first_five_used_mib_median": statistics.median(first_window) if first_window else None,
            "last_five_used_mib_median": statistics.median(last_window) if last_window else None,
            "used_mib_median_drift": median_drift,
            "max_allowed_median_drift_mib": args.max_median_drift_mib,
            "memory_stable": median_drift is not None
            and median_drift <= args.max_median_drift_mib,
        }
        report["runtime_status_before_unload"] = runtime.status()
        report["ok"] = bool(
            len(entries) == args.rounds
            and report["summary"]["all_outputs_valid"]
            and report["summary"]["memory_stable"]
        )
        exit_code = 0 if report["ok"] else 1
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            runtime.unload()
        finally:
            report["finished_at"] = datetime.now(timezone.utc).isoformat()
            report["gpu_after_unload"] = _gpu_memory(args.device)
            _write_json_atomic(report_path, report)
            print(f"Long-run report: {report_path}", flush=True)
    return exit_code


def _gpu_memory(device: str) -> dict[str, Any] | None:
    if not str(device).startswith("cuda"):
        return None
    index = int(str(device).split(":", 1)[1]) if ":" in str(device) else 0
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=name,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        return {"error": completed.stderr.strip()}
    parts = [value.strip() for value in completed.stdout.strip().split(",")]
    if len(parts) < 3:
        return {"error": f"unexpected nvidia-smi output: {completed.stdout.strip()}"}
    return {"name": parts[0], "used_mib": int(parts[1]), "free_mib": int(parts[2])}


def _wav_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 44:
        raise RuntimeError(f"invalid TTS output: {path}")
    with wave.open(str(path), "rb") as reader:
        frames = reader.getnframes()
        sample_rate = reader.getframerate()
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
    if frames <= 0 or sample_rate <= 0:
        raise RuntimeError(f"empty TTS output: {path}")
    return {
        "output_path": str(path),
        "output_bytes": path.stat().st_size,
        "output_sha256": _sha256(path),
        "wav_duration_seconds": frames / sample_rate,
        "wav_sample_rate": sample_rate,
        "wav_channels": channels,
        "wav_sample_width": sample_width,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
