from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROLE_NAMES = ("旁白", "主持人", "嘉宾", "记者", "工程师", "导演", "店主", "旅人")
LINE_TEMPLATES = (
    "第{index}句用于验证多角色批量配音。",
    "我们正在检查第{index}条真实模型输出。",
    "请保持语气自然，并清楚读出第{index}句。",
    "这段对白属于第{role}，编号是{index}。",
)


def load_modules(comfy_root: Path):
    sys.path.insert(0, str(comfy_root.resolve()))
    package = "fireredaudio_v015_real_scale"
    spec = importlib.util.spec_from_file_location(
        package,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载节点包")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    return (
        sys.modules[f"{package}.nodes_v3"],
        sys.modules[f"{package}.runtime.production"],
        sys.modules[f"{package}.runtime.types"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="100-line / 8-role real-model BatchDubbing acceptance"
    )
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--lines", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.lines < 100:
        raise SystemExit("发布规模验收要求 --lines 至少为 100")
    if not 1 <= args.batch_size <= 8:
        raise SystemExit("--batch-size 必须在 1–8")

    nodes, production, types = load_modules(Path(args.comfy_root))
    import folder_paths

    model_root = Path(args.model_root).resolve()
    runtime_python = Path(args.runtime_python).resolve()
    reference_audio = Path(args.reference_audio).resolve()
    output_root = Path(args.output_root).resolve()
    if not model_root.is_dir() or not runtime_python.is_file() or not reference_audio.is_file():
        raise FileNotFoundError("真模型、隔离 Python 或参考音频不存在")
    output_root.mkdir(parents=True, exist_ok=True)
    original_output = folder_paths.get_output_directory
    folder_paths.get_output_directory = lambda: str(output_root)
    try:
        profiles = [
            production.create_voice_profile(
                role,
                reference_audio,
                "同时，他强调微调要科学有序。",
                "zh",
            )
            for role in ROLE_NAMES
        ]
        bank = production.create_voice_bank(profiles)
        lines = []
        for offset in range(args.lines):
            index = offset + 1
            role = ROLE_NAMES[offset % len(ROLE_NAMES)]
            text = LINE_TEMPLATES[offset % len(LINE_TEMPLATES)].format(
                index=index, role=role
            )
            lines.append(
                production.ScriptLine(
                    f"scale-line-{index:03d}", index, role, text, "zh"
                )
            )
        plan = production.ScriptPlan("role_script", tuple(lines), ())
        handle = types.RuntimeHandle(
            model_root=str(model_root),
            device="cuda:0",
            memory_mode="sequential",
            acceleration_mode="flash_attention",
            runtime_python=str(runtime_python),
            release_after=False,
        )
        settings = types.GenerationSettings(quality_preset="fast", seed=15000)
        started = time.perf_counter()
        first = nodes.T8FireRedAudioBatchDubbing.execute(
            handle,
            plan,
            bank,
            "v015-real-100-lines-8-roles",
            "projects",
            True,
            False,
            args.batch_size,
            settings,
        )
        batch, manifest_path, report_text = first.result
        elapsed = round(time.perf_counter() - started, 3)
        report = json.loads(report_text)
        successful = batch.successful_items()
        expected_batches = math.ceil(int(report.get("generated") or 0) / args.batch_size)
        if len(successful) != args.lines or report.get("failed") != 0:
            raise AssertionError(
                f"百行真模型输出不完整：{len(successful)}/{args.lines}，failed={report.get('failed')}"
            )
        if report.get("batch_count") != expected_batches:
            raise AssertionError(
                f"批次数错误：{report.get('batch_count')} != {expected_batches}"
            )
        role_counts = Counter(str(item.get("speaker")) for item in successful)
        if set(role_counts) != set(ROLE_NAMES):
            raise AssertionError(f"没有覆盖 8 个角色：{sorted(role_counts)}")

        durations = []
        hashes = {}
        for item in successful:
            path = Path(str(item["output_path"]))
            metrics = production.wav_metrics(path)
            if metrics["duration_seconds"] <= 0 or path.stat().st_size <= 44:
                raise AssertionError(f"空或损坏的真实模型音频：{path}")
            durations.append(float(metrics["duration_seconds"]))
            hashes[str(item["line_id"])] = production.file_digest(path)

        resume_started = time.perf_counter()
        resumed = nodes.T8FireRedAudioBatchDubbing.execute(
            handle,
            plan,
            bank,
            "v015-real-100-lines-8-roles",
            "projects",
            True,
            False,
            args.batch_size,
            settings,
        )
        resumed_batch, _resumed_manifest, resumed_text = resumed.result
        resume_elapsed = round(time.perf_counter() - resume_started, 3)
        resumed_report = json.loads(resumed_text)
        if resumed_report.get("cache_hits") != args.lines or resumed_report.get("generated") != 0:
            raise AssertionError("百行 Manifest 二次执行没有 100% 恢复")
        hashes_after = {
            str(item["line_id"]): production.file_digest(item["output_path"])
            for item in resumed_batch.successful_items()
        }
        if hashes_after != hashes:
            raise AssertionError("百行恢复改变了既有输出哈希")

        performance = list(report.get("performance") or [])
        totals = [float(item["total_seconds"]) for item in performance if item.get("total_seconds")]
        result = {
            "ok": True,
            "lines_requested": args.lines,
            "roles_requested": len(ROLE_NAMES),
            "role_counts": dict(sorted(role_counts.items())),
            "batch_size": args.batch_size,
            "batch_count": report.get("batch_count"),
            "generated_this_run": report.get("generated"),
            "preexisting_cache_hits": report.get("cache_hits"),
            "failed": report.get("failed"),
            "execution_model": report.get("execution_model"),
            "worker_batch_route": report.get("worker_batch_route"),
            "elapsed_seconds": elapsed,
            "resume_elapsed_seconds": resume_elapsed,
            "resume_cache_hits": resumed_report.get("cache_hits"),
            "audio_duration_seconds": {
                "total": round(sum(durations), 3),
                "median": round(statistics.median(durations), 3),
                "minimum": round(min(durations), 3),
                "maximum": round(max(durations), 3),
            },
            "batch_total_seconds": {
                "median": round(statistics.median(totals), 3) if totals else None,
                "maximum": round(max(totals), 3) if totals else None,
            },
            "manifest_path": str(manifest_path),
            "hashes": hashes,
            "source_files_overwritten": False,
        }
        report_path = output_root / "v015-real-scale-acceptance.json"
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "hashes"}
                | {"report_path": str(report_path)},
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        try:
            nodes.WORKER_MANAGER.stop()
        finally:
            folder_paths.get_output_directory = original_output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
