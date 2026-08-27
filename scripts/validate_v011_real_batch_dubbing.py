from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_modules(comfy_root: Path):
    sys.path.insert(0, str(comfy_root.resolve()))
    package = "fireredaudio_v011_real_validation"
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
    parser = argparse.ArgumentParser(description="Real-model smoke for v0.11 BatchDubbing")
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
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
        profile = production.create_voice_profile(
            "旁白",
            reference_audio,
            "同时，他强调微调要科学有序。",
            "zh",
        )
        bank = production.create_voice_bank([profile])
        plan = production.ScriptPlan(
            "role_script",
            (
                production.ScriptLine("real-line-1", 1, "旁白", "欢迎使用批量创作闭环。", "zh"),
                production.ScriptLine("real-line-2", 2, "旁白", "这两句会先生成潜变量，再统一完成解码。", "zh"),
            ),
            (),
        )
        handle = types.RuntimeHandle(
            model_root=str(model_root),
            device="cuda:0",
            memory_mode="sequential",
            acceleration_mode="auto_safe",
            runtime_python=str(runtime_python),
            release_after=True,
        )
        settings = types.GenerationSettings(seed=4200)
        started = time.perf_counter()
        first = nodes.T8FireRedAudioBatchDubbing.execute(
            handle,
            plan,
            bank,
            "v011-real-batch",
            "projects",
            False,
            False,
            2,
            settings,
        )
        first_elapsed = time.perf_counter() - started
        batch, manifest_path, report_text = first.result
        report = json.loads(report_text)
        if len(batch.successful_items()) != 2 or report["failed"] != 0:
            raise AssertionError("真模型 BatchDubbing 没有产生 2/2 成功结果")
        if report["batch_count"] != 1 or report["worker_batch_route"] is not True:
            raise AssertionError("真模型 BatchDubbing 没有使用单次 tts-batch")
        if report["execution_model"] != "latent_first_decode_later":
            raise AssertionError(f"意外的执行模型：{report['execution_model']}")
        hashes_before = {
            str(item["line_id"]): production.file_digest(item["output_path"])
            for item in batch.successful_items()
        }
        metrics = {
            str(item["line_id"]): production.wav_metrics(item["output_path"])
            for item in batch.successful_items()
        }
        if any(value["duration_seconds"] <= 0 for value in metrics.values()):
            raise AssertionError("真模型批量输出包含空音频")

        resumed_started = time.perf_counter()
        resumed = nodes.T8FireRedAudioBatchDubbing.execute(
            handle,
            plan,
            bank,
            "v011-real-batch",
            "projects",
            True,
            False,
            2,
            settings,
        )
        resumed_elapsed = time.perf_counter() - resumed_started
        resumed_batch, _resumed_manifest, resumed_report_text = resumed.result
        resumed_report = json.loads(resumed_report_text)
        hashes_after = {
            str(item["line_id"]): production.file_digest(item["output_path"])
            for item in resumed_batch.successful_items()
        }
        if resumed_report["cache_hits"] != 2 or resumed_report["generated"] != 0:
            raise AssertionError("真模型批量恢复没有 2/2 命中 Manifest 缓存")
        if resumed_report["execution_model"] != "manifest_cache_only":
            raise AssertionError("缓存恢复报告没有标记 manifest_cache_only")
        if hashes_before != hashes_after:
            raise AssertionError("缓存恢复改变了既有输出文件")
        result = {
            "manifest_path": str(manifest_path),
            "execution_model": report["execution_model"],
            "batch_count": report["batch_count"],
            "batch_size": report["batch_size"],
            "generated": report["generated"],
            "failed": report["failed"],
            "cold_elapsed_seconds": round(first_elapsed, 3),
            "resume_elapsed_seconds": round(resumed_elapsed, 3),
            "cache_hits": resumed_report["cache_hits"],
            "hashes": hashes_before,
            "metrics": metrics,
            "source_files_overwritten": False,
        }
        report_path = output_root / "v011-real-batch-smoke.json"
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({**result, "report_path": str(report_path)}, ensure_ascii=False, indent=2))
    finally:
        try:
            nodes.WORKER_MANAGER.stop()
        finally:
            folder_paths.get_output_directory = original_output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
