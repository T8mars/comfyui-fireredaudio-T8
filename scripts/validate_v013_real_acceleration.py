from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_modules(comfy_root: Path):
    sys.path.insert(0, str(comfy_root.resolve()))
    package = "fireredaudio_v013_real_acceleration"
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
        sys.modules[f"{package}.runtime.audio_adapter"],
        sys.modules[f"{package}.runtime.production"],
        sys.modules[f"{package}.runtime.types"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Real off/FlashAttention/DeepSpeed benchmark for v0.13")
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--memory-mode", default="sequential")
    args = parser.parse_args()
    nodes, audio_adapter, production, types = load_modules(Path(args.comfy_root))
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
        handle = types.RuntimeHandle(
            model_root=str(model_root),
            device="cuda:0",
            memory_mode=str(args.memory_mode),
            acceleration_mode="off",
            runtime_python=str(runtime_python),
            release_after=False,
        )
        settings = types.GenerationSettings(
            quality_preset="custom",
            seed=4200,
            max_new_audio_steps=300,
            min_new_audio_steps=6,
            max_new_text_tokens=128,
            n_timesteps=6,
            inference_cfg=1.8,
        )
        result = nodes.T8FireRedAudioAccelerationBenchmark.execute(
            handle,
            audio_adapter.wav_to_audio(reference_audio),
            "同时，他强调微调要科学有序。",
            "这是一句固定的加速测试。",
            "zh",
            "off,flash_attention,deepspeed",
            1,
            3,
            10.0,
            True,
            "v013-real-acceleration",
            "benchmark",
            settings,
        )
        batch, recommendation, report_text, manifest_path = result.result
        report = json.loads(report_text)
        by_mode = {item["requested_mode"]: item for item in report["mode_reports"]}
        missing = sorted({"off", "flash_attention", "deepspeed"} - set(by_mode))
        if missing:
            raise AssertionError(f"基准缺少模式：{missing}")
        failures = {
            mode: item.get("error")
            for mode, item in by_mode.items()
            if item.get("status") != "complete"
        }
        if failures:
            raise AssertionError(f"真模型加速模式失败：{failures}")
        if len(batch.items) != 9:
            raise AssertionError(f"正式测量音频应为 9 条，实际 {len(batch.items)}")
        if any(len(item.get("runs") or []) != 3 for item in by_mode.values()):
            raise AssertionError("每个模式必须有三次正式测量")
        for item in batch.items:
            metrics = production.wav_metrics(item["output_path"])
            if metrics["duration_seconds"] <= 0:
                raise AssertionError(f"基准输出为空：{item['output_path']}")
        if report["settings_modified"]:
            raise AssertionError("加速向导修改了模型设置")
        summary = {
            "manifest_path": manifest_path,
            "recommendation": recommendation,
            "recommended_mode": report["recommended_mode"],
            "formal_outputs": len(batch.items),
            "settings_modified": report["settings_modified"],
            "modes": {
                mode: {
                    "effective_mode": item["effective_mode"],
                    "fallback_detected": item["fallback_detected"],
                    "median_total_seconds": item["median_total_seconds"],
                    "median_rtf": item["median_rtf"],
                    "peak_vram_gib": round(item["peak_vram_bytes"] / 1024**3, 3),
                    "reproducible_hash": item["reproducible_hash"],
                    "improvement_percent": item.get("improvement_percent"),
                    "eligible_for_recommendation": item.get("eligible_for_recommendation", mode == "off"),
                }
                for mode, item in by_mode.items()
            },
        }
        summary_path = output_root / "v013-real-acceleration-summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({**summary, "summary_path": str(summary_path)}, ensure_ascii=False, indent=2))
    finally:
        try:
            nodes.WORKER_MANAGER.stop()
        finally:
            folder_paths.get_output_directory = original_output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
