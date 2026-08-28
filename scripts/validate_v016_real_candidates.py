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
    package = "fireredaudio_v017_real_candidate_validation"
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
    parser = argparse.ArgumentParser(description="Real-model v0.17 creative candidate smoke")
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--source-audio", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    nodes, production, types = load_modules(Path(args.comfy_root))
    import folder_paths

    model_root = Path(args.model_root).resolve()
    runtime_python = Path(args.runtime_python).resolve()
    reference_audio = Path(args.reference_audio).resolve()
    source_audio = Path(args.source_audio).resolve()
    output_root = Path(args.output_root).resolve()
    if not model_root.is_dir() or not runtime_python.is_file() or not reference_audio.is_file() or not source_audio.is_file():
        raise FileNotFoundError("真模型、隔离 Python、参考音频或源 Take 不存在")
    output_root.mkdir(parents=True, exist_ok=True)
    original_output = folder_paths.get_output_directory
    folder_paths.get_output_directory = lambda: str(output_root)
    source_hash = production.file_digest(source_audio)
    started = time.perf_counter()
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
                production.ScriptLine(
                    "real-line-1",
                    1,
                    "旁白",
                    "欢迎使用A P I,于二零二六年八月二十八日进入制作审核。",
                    "zh",
                ),
            ),
            (),
        )
        source_batch = production.AudioBatch(
            "v015-real-source",
            (
                {
                    "line_id": "real-line-1",
                    "index": 1,
                    "speaker": "旁白",
                    "text": plan.lines[0].text,
                    "language": "zh",
                    "status": "complete",
                    "output_path": str(source_audio),
                    "seed": 8140,
                },
            ),
        )
        handle = types.RuntimeHandle(
            model_root=str(model_root),
            device="cuda:0",
            memory_mode="sequential",
            acceleration_mode="auto_safe",
            runtime_python=str(runtime_python),
            release_after=False,
        )
        settings = types.GenerationSettings(quality_preset="fast", seed=9301)
        output = nodes.T8FireRedAudioCreativeCandidatePool.execute(
            handle,
            source_batch,
            plan,
            bank,
            "real-line-1",
            2,
            9301,
            97,
            True,
            False,
            "v017-real-candidates",
            "fireredaudio/candidates",
            settings,
        )
        candidates, source_line_id, manifest_path, report_text = output.result
        report = json.loads(report_text)
        playable = candidates.successful_items()
        if source_line_id != "real-line-1" or len(playable) != 3:
            raise AssertionError("真模型候选池没有得到原 Take + 2 个新候选")
        if report["requested_seeds"] != [9301, 9398] or report["generated_count"] != 2:
            raise AssertionError("真模型候选 Seed 证据不正确")
        if report["distinct_audio_hashes"] < 2:
            raise AssertionError("真模型候选全部为相同哈希")
        if not report["diversity_prefilter_passed"] or not report["human_listening_required"]:
            raise AssertionError("真模型候选没有通过声学重复预筛或缺少人工盲听门禁")
        if production.file_digest(source_audio) != source_hash:
            raise AssertionError("真模型候选生成覆盖了源 Take")
        evidence = {
            "schema_version": 1,
            "node_version": nodes.NODE_VERSION,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "model_root": str(model_root),
            "reference_audio": str(reference_audio),
            "source_audio": str(source_audio),
            "source_sha256_before": source_hash,
            "source_sha256_after": production.file_digest(source_audio),
            "manifest_path": manifest_path,
            "requested_seeds": report["requested_seeds"],
            "generated_count": report["generated_count"],
            "playable_count": report["playable_count"],
            "distinct_audio_hashes": report["distinct_audio_hashes"],
            "duplicate_candidate_groups": report["duplicate_candidate_groups"],
            "minimum_acoustic_difference": report["minimum_acoustic_difference"],
            "pairwise_acoustic_evidence": report["pairwise_acoustic_evidence"],
            "acoustic_near_duplicate_pairs": report["acoustic_near_duplicate_pairs"],
            "diversity_prefilter_passed": report["diversity_prefilter_passed"],
            "human_listening_required": report["human_listening_required"],
            "blind_filenames": report["blind_filenames"],
            "performance": report.get("performance"),
            "items": [
                {
                    "line_id": item.get("line_id"),
                    "origin": item.get("candidate_origin"),
                    "seed": item.get("seed"),
                    "output_path": item.get("output_path"),
                    "output_sha256": item.get("output_sha256"),
                    "duration_seconds": (item.get("metrics") or {}).get("duration_seconds"),
                }
                for item in playable
            ],
        }
        report_path = output_root / "v017-real-candidate-acceptance.json"
        report_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
    finally:
        try:
            nodes.WORKER_MANAGER.unload(
                types.RuntimeHandle(
                    model_root=str(model_root),
                    device="cuda:0",
                    memory_mode="sequential",
                    acceleration_mode="auto_safe",
                    runtime_python=str(runtime_python),
                    release_after=False,
                )
            )
        except Exception:
            pass
        nodes.WORKER_MANAGER.stop()
        folder_paths.get_output_directory = original_output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
