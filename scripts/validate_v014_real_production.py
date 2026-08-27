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
    package = "fireredaudio_v014_real_production"
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
        description="Real-model v0.14 creator review, repair and delivery acceptance"
    )
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
    if (
        not model_root.is_dir()
        or not runtime_python.is_file()
        or not reference_audio.is_file()
    ):
        raise FileNotFoundError("真模型、隔离 Python 或参考音频不存在")
    output_root.mkdir(parents=True, exist_ok=True)
    original_output = folder_paths.get_output_directory
    folder_paths.get_output_directory = lambda: str(output_root)
    started = time.perf_counter()
    try:
        profile = production.create_voice_profile(
            "旁白",
            reference_audio,
            "同时，他强调微调要科学有序。",
            "zh",
        )
        bank = production.create_voice_bank([profile])
        source_plan = production.ScriptPlan(
            "role_script",
            (
                production.ScriptLine(
                    "real-line-1",
                    1,
                    "旁白",
                    "欢迎使用ＡＰＩ，于2026-08-28进入制作审核。",
                    "zh",
                ),
                production.ScriptLine(
                    "real-line-2",
                    2,
                    "旁白",
                    "第二句将经过时长检查、语音质检和人工审核。",
                    "zh",
                ),
            ),
            (),
        )
        normalized_output = nodes.T8FireRedAudioTextNormalizer.execute(
            source_plan,
            '{"API":"A P I"}',
            True,
            True,
            True,
            False,
        )
        normalized_plan, comparison_text, changed_ids, normalization_text = (
            normalized_output.result
        )
        comparison = json.loads(comparison_text)
        normalization = json.loads(normalization_text)
        if changed_ids.splitlines() != ["real-line-1"]:
            raise AssertionError("真模型链文本规范化清单错误")
        if (
            comparison["lines"][0]["source_text"]
            == comparison["lines"][0]["spoken_text"]
        ):
            raise AssertionError("真模型链没有区分原文与朗读文本")

        handle = types.RuntimeHandle(
            model_root=str(model_root),
            device="cuda:0",
            memory_mode="sequential",
            acceleration_mode="auto_safe",
            runtime_python=str(runtime_python),
            release_after=False,
        )
        settings = types.GenerationSettings(seed=8140)
        generated_output = nodes.T8FireRedAudioBatchDubbing.execute(
            handle,
            normalized_plan,
            bank,
            "v014-real-production",
            "projects",
            True,
            False,
            2,
            settings,
        )
        generated_batch, generated_manifest, generated_text = generated_output.result
        generated_report = json.loads(generated_text)
        if len(generated_batch.successful_items()) != 2 or generated_report["failed"]:
            raise AssertionError("真模型链没有生成 2/2 台词")
        source_hashes = {
            str(item["line_id"]): production.file_digest(item["output_path"])
            for item in generated_batch.successful_items()
        }
        durations = {
            str(item["line_id"]): production.wav_metrics(item["output_path"])[
                "duration_seconds"
            ]
            for item in generated_batch.successful_items()
        }
        timed_items = []
        cursor = 0.0
        for item in generated_batch.items:
            copy = dict(item)
            line_id = str(copy["line_id"])
            ratio = 1.08 if line_id == "real-line-1" else 1.25
            available = durations[line_id] / ratio
            copy["start_seconds"] = cursor
            copy["end_seconds"] = cursor + available
            cursor += available
            timed_items.append(copy)
        timed_batch = production.AudioBatch(generated_manifest, tuple(timed_items))
        timed_by_id = {str(item["line_id"]): item for item in timed_items}
        timed_plan = production.ScriptPlan(
            normalized_plan.source_format,
            tuple(
                production.ScriptLine(
                    line.line_id,
                    line.index,
                    line.speaker,
                    line.text,
                    line.language,
                    timed_by_id[line.line_id]["start_seconds"],
                    timed_by_id[line.line_id]["end_seconds"],
                    source_text=line.source_text,
                    normalization=line.normalization,
                )
                for line in normalized_plan.lines
            ),
            normalized_plan.issues,
        )

        fitted_output = nodes.T8FireRedAudioDurationFit.execute(
            timed_batch,
            "safe_stretch",
            0.05,
            1.15,
            False,
            0.90,
            "v014-real-duration",
            "duration-fit",
        )
        fitted_batch, fit_manifest, duration_retry_ids, fit_text = fitted_output.result
        fit_report = json.loads(fit_text)
        if fit_report["adapted_line_ids"] != ["real-line-1"]:
            raise AssertionError("安全时长适配没有处理 real-line-1")
        if duration_retry_ids.splitlines() != ["real-line-2"]:
            raise AssertionError("超出安全倍率的台词没有进入重做清单")

        first_qa_output = nodes.T8FireRedAudioSpeechQA.execute(
            handle,
            fitted_batch,
            1.0,
            0.001,
            0.80,
            0.10,
            512,
        )
        first_qa, first_qa_text, first_qa_failed = first_qa_output.result
        first_qa_report = json.loads(first_qa_text)
        if "real-line-2" not in first_qa_failed.splitlines():
            raise AssertionError("QA 没有识别未适配的超时台词")

        first_review_output = nodes.T8FireRedAudioLineReview.execute(
            fitted_batch,
            "{}",
            "{}",
            "{}",
            "v014-real-first-review",
            "line-reviews",
            40,
            first_qa,
        )
        (
            first_reviewed,
            first_approved,
            first_retry_ids,
            first_review_ids,
            first_review_manifest,
            first_review_text,
        ) = first_review_output.result
        if first_retry_ids.splitlines() != ["real-line-2"]:
            raise AssertionError("逐句审核没有把 QA 失败项送入定向返修")
        if len(first_approved.successful_items()) != 1 or first_review_ids:
            raise AssertionError("逐句审核首轮路由不正确")

        repair_output = nodes.T8FireRedAudioBatchRetry.execute(
            handle,
            first_reviewed,
            timed_plan,
            bank,
            first_retry_ids,
            "v014-real-repair",
            "repairs",
            "increment",
            7,
            1,
            1,
            settings,
            True,
            0.10,
        )
        repaired_batch, repair_manifest, repair_text = repair_output.result
        repair_report = json.loads(repair_text)
        if repair_report["repaired_line_ids"]:
            if repair_report["repaired_line_ids"] != ["real-line-2"]:
                raise AssertionError("定向返修替换了错误台词")
            repaired_item = next(
                item
                for item in repaired_batch.items
                if item["line_id"] == "real-line-2"
            )
            if (
                "human_review" in repaired_item
                or "previous_human_review" not in repaired_item
            ):
                raise AssertionError("返修后没有清除旧审核决定或保留历史审核")
            final_source_batch = repaired_batch
            repair_outcome = "repaired_within_cue"
        else:
            if repair_report["failed_line_ids"] != ["real-line-2"]:
                raise AssertionError("不合格返修没有保持失败状态")
            if repair_report["cue_rejected_line_ids"] != ["real-line-2"]:
                raise AssertionError("仍然超时的返修音频没有被时间槽门禁拒绝")
            final_source_batch = first_reviewed
            repair_outcome = "blocked_by_cue_gate"

        final_qa_output = nodes.T8FireRedAudioSpeechQA.execute(
            handle,
            final_source_batch,
            1.0,
            0.001,
            0.80,
            0.10,
            512,
        )
        final_qa, final_qa_text, final_qa_failed = final_qa_output.result
        final_review_output = nodes.T8FireRedAudioLineReview.execute(
            final_source_batch,
            '{"real-line-1":"approve","real-line-2":"approve"}',
            '{"real-line-1":5,"real-line-2":4}',
            '{"real-line-1":"真模型试听候选","real-line-2":"返修后人工确认"}',
            "v014-real-final-review",
            "line-reviews",
            40,
            final_qa,
        )
        (
            final_reviewed,
            final_approved,
            final_retry_ids,
            final_review_ids,
            final_review_manifest,
            final_review_text,
        ) = final_review_output.result
        final_review_report = json.loads(final_review_text)
        if len(final_approved.successful_items()) != 2:
            raise AssertionError("人工终审没有产生 2/2 可交付音频")
        if final_retry_ids or final_review_ids:
            raise AssertionError("人工终审后仍残留重做或复核条目")
        if len((final_review_output.ui or {}).get("audio") or []) != 2:
            raise AssertionError("真模型终审没有注册 2 个原生试听/下载")

        export_output = nodes.T8FireRedAudioSaveAudioBatch.execute(
            final_approved,
            "wav",
            "v014-real-approved",
            "exports",
            True,
            False,
            40,
        )
        _, export_manifest, zip_path, export_text = export_output.result
        export_report = json.loads(export_text)
        if export_report["saved"] != 2 or not Path(zip_path).is_file():
            raise AssertionError("真模型终审交付包不完整")

        resumed_output = nodes.T8FireRedAudioAudioBatchResume.execute(
            final_review_manifest,
            "error",
            False,
            False,
        )
        resumed_batch, resolved_manifest, resume_text = resumed_output.result
        resume_report = json.loads(resume_text)
        if len(resumed_batch.successful_items()) != 2 or resume_report["reviewed"] != 2:
            raise AssertionError("真模型终审会话无法跨会话恢复")
        for item in generated_batch.successful_items():
            if (
                production.file_digest(item["output_path"])
                != source_hashes[item["line_id"]]
            ):
                raise AssertionError("时长适配或返修覆盖了原始生成文件")

        result = {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "normalization": normalization,
            "generated": generated_report,
            "durations": durations,
            "duration_fit": fit_report,
            "first_qa": first_qa_report,
            "first_review": json.loads(first_review_text),
            "repair": repair_report,
            "repair_outcome": repair_outcome,
            "final_qa": json.loads(final_qa_text),
            "final_qa_failed_line_ids": final_qa_failed.splitlines(),
            "final_review": final_review_report,
            "artifacts": {
                "generated_manifest": str(generated_manifest),
                "duration_fit_manifest": str(fit_manifest),
                "first_review_manifest": str(first_review_manifest),
                "repair_manifest": str(repair_manifest),
                "final_review_manifest": str(final_review_manifest),
                "resolved_resume_manifest": str(resolved_manifest),
                "export_manifest": str(export_manifest),
                "zip_path": str(zip_path),
            },
            "native_audio_downloads": len(
                (final_review_output.ui or {}).get("audio") or []
            ),
            "source_files_overwritten": False,
        }
        report_path = output_root / "v014-real-production-acceptance.json"
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {**result, "report_path": str(report_path)},
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
