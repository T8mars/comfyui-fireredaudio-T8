from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_tone(path: Path, *, seconds: float, frequency: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 24_000
    frames = bytearray()
    for index in range(round(seconds * sample_rate)):
        value = round(math.sin(2 * math.pi * frequency * index / sample_rate) * 7000)
        frames.extend(int(value).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)


def load_modules(comfy_root: Path):
    sys.path.insert(0, str(comfy_root.resolve()))
    package = "fireredaudio_v014_validation"
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
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute v0.14 text, duration, line-review and resume creator loop"
    )
    parser.add_argument("--comfy-root", required=True)
    args = parser.parse_args()
    nodes, production = load_modules(Path(args.comfy_root))
    import folder_paths

    with tempfile.TemporaryDirectory(prefix="fireredaudio-v014-") as raw:
        root = Path(raw)
        output = root / "output"
        output.mkdir()
        original_output = folder_paths.get_output_directory
        folder_paths.get_output_directory = lambda: str(output)
        try:
            safe = output / "source" / "safe.wav"
            unsafe = output / "source" / "unsafe.wav"
            write_tone(safe, seconds=1.10, frequency=330.0)
            write_tone(unsafe, seconds=1.50, frequency=440.0)
            source_hashes = {
                "safe": production.file_digest(safe),
                "unsafe": production.file_digest(unsafe),
            }
            plan = production.ScriptPlan(
                "srt",
                (
                    production.ScriptLine(
                        "safe",
                        1,
                        "旁白",
                        "ＡＰＩ 于 2026-08-28 发布。",
                        "zh",
                        0.0,
                        1.0,
                    ),
                    production.ScriptLine(
                        "unsafe",
                        2,
                        "旁白",
                        "这一句需要重新生成。",
                        "zh",
                        1.0,
                        2.0,
                    ),
                ),
                (),
            )
            normalized = nodes.T8FireRedAudioTextNormalizer.execute(
                plan,
                '{"API":"A P I"}',
                True,
                True,
                True,
                False,
            )
            normalized_plan, comparison_text, changed_ids, normalization_text = (
                normalized.result
            )
            comparison = json.loads(comparison_text)
            normalization = json.loads(normalization_text)
            if normalized_plan.lines[0].source_text != plan.lines[0].text:
                raise AssertionError("规范化没有保留原文")
            if (
                "A P I" not in normalized_plan.lines[0].text
                or "二零二六年" not in normalized_plan.lines[0].text
            ):
                raise AssertionError("词典或中文日期规范化没有生效")
            if changed_ids.splitlines() != ["safe"] or normalization["changed"] != 1:
                raise AssertionError("规范化变更清单错误")
            if (
                comparison["lines"][0]["source_text"]
                == comparison["lines"][0]["spoken_text"]
            ):
                raise AssertionError("原文/朗读文本对照没有区分")

            batch = production.AudioBatch(
                "source-manifest.json",
                (
                    {
                        **normalized_plan.lines[0].to_dict(),
                        "status": "complete",
                        "output_path": str(safe),
                    },
                    {
                        **normalized_plan.lines[1].to_dict(),
                        "status": "complete",
                        "output_path": str(unsafe),
                    },
                ),
            )
            fitted = nodes.T8FireRedAudioDurationFit.execute(
                batch,
                "safe_stretch",
                0.05,
                1.15,
                False,
                0.90,
                "validation-fit",
                "fireredaudio/duration-fit",
            )
            fitted_batch, fit_manifest, duration_retry_ids, fit_text = fitted.result
            fit_report = json.loads(fit_text)
            if (
                fit_report["adapted_line_ids"] != ["safe"]
                or duration_retry_ids != "unsafe"
            ):
                raise AssertionError("字幕时长适配没有区分安全拉伸和重做")
            if not Path(fit_manifest).is_file():
                raise AssertionError("字幕时长适配 Manifest 缺失")
            if (
                production.file_digest(safe) != source_hashes["safe"]
                or production.file_digest(unsafe) != source_hashes["unsafe"]
            ):
                raise AssertionError("字幕时长适配覆盖了源文件")

            qa = {
                "items": [
                    {
                        "line_id": "safe",
                        "passed": True,
                        "checks": {
                            "text": True,
                            "clipping": True,
                            "silence": True,
                            "cue_duration": True,
                        },
                    },
                    {
                        "line_id": "unsafe",
                        "passed": False,
                        "checks": {
                            "text": True,
                            "clipping": True,
                            "silence": True,
                            "cue_duration": False,
                        },
                    },
                ]
            }
            reviewed = nodes.T8FireRedAudioLineReview.execute(
                fitted_batch,
                "{}",
                "{}",
                "{}",
                "validation-review",
                "fireredaudio/line-reviews",
                40,
                qa,
            )
            (
                reviewed_batch,
                approved_batch,
                retry_ids,
                review_ids,
                review_manifest,
                review_text,
            ) = reviewed.result
            report = json.loads(review_text)
            if (
                report["approved_line_ids"] != ["safe"]
                or retry_ids != "unsafe"
                or review_ids
            ):
                raise AssertionError("QA 审核路由错误")
            if len(approved_batch.successful_items()) != 1:
                raise AssertionError("仅通过批次没有正确隔离待重做条目")
            ui_payload = (reviewed.ui or {}).get("fireredaudio_review", [])
            if not ui_payload or len(ui_payload[0].get("rows") or []) != 2:
                raise AssertionError("逐句审核台没有输出两行可视化数据")
            if len((reviewed.ui or {}).get("audio") or []) != 2:
                raise AssertionError("逐句审核台没有注册原生音频下载")

            overridden = nodes.T8FireRedAudioLineReview.execute(
                reviewed_batch,
                '{"unsafe":"approve"}',
                '{"unsafe":5}',
                '{"unsafe":"人工确认表演时长可接受"}',
                "validation-review-override",
                "fireredaudio/line-reviews",
                40,
                qa,
            )
            (
                overridden_batch,
                overridden_approved,
                _retry,
                _review,
                overridden_manifest,
                _report,
            ) = overridden.result
            if len(overridden_approved.successful_items()) != 2:
                raise AssertionError("人工决定没有覆盖自动建议")
            unsafe_review = overridden_batch.items[1]["human_review"]
            if unsafe_review["rating"] != 5.0 or not unsafe_review["manual_override"]:
                raise AssertionError("人工评分或覆盖状态没有落盘")

            resumed = nodes.T8FireRedAudioAudioBatchResume.execute(
                overridden_manifest,
                "error",
                False,
                False,
            )
            resumed_batch, resolved_manifest, resume_text = resumed.result
            resume_report = json.loads(resume_text)
            if Path(resolved_manifest) != Path(overridden_manifest):
                raise AssertionError("恢复节点改变了 Manifest 路径")
            if (
                len(resumed_batch.successful_items()) != 2
                or resume_report["reviewed"] != 2
            ):
                raise AssertionError("跨会话恢复没有保留音频和审核状态")
            if (
                resumed_batch.items[1]["human_review"]["note"]
                != "人工确认表演时长可接受"
            ):
                raise AssertionError("跨会话恢复丢失审核备注")

            print(
                json.dumps(
                    {
                        "text_normalization": {
                            "changed_line_ids": changed_ids.splitlines(),
                            "source_and_spoken_preserved": True,
                        },
                        "duration_fit": {
                            "adapted_line_ids": fit_report["adapted_line_ids"],
                            "retry_line_ids": fit_report["retry_line_ids"],
                            "source_files_overwritten": fit_report[
                                "source_files_overwritten"
                            ],
                        },
                        "line_review": {
                            "rows": len(ui_payload[0]["rows"]),
                            "native_audio_downloads": len(
                                (reviewed.ui or {}).get("audio") or []
                            ),
                            "qa_retry": retry_ids.splitlines(),
                            "manual_override_approved": len(
                                overridden_approved.successful_items()
                            ),
                        },
                        "resume": {
                            "playable": len(resumed_batch.successful_items()),
                            "reviewed": resume_report["reviewed"],
                            "manifest": str(overridden_manifest),
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        finally:
            folder_paths.get_output_directory = original_output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
