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


def write_tone(path: Path, *, frequency: float = 440.0, seconds: float = 0.08) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 24_000
    frames = bytearray()
    for index in range(round(sample_rate * seconds)):
        value = round(math.sin(2 * math.pi * frequency * index / sample_rate) * 8000)
        frames.extend(int(value).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)


def load_modules(comfy_root: Path):
    sys.path.insert(0, str(comfy_root.resolve()))
    package = "fireredaudio_v011_validation"
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
    parser = argparse.ArgumentParser(description="Execute the v0.11 creator loop without a real model")
    parser.add_argument("--comfy-root", required=True)
    args = parser.parse_args()
    nodes, production, types = load_modules(Path(args.comfy_root))
    import folder_paths

    with tempfile.TemporaryDirectory(prefix="fireredaudio-v011-") as raw:
        root = Path(raw)
        output = root / "output"
        output.mkdir()
        original_output = folder_paths.get_output_directory
        folder_paths.get_output_directory = lambda: str(output)
        try:
            prompt_a = root / "prompt-a.wav"
            prompt_b = root / "prompt-b.wav"
            write_tone(prompt_a, frequency=330.0)
            write_tone(prompt_b, frequency=550.0)
            profile_a = production.create_voice_profile("旁白", prompt_a, "旁白参考", "zh")
            profile_b = production.create_voice_profile("角色甲", prompt_b, "角色参考", "zh")
            bank = production.create_voice_bank([profile_a, profile_b])
            plan = production.ScriptPlan(
                "role_script",
                (
                    production.ScriptLine("line-1", 1, "旁白", "第一句", "zh"),
                    production.ScriptLine("line-2", 2, "角色甲", "第二句", "zh"),
                    production.ScriptLine("line-3", 3, "旁白", "第三句", "zh"),
                ),
                (),
            )
            handle = types.RuntimeHandle(model_root=str(root / "model"), release_after=False)
            settings = types.GenerationSettings(seed=42)
            batch_calls: list[list[dict]] = []

            def fake_batch(_handle, requests):
                copied = [dict(request) for request in requests]
                batch_calls.append(copied)
                outcomes = []
                for index, request in enumerate(requests):
                    path = Path(request["output_path"])
                    write_tone(path, frequency=440.0 + index * 30)
                    outcomes.append(
                        {
                            "ok": True,
                            "index": index,
                            "result": {
                                "output_path": str(path),
                                "performance": {"execution_model": "latent_first_decode_later"},
                            },
                        }
                    )
                return {
                    "outcomes": outcomes,
                    "performance": {
                        "execution_model": "latent_first_decode_later",
                        "batch_size": len(requests),
                    },
                }

            nodes._infer_tts_batch = fake_batch
            first = nodes.T8FireRedAudioBatchDubbing.execute(
                handle,
                plan,
                bank,
                "creator-loop",
                "fireredaudio/projects",
                True,
                True,
                2,
                settings,
            )
            batch, manifest_path, report_text = first.result
            report = json.loads(report_text)
            if [len(call) for call in batch_calls] != [2, 1]:
                raise AssertionError(f"未按 2 条分块执行：{[len(call) for call in batch_calls]}")
            if {request["prompt_text"] for request in batch_calls[0]} != {"旁白参考", "角色参考"}:
                raise AssertionError("异构角色参考没有进入同一 tts-batch")
            if report["execution_model"] != "latent_first_decode_later":
                raise AssertionError("批量报告没有保留 Worker execution_model")
            if len(batch.successful_items()) != 3:
                raise AssertionError("初次批量配音没有产生 3 条成功音频")

            before_resume_calls = len(batch_calls)
            resumed = nodes.T8FireRedAudioBatchDubbing.execute(
                handle,
                plan,
                bank,
                "creator-loop",
                "fireredaudio/projects",
                True,
                True,
                2,
                settings,
            )
            resumed_batch, _resumed_manifest, resumed_report_text = resumed.result
            resumed_report = json.loads(resumed_report_text)
            if len(batch_calls) != before_resume_calls or resumed_report["cache_hits"] != 3:
                raise AssertionError("Manifest 恢复没有完整复用成功条目")

            original_line1 = Path(resumed_batch.items[0]["output_path"])
            original_digest = production.file_digest(original_line1)
            repair_calls: list[list[dict]] = []

            def fake_repair(_handle, requests):
                copied = [dict(request) for request in requests]
                repair_calls.append(copied)
                outcomes = []
                for index, request in enumerate(requests):
                    attempt = len(repair_calls)
                    if attempt == 1:
                        outcomes.append({"ok": False, "index": index, "error": "simulated QA retry"})
                    else:
                        path = Path(request["output_path"])
                        write_tone(path, frequency=770.0)
                        outcomes.append(
                            {"ok": True, "index": index, "result": {"output_path": str(path)}}
                        )
                return {
                    "outcomes": outcomes,
                    "performance": {
                        "execution_model": "latent_first_decode_later",
                        "batch_size": len(requests),
                    },
                }

            nodes._infer_tts_batch = fake_repair
            repaired = nodes.T8FireRedAudioBatchRetry.execute(
                handle,
                resumed_batch,
                plan,
                bank,
                "line-2",
                "creator-loop-repair",
                "fireredaudio/repairs",
                "increment",
                7,
                2,
                8,
                settings,
            )
            repaired_batch, repair_manifest, repair_report_text = repaired.result
            repair_report = json.loads(repair_report_text)
            if [call[0]["seed"] for call in repair_calls] != [42, 49]:
                raise AssertionError("返修 Seed 增量不正确")
            if repair_report["repaired_line_ids"] != ["line-2"]:
                raise AssertionError("失败项没有在第二次尝试中完成返修")
            if production.file_digest(original_line1) != original_digest:
                raise AssertionError("返修覆盖了已通过的源音频")
            if repaired_batch.items[0]["output_path"] != resumed_batch.items[0]["output_path"]:
                raise AssertionError("返修改变了非目标条目的路径")
            if repaired_batch.items[1]["output_path"] == resumed_batch.items[1]["output_path"]:
                raise AssertionError("返修条目没有切换到新文件")

            calls_before_passthrough = len(repair_calls)
            passthrough = nodes.T8FireRedAudioBatchRetry.execute(
                handle,
                repaired_batch,
                plan,
                bank,
                "",
                "unused",
                "fireredaudio/repairs",
                "increment",
                1,
                2,
                8,
                settings,
            )
            if passthrough.result[0] is not repaired_batch or len(repair_calls) != calls_before_passthrough:
                raise AssertionError("QA 全通过时没有无推理透传 AudioBatch")
            if json.loads(passthrough.result[2])["action"] != "passthrough_no_qa_failures":
                raise AssertionError("QA 全通过透传报告缺少明确状态")

            selected = nodes.T8FireRedAudioAudioBatchSelect.execute(
                repaired_batch, "line_id", 1, "line-2", ""
            )
            if selected.result[2] != "line-2":
                raise AssertionError("AudioBatch 选择节点没有选中指定 line ID")

            exported = nodes.T8FireRedAudioSaveAudioBatch.execute(
                repaired_batch,
                "wav",
                "creator-loop-delivery",
                "fireredaudio/exports",
                True,
                True,
                16,
            )
            saved_batch, export_manifest, zip_path, export_report_text = exported.result
            export_report = json.loads(export_report_text)
            if saved_batch is not repaired_batch or export_report["saved"] != 3:
                raise AssertionError("批量导出没有保存全部成功 Take")
            if not Path(export_manifest).is_file() or not Path(zip_path).is_file():
                raise AssertionError("批量导出缺少 Manifest 或 ZIP")
            if len((exported.ui or {}).get("audio", [])) != 3:
                raise AssertionError("批量导出没有注册全部原生试听/下载资产")

            result = {
                "batch_chunks": [len(call) for call in batch_calls],
                "cache_hits": resumed_report["cache_hits"],
                "repair_attempt_seeds": [call[0]["seed"] for call in repair_calls],
                "repair_manifest": Path(repair_manifest).name,
                "exported": export_report["saved"],
                "zip_created": True,
                "source_preserved": True,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            folder_paths.get_output_directory = original_output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
