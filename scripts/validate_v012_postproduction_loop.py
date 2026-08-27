from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
import wave
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_tone(
    path: Path,
    *,
    frequency: float = 440.0,
    seconds: float = 0.5,
    sample_rate: int = 24_000,
    channels: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for index in range(round(sample_rate * seconds)):
        for channel in range(channels):
            value = round(
                math.sin(2 * math.pi * frequency * (channel + 1) * index / sample_rate)
                * (5000 + channel * 1000)
            )
            frames.extend(int(value).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)


def load_modules(comfy_root: Path):
    sys.path.insert(0, str(comfy_root.resolve()))
    package = "fireredaudio_v012_validation"
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
        sys.modules[f"{package}.runtime.audio_adapter"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute v0.12 local repair and production package loops")
    parser.add_argument("--comfy-root", required=True)
    args = parser.parse_args()
    nodes, production, audio_adapter = load_modules(Path(args.comfy_root))
    import folder_paths

    with tempfile.TemporaryDirectory(prefix="fireredaudio-v012-") as raw:
        root = Path(raw)
        output = root / "output"
        temporary = root / "temp"
        output.mkdir()
        temporary.mkdir()
        original_output = folder_paths.get_output_directory
        original_temp = folder_paths.get_temp_directory
        folder_paths.get_output_directory = lambda: str(output)
        folder_paths.get_temp_directory = lambda: str(temporary)
        try:
            source = root / "source-stereo-48k.wav"
            replacement = root / "replacement-mono-24k.wav"
            second = root / "second.wav"
            write_tone(source, frequency=330.0, seconds=0.8, sample_rate=48_000, channels=2)
            write_tone(replacement, frequency=770.0, seconds=0.25, sample_rate=24_000)
            write_tone(second, frequency=550.0, seconds=0.30, sample_rate=24_000)
            source_digest = production.file_digest(source)

            ranged = nodes.T8FireRedAudioLocalRepairRange.execute(
                audio_adapter.wav_to_audio(source),
                "manual",
                0.20,
                0.50,
                1,
                25,
                "",
            )
            original_audio, _repair_clip, repair_plan, range_report_text = ranged.result
            range_report = json.loads(range_report_text)
            if range_report["sample_rate"] != 48_000 or range_report["channels"] != 2:
                raise AssertionError("修复范围没有保留源采样率/声道")
            located = nodes.T8FireRedAudioLocalRepairRange.execute(
                audio_adapter.wav_to_audio(source),
                "locator_json",
                0.0,
                1.0,
                1,
                0,
                json.dumps(
                    {"matches": [{"start_time": "00:00:00.100", "end_time": "00:00:00.300", "label": "定位范围"}]},
                    ensure_ascii=False,
                ),
            )
            located_report = json.loads(located.result[3])
            if located_report["range_source"] != "locator_json" or located_report["range_label"] != "定位范围":
                raise AssertionError("定位 JSON 没有驱动局部修复范围")

            applied = nodes.T8FireRedAudioLocalRepairApply.execute(
                repair_plan,
                audio_adapter.wav_to_audio(replacement),
                30,
            )
            original_ab, repaired_audio, replacement_report_text = applied.result
            replacement_report = json.loads(replacement_report_text)
            if production.file_digest(source) != source_digest:
                raise AssertionError("局部修复覆盖了源音频")
            if replacement_report["crossfade_curve"] != "equal_power":
                raise AssertionError("局部修复未使用等功率交叉淡化")
            if replacement_report["sample_rate"] != 48_000 or replacement_report["channels"] != 2:
                raise AssertionError("回填输出没有保留源采样率/声道")
            if original_audio["sample_rate"] != original_ab["sample_rate"]:
                raise AssertionError("A/B 原版输出不一致")

            repaired_path = Path(replacement_report["output_path"])
            source_manifest = root / "source-manifest.json"
            items = [
                {
                    "line_id": "line-1",
                    "index": 1,
                    "speaker": "旁白",
                    "scene": "开场",
                    "text": "第一句",
                    "status": "complete",
                    "output_path": str(repaired_path),
                    "start_seconds": 0.0,
                    "worker_report": {"code_revision": "code-test", "model_revision": "model-test"},
                },
                {
                    "line_id": "line-2",
                    "index": 2,
                    "speaker": "角色甲",
                    "scene": "冲突",
                    "text": "第二句",
                    "status": "complete",
                    "output_path": str(second),
                    "start_seconds": 0.9,
                    "worker_report": {"code_revision": "code-test", "model_revision": "model-test"},
                },
            ]
            production.write_manifest(
                source_manifest,
                {
                    "manifest_version": production.MANIFEST_VERSION,
                    "settings": {"seed": 42, "quality_preset": "balanced"},
                    "model_identity": "model-identity-test",
                    "items": items,
                },
            )
            batch = production.AudioBatch(str(source_manifest), tuple(items))
            packaged = nodes.T8FireRedAudioProductionPackage.execute(
                batch,
                "postproduction-loop",
                "fireredaudio/deliveries",
                48_000,
                30,
                True,
                True,
                True,
                repaired_audio,
                audio_adapter.wav_to_audio(second),
                audio_adapter.wav_to_audio(source),
                "1\n00:00:00,000 --> 00:00:00,500\n第一句\n",
                None,
            )
            returned_batch, master_audio, manifest_path, zip_path, package_report_text = packaged.result
            package_report = json.loads(package_report_text)
            if returned_batch is not batch or master_audio["sample_rate"] != 48_000:
                raise AssertionError("制作包没有透传批次或返回 Master")
            if not package_report["source_hashes_preserved"]:
                raise AssertionError("制作包改变了源 Take")
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            if manifest["versions"]["worker_model_revisions"] != ["model-test"]:
                raise AssertionError("制作包没有记录模型版本")
            if manifest["generation"]["settings"]["seed"] != 42:
                raise AssertionError("制作包没有记录生成参数")
            if package_report["role_stems"] != 2 or package_report["scene_stems"] != 2:
                raise AssertionError("制作包没有完整导出角色/场景分轨")
            expected_entries = {
                "mix/master.wav",
                "mix/dialogue-master.wav",
                "assets/bgm.wav",
                "assets/room-tone.wav",
                "subtitles/postproduction-loop.srt",
                "subtitles/postproduction-loop.vtt",
                "production-manifest.json",
            }
            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())
            if not expected_entries.issubset(names):
                raise AssertionError(f"交付 ZIP 缺少文件：{sorted(expected_entries - names)}")
            if len((packaged.ui or {}).get("audio", [])) < 6:
                raise AssertionError("制作包没有注册 Master、分轨与素材试听")

            print(
                json.dumps(
                    {
                        "local_repair": {
                            "source_preserved": True,
                            "sample_rate": replacement_report["sample_rate"],
                            "channels": replacement_report["channels"],
                            "crossfade": replacement_report["crossfade_curve"],
                        },
                        "production_package": {
                            "role_stems": package_report["role_stems"],
                            "scene_stems": package_report["scene_stems"],
                            "subtitle_cues": package_report["subtitle_cues"],
                            "zip_entries": len(names),
                            "model_version_recorded": True,
                            "settings_recorded": True,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        finally:
            folder_paths.get_output_directory = original_output
            folder_paths.get_temp_directory = original_temp
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
