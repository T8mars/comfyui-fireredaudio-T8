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


def write_recording(path: Path, *, seconds: float = 18.0, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for index in range(round(seconds * sample_rate)):
        timestamp = index / sample_rate
        if 1.0 <= timestamp < 8.0 or 10.0 <= timestamp < 17.0:
            amplitude = 6500 if timestamp < 8.0 else 9500
            value = round(math.sin(2 * math.pi * 190.0 * timestamp) * amplitude)
        else:
            value = 0
        frames.extend(int(value).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)


def write_tone(path: Path, *, frequency: float = 440.0, seconds: float = 0.25) -> None:
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
    package = "fireredaudio_v013_validation"
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
        sys.modules[f"{package}.runtime.types"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute v0.13 reference, review and benchmark tools")
    parser.add_argument("--comfy-root", required=True)
    args = parser.parse_args()
    nodes, production, audio_adapter, runtime_types = load_modules(Path(args.comfy_root))
    import folder_paths

    with tempfile.TemporaryDirectory(prefix="fireredaudio-v013-") as raw:
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
            recording = root / "long-recording.wav"
            write_recording(recording)
            before = production.file_digest(recording)
            screened = nodes.T8FireRedAudioReferenceCandidates.execute(
                audio_adapter.wav_to_audio(recording),
                "validation-reference",
                "fireredaudio/reference-candidates",
                3.0,
                6.0,
                8.0,
                0.2,
                4,
                False,
                "zh",
                None,
            )
            recommended_audio, candidates, recommended_id, candidate_manifest, ranking_text = screened.result
            ranking = json.loads(ranking_text)
            if production.file_digest(recording) != before or not ranking["source_preserved"]:
                raise AssertionError("参考筛选覆盖了源录音")
            if not 2 <= len(candidates.items) <= 4:
                raise AssertionError("参考筛选候选数量异常")
            if recommended_audio["sample_rate"] != 8000:
                raise AssertionError("推荐候选没有保留源采样率")
            if len((screened.ui or {}).get("audio", [])) != len(candidates.items):
                raise AssertionError("参考候选没有注册原生试听/下载")
            if not Path(candidate_manifest).is_file():
                raise AssertionError("参考候选 Manifest 缺失")

            second_id = str(candidates.items[min(1, len(candidates.items) - 1)]["line_id"])
            reviewed = nodes.T8FireRedAudioTakeReviewBoard.execute(
                candidates,
                1,
                second_id,
                json.dumps({recommended_id: 4, second_id: 5}, ensure_ascii=False),
                json.dumps({second_id: "人工试听采用"}, ensure_ascii=False),
                "validation-review",
                "fireredaudio/reviews",
                4,
            )
            selected_audio, reviewed_batch, selected_id, review_manifest, review_text = reviewed.result
            review = json.loads(review_text)
            if selected_id != second_id or review["selected_line_id"] != second_id:
                raise AssertionError("评审板没有采用指定 line ID")
            if not Path(review_manifest).is_file() or selected_audio["sample_rate"] != 8000:
                raise AssertionError("评审结果或选中音频无效")
            adopted = [
                item for item in reviewed_batch.items if item["human_review"]["adopted"]
            ]
            if len(adopted) != 1 or adopted[0]["line_id"] != second_id:
                raise AssertionError("评审 Manifest 的唯一采用状态错误")

            reference = root / "reference.wav"
            write_tone(reference)
            original_infer = nodes._infer
            original_client = nodes._client

            class FakeClient:
                def __init__(self, handle):
                    self.handle = handle

                def unload(self):
                    return {"unloaded": True}

                def health(self):
                    requested = self.handle.acceleration_mode
                    fallback = requested == "deepspeed"
                    return {
                        "acceleration": {
                            "selection": {
                                "requested": requested,
                                "effective": "off" if fallback else requested,
                                "available": not fallback,
                                "reason": "validation fallback" if fallback else "validation available",
                            }
                        }
                    }

            def fake_client(handle):
                return FakeClient(handle)

            def fake_infer(handle, request):
                mode = handle.acceleration_mode
                frequency = {"off": 330.0, "flash_attention": 440.0, "deepspeed": 550.0}[mode]
                write_tone(Path(request["output_path"]), frequency=frequency)
                elapsed = {"off": 10.0, "flash_attention": 8.0, "deepspeed": 7.0}[mode]
                return {
                    "output_path": request["output_path"],
                    "performance": {
                        "total_seconds": elapsed,
                        "rtf": elapsed / 0.25,
                        "gpu_peak_allocated_bytes": {
                            "off": 10,
                            "flash_attention": 9,
                            "deepspeed": 11,
                        }[mode]
                        * 1024**3,
                        "acceleration_mode": mode,
                    },
                }

            nodes._client = fake_client
            nodes._infer = fake_infer
            try:
                benchmarked = nodes.T8FireRedAudioAccelerationBenchmark.execute(
                    runtime_types.RuntimeHandle(model_root=str(root / "models")),
                    audio_adapter.wav_to_audio(reference),
                    "固定参考逐字稿。",
                    "固定目标文本。",
                    "zh",
                    "off,flash_attention,deepspeed",
                    1,
                    3,
                    10.0,
                    True,
                    "validation-benchmark",
                    "fireredaudio/benchmarks",
                    runtime_types.GenerationSettings(quality_preset="fast", seed=42),
                )
            finally:
                nodes._client = original_client
                nodes._infer = original_infer
            benchmark_batch, recommendation, benchmark_text, benchmark_manifest = benchmarked.result
            benchmark = json.loads(benchmark_text)
            if benchmark["recommended_mode"] != "flash_attention":
                raise AssertionError(f"基准建议错误：{recommendation}")
            if benchmark["settings_modified"]:
                raise AssertionError("加速向导不应自动修改设置")
            if len(benchmark_batch.items) != 9:
                raise AssertionError("加速向导没有完成每模式三次正式测量")
            deep = next(item for item in benchmark["mode_reports"] if item["requested_mode"] == "deepspeed")
            if not deep["fallback_detected"] or deep.get("eligible_for_recommendation"):
                raise AssertionError("加速向导没有排除回退模式")
            if not Path(benchmark_manifest).is_file():
                raise AssertionError("基准 Manifest 缺失")

            print(
                json.dumps(
                    {
                        "reference_candidates": {
                            "count": len(candidates.items),
                            "source_preserved": True,
                            "native_previews": len((screened.ui or {}).get("audio", [])),
                        },
                        "take_review": {
                            "selected_line_id": selected_id,
                            "single_adopted": True,
                            "source_files_overwritten": review["source_files_overwritten"],
                        },
                        "acceleration_benchmark": {
                            "formal_runs": len(benchmark_batch.items),
                            "recommended_mode": benchmark["recommended_mode"],
                            "fallback_excluded": deep["fallback_detected"],
                            "settings_modified": benchmark["settings_modified"],
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
