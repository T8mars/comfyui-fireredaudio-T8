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


def write_tone(path: Path, *, frequency: float, seconds: float = 0.12) -> None:
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
    package = "fireredaudio_v016_candidate_validation"
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
    parser = argparse.ArgumentParser(description="Validate creative candidates and explicit adoption")
    parser.add_argument("--comfy-root", required=True)
    args = parser.parse_args()
    nodes, production, types = load_modules(Path(args.comfy_root))
    import folder_paths

    with tempfile.TemporaryDirectory(prefix="fireredaudio-v016-candidates-") as raw:
        root = Path(raw)
        output = root / "output"
        output.mkdir()
        original_output = folder_paths.get_output_directory
        folder_paths.get_output_directory = lambda: str(output)
        try:
            prompt = root / "prompt.wav"
            original_a = root / "original-a.wav"
            original_b = root / "original-b.wav"
            write_tone(prompt, frequency=330.0)
            write_tone(original_a, frequency=440.0)
            write_tone(original_b, frequency=550.0)
            original_a_hash = production.file_digest(original_a)
            original_b_hash = production.file_digest(original_b)
            profile = production.create_voice_profile("旁白", prompt, "参考逐字稿", "zh")
            bank = production.create_voice_bank([profile])
            plan = production.ScriptPlan(
                "role_script",
                (
                    production.ScriptLine("line-1", 1, "旁白", "第一句", "zh"),
                    production.ScriptLine("line-2", 2, "旁白", "第二句", "zh"),
                ),
                (),
            )
            source_batch = production.AudioBatch(
                "source-manifest.json",
                (
                    {"line_id": "line-1", "index": 1, "speaker": "旁白", "text": "第一句", "language": "zh", "status": "complete", "output_path": str(original_a)},
                    {"line_id": "line-2", "index": 2, "speaker": "旁白", "text": "第二句", "language": "zh", "status": "complete", "output_path": str(original_b)},
                ),
            )
            handle = types.RuntimeHandle(model_root=str(root / "model"), release_after=False)
            settings = types.GenerationSettings(seed=42)
            observed_seeds: list[int] = []

            def fake_batch(_handle, requests):
                outcomes = []
                for index, request in enumerate(requests):
                    observed_seeds.append(int(request["seed"]))
                    path = Path(request["output_path"])
                    write_tone(path, frequency=660.0 + index * 110.0)
                    outcomes.append(
                        {
                            "ok": True,
                            "index": index,
                            "result": {"output_path": str(path), "seed": int(request["seed"])},
                        }
                    )
                return {"outcomes": outcomes, "performance": {"batch_size": len(requests)}}

            nodes._infer_tts_batch = fake_batch
            pool_output = nodes.T8FireRedAudioCreativeCandidatePool.execute(
                handle,
                source_batch,
                plan,
                bank,
                "line-1",
                3,
                1001,
                97,
                True,
                False,
                "candidate-validation",
                "fireredaudio/candidates",
                settings,
            )
            candidates, source_line_id, _manifest_path, report_text = pool_output.result
            report = json.loads(report_text)
            if observed_seeds != [1001, 1098, 1195]:
                raise AssertionError(f"候选 Seed 证据错误：{observed_seeds}")
            if report["requested_seeds"] != observed_seeds or source_line_id != "line-1":
                raise AssertionError("候选报告没有保留目标 line ID/Seed")
            if report["distinct_audio_hashes"] != 4 or report["duplicate_candidate_groups"]:
                raise AssertionError("候选音频没有形成可区分的候选池")
            if not report["blind_filenames"] or report["automatic_adoption"]:
                raise AssertionError("候选池没有保持盲听和显式采用语义")

            review_output = nodes.T8FireRedAudioTakeReviewBoard.execute(
                candidates,
                1,
                "candidate-002",
                '{"candidate-002": 5}',
                '{"candidate-002": "人工采用"}',
                "candidate-review",
                "fireredaudio/reviews",
                8,
            )
            _selected_audio, reviewed, selected_id, _review_manifest, _review_report = review_output.result
            adopted_output = nodes.T8FireRedAudioCandidateApply.execute(
                source_batch,
                reviewed,
                selected_id,
                "candidate-adoption",
                "fireredaudio/candidate-adoptions",
            )
            adopted, _audio, adoption_manifest, adoption_text = adopted_output.result
            adoption = json.loads(adoption_text)
            if adoption["selected_candidate_id"] != "candidate-002" or adoption["selected_seed"] != 1098:
                raise AssertionError("显式采用没有保存候选 ID/Seed")
            if adopted.items[0]["output_path"] == str(original_a):
                raise AssertionError("目标台词没有回填选中候选")
            if adopted.items[1]["output_path"] != str(original_b):
                raise AssertionError("候选采用改变了非目标台词")
            if production.file_digest(original_a) != original_a_hash or production.file_digest(original_b) != original_b_hash:
                raise AssertionError("候选生成或采用覆盖了源音频")
            if not Path(adoption_manifest).is_file():
                raise AssertionError("候选采用 Manifest 未写入")
            print(json.dumps({"observed_seeds": observed_seeds, "candidate_report": report, "adoption": adoption}, ensure_ascii=False, indent=2))
        finally:
            folder_paths.get_output_directory = original_output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
