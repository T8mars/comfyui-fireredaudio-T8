from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fireredaudio_production", ROOT / "runtime" / "production.py"
)
PRODUCTION = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = PRODUCTION
SPEC.loader.exec_module(PRODUCTION)

POST_SPEC = importlib.util.spec_from_file_location(
    "fireredaudio_postproduction", ROOT / "runtime" / "postproduction.py"
)
POSTPRODUCTION = importlib.util.module_from_spec(POST_SPEC)
assert POST_SPEC and POST_SPEC.loader
sys.modules[POST_SPEC.name] = POSTPRODUCTION
POST_SPEC.loader.exec_module(POSTPRODUCTION)


def write_tone(path: Path, *, seconds: float = 0.1, sample_rate: int = 24000, frequency: float = 440.0) -> None:
    frames = bytearray()
    for index in range(round(seconds * sample_rate)):
        value = round(math.sin(2 * math.pi * frequency * index / sample_rate) * 8000)
        frames.extend(int(value).to_bytes(2, byteorder="little", signed=True))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)


def write_delayed_tone(
    path: Path,
    *,
    silence_seconds: float,
    tone_seconds: float = 0.5,
    sample_rate: int = 24000,
    amplitude: int = 8000,
) -> None:
    frames = bytearray(round(silence_seconds * sample_rate) * 2)
    for index in range(round(tone_seconds * sample_rate)):
        value = round(math.sin(2 * math.pi * 440.0 * index / sample_rate) * amplitude)
        frames.extend(int(value).to_bytes(2, byteorder="little", signed=True))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)


class ProductionWorkflowTests(unittest.TestCase):
    def test_desktop_project_exchange_loads_voice_plan_and_adopted_take(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            project = Path(raw) / "demo.firered"
            scripts = project / "scripts"
            assets = project / "assets"
            segments = project / "segments"
            scripts.mkdir(parents=True)
            assets.mkdir()
            segments.mkdir()
            prompt = assets / "voice.wav"
            take = segments / "take.wav"
            write_tone(prompt)
            write_tone(take)
            payload = {
                "format": "t8.firered.project.exchange",
                "version": 1,
                "project_root": "..",
                "project": {"name": "交换测试"},
                "voice_bank": {
                    "profiles": [
                        {
                            "profile_id": "voice-1",
                            "name": "旁白",
                            "prompt_audio": "assets/voice.wav",
                            "prompt_audio_sha256": PRODUCTION.file_digest(prompt),
                            "prompt_text": "参考文本",
                            "language": "zh",
                            "tags": ["沉稳"],
                        }
                    ]
                },
                "script_plan": {
                    "source_format": "desktop-project",
                    "lines": [
                        {
                            "line_id": "line-1",
                            "index": 1,
                            "speaker": "旁白",
                            "text": "目标台词",
                            "language": "zh",
                        }
                    ],
                    "issues": [],
                },
                "audio_batch": {
                    "items": [
                        {
                            "line_id": "line-1",
                            "status": "complete",
                            "output_path": "segments/take.wav",
                        }
                    ]
                },
            }
            exchange = scripts / "exchange.json"
            exchange.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            bank, plan, batch, report = PRODUCTION.load_project_exchange(exchange)
            self.assertEqual(bank.profiles[0].name, "旁白")
            self.assertEqual(plan.lines[0].text, "目标台词")
            self.assertEqual(len(batch.successful_items()), 1)
            self.assertEqual(report["adopted_takes"], 1)

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.narrator_wav = self.root / "narrator.wav"
        self.actor_wav = self.root / "actor.wav"
        write_tone(self.narrator_wav)
        write_tone(self.actor_wav, frequency=660.0)
        self.narrator = PRODUCTION.create_voice_profile(
            "旁白", self.narrator_wav, "参考音频内容", "zh", "温暖,清晰"
        )
        self.actor = PRODUCTION.create_voice_profile(
            "Alice", self.actor_wav, "Reference speech", "en"
        )
        self.bank = PRODUCTION.create_voice_bank([self.narrator, self.actor])

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_voice_bank_rejects_duplicate_names_case_insensitively(self) -> None:
        duplicate = PRODUCTION.create_voice_profile(
            "alice", self.narrator_wav, "another reference", "en"
        )
        with self.assertRaisesRegex(ValueError, "音色名称重复"):
            PRODUCTION.create_voice_bank([self.actor, duplicate])

    def test_role_script_preflight_resolves_speakers_and_inline_timing(self) -> None:
        plan = PRODUCTION.parse_script(
            "旁白：欢迎收听\n[00:00:01,000 --> 00:00:02,500] Alice: Welcome home",
            "role_script",
            self.bank,
        )
        self.assertTrue(plan.valid)
        self.assertEqual([line.speaker for line in plan.lines], ["旁白", "Alice"])
        self.assertEqual(plan.lines[1].language, "en")
        self.assertEqual(plan.lines[1].start_seconds, 1.0)
        self.assertEqual(plan.lines[1].end_seconds, 2.5)

    def test_srt_preflight_reports_unknown_speaker_and_overlap(self) -> None:
        script = """1
00:00:00,000 --> 00:00:02,000
[旁白] 第一行

2
00:00:01,500 --> 00:00:03,000
[未知角色] 第二行
"""
        plan = PRODUCTION.parse_script(script, "srt", self.bank)
        self.assertFalse(plan.valid)
        messages = [item["message"] for item in plan.issues]
        self.assertTrue(any("没有角色" in message for message in messages))
        self.assertTrue(any("重叠" in message for message in messages))

    def test_auto_detects_single_cue_srt_without_blank_separator(self) -> None:
        plan = PRODUCTION.parse_script(
            "1\n00:00:00,000 --> 00:00:02,000\n[旁白] 单条字幕",
            "auto",
            self.bank,
        )
        self.assertEqual(plan.source_format, "srt")
        self.assertTrue(plan.valid)

    def test_line_ids_survive_unrelated_insertions(self) -> None:
        original = PRODUCTION.parse_script("旁白：甲\n旁白：乙", "role_script", self.bank)
        changed = PRODUCTION.parse_script("Alice: Intro\n旁白：甲\n旁白：乙", "role_script", self.bank)
        self.assertEqual(
            [line.line_id for line in original.lines],
            [line.line_id for line in changed.lines[1:]],
        )

    def test_manifest_atomic_roundtrip_and_index(self) -> None:
        target = self.root / "project" / "manifest.json"
        payload = {
            "manifest_version": PRODUCTION.MANIFEST_VERSION,
            "items": [{"line_id": "a", "status": "complete"}],
        }
        PRODUCTION.write_manifest(target, payload)
        loaded = PRODUCTION.load_manifest(target)
        self.assertEqual(loaded, payload)
        self.assertEqual(PRODUCTION.manifest_items_by_id(loaded)["a"]["status"], "complete")
        self.assertFalse(target.with_name("manifest.json.tmp").exists())

    def test_manifest_resume_requires_matching_fingerprint_path_and_file(self) -> None:
        output = self.root / "line.wav"
        write_tone(output)
        item = {"status": "complete", "fingerprint": "expected", "output_path": str(output)}
        self.assertTrue(PRODUCTION.can_reuse_manifest_item(item, "expected", output))
        self.assertFalse(PRODUCTION.can_reuse_manifest_item(item, "changed", output))
        self.assertFalse(PRODUCTION.can_reuse_manifest_item({**item, "status": "failed"}, "expected", output))
        output.unlink()
        self.assertFalse(PRODUCTION.can_reuse_manifest_item(item, "expected", output))

    def test_timeline_sequence_and_overlay_render(self) -> None:
        items = [
            {"line_id": "a", "speaker": "旁白", "status": "complete", "output_path": str(self.narrator_wav)},
            {"line_id": "b", "speaker": "Alice", "status": "complete", "output_path": str(self.actor_wav)},
        ]
        sequence_path = self.root / "sequence.wav"
        report = PRODUCTION.render_timeline_to_wav(
            items, sequence_path, mode="sequence", gap_ms=100, peak_policy="limit"
        )
        self.assertTrue(sequence_path.is_file())
        self.assertAlmostEqual(report["duration_seconds"], 0.3, places=3)
        overlay_path = self.root / "overlay.wav"
        overlay = PRODUCTION.render_timeline_to_wav(
            items, overlay_path, mode="overlay", peak_policy="limit"
        )
        self.assertAlmostEqual(overlay["duration_seconds"], 0.1, places=3)
        self.assertEqual(len(overlay["placements"]), 2)

    def test_timeline_crossfade_overlaps_adjacent_sequence_clips(self) -> None:
        items = [
            {"line_id": "a", "status": "complete", "output_path": str(self.narrator_wav)},
            {"line_id": "b", "status": "complete", "output_path": str(self.actor_wav)},
        ]
        target = self.root / "crossfade.wav"
        report = PRODUCTION.render_timeline_to_wav(
            items,
            target,
            mode="sequence",
            gap_ms=500,
            crossfade_ms=50,
            peak_policy="limit",
        )
        self.assertAlmostEqual(report["duration_seconds"], 0.15, places=3)
        self.assertEqual(len(report["crossfades"]), 1)
        self.assertAlmostEqual(report["crossfades"][0]["duration_seconds"], 0.05, places=3)

    def test_timeline_room_tone_fills_only_real_gaps(self) -> None:
        items = [
            {
                "line_id": "a",
                "status": "complete",
                "output_path": str(self.narrator_wav),
                "start_seconds": 0.0,
            },
            {
                "line_id": "b",
                "status": "complete",
                "output_path": str(self.actor_wav),
                "start_seconds": 0.3,
            },
        ]
        room_tone = self.root / "room-tone.wav"
        write_tone(room_tone, seconds=0.05, frequency=80.0)
        target = self.root / "filled.wav"
        report = PRODUCTION.render_timeline_to_wav(
            items,
            target,
            mode="timeline",
            gap_ms=0,
            gap_fill_path=room_tone,
            auto_fill_gaps=True,
        )
        self.assertEqual(len(report["filled_gaps"]), 1)
        self.assertAlmostEqual(report["filled_gaps"][0]["duration_seconds"], 0.2, places=3)
        audio, sample_rate = PRODUCTION._read_wav_tensor(target)
        gap = audio[:, round(0.14 * sample_rate) : round(0.26 * sample_rate)]
        self.assertGreater(float(gap.abs().max().item()), 0.01)

    @unittest.skipUnless(
        (ROOT / "tools" / "ffmpeg.exe").is_file() or shutil.which("ffmpeg"),
        "FFmpeg is required for synchronized loudness A/B",
    )
    def test_synchronized_ab_aligns_onsets_matches_loudness_and_preserves_sources(self) -> None:
        source_a = self.root / "delayed-a.wav"
        source_b = self.root / "delayed-b.wav"
        output_a = self.root / "synced-a.wav"
        output_b = self.root / "synced-b.wav"
        write_delayed_tone(source_a, silence_seconds=0.15, amplitude=5000)
        write_delayed_tone(source_b, silence_seconds=0.35, amplitude=12000)
        digest_a = PRODUCTION.file_digest(source_a)
        digest_b = PRODUCTION.file_digest(source_b)
        report = POSTPRODUCTION.prepare_synchronized_ab(
            source_a,
            source_b,
            output_a,
            output_b,
            target_lufs=-20.0,
            preroll_ms=20,
        )
        self.assertEqual(PRODUCTION.file_digest(source_a), digest_a)
        self.assertEqual(PRODUCTION.file_digest(source_b), digest_b)
        self.assertTrue(report["source_preserved"])
        self.assertGreater(report["B"]["trimmed_seconds"], report["A"]["trimmed_seconds"])
        audio_a, rate_a = PRODUCTION._read_wav_tensor(output_a)
        audio_b, rate_b = PRODUCTION._read_wav_tensor(output_b)
        self.assertEqual(rate_a, rate_b)
        self.assertEqual(audio_a.shape, audio_b.shape)
        loudness_a = report["A"]["after"]["integrated_lufs"]
        loudness_b = report["B"]["after"]["integrated_lufs"]
        self.assertIsNotNone(loudness_a)
        self.assertIsNotNone(loudness_b)
        self.assertLess(abs(float(loudness_a) - float(loudness_b)), 0.3)

    def test_text_error_rate_uses_cer_for_zh_and_wer_for_en(self) -> None:
        metric, value = PRODUCTION.text_error_rate("你好，世界", "你好世界", "zh")
        self.assertEqual(metric, "cer")
        self.assertEqual(value, 0.0)
        metric, value = PRODUCTION.text_error_rate("Hello brave world", "hello world", "en")
        self.assertEqual(metric, "wer")
        self.assertAlmostEqual(value, 1 / 3)

    def test_wav_metrics_are_finite(self) -> None:
        metrics = PRODUCTION.wav_metrics(self.narrator_wav)
        self.assertEqual(metrics["sample_rate"], 24000)
        self.assertAlmostEqual(metrics["duration_seconds"], 0.1, places=3)
        self.assertLess(metrics["silence_ratio"], 0.1)
        json.dumps(metrics)


class ExampleWorkflowTests(unittest.TestCase):
    def test_production_examples_are_well_formed_and_linked(self) -> None:
        for name in ("13_role_dubbing_pipeline", "14_srt_dubbing_pipeline"):
            ui = json.loads((ROOT / "example_workflows" / "ui" / f"{name}.json").read_text(encoding="utf-8"))
            api = json.loads((ROOT / "example_workflows" / "api" / f"{name}.json").read_text(encoding="utf-8"))
            node_ids = {node["id"] for node in ui["nodes"]}
            self.assertEqual(len(node_ids), len(ui["nodes"]), name)
            link_ids = {link[0] for link in ui["links"]}
            self.assertEqual(len(link_ids), len(ui["links"]), name)
            for _link_id, source, _slot, target, _target_slot, _kind in ui["links"]:
                self.assertIn(source, node_ids, name)
                self.assertIn(target, node_ids, name)
            class_types = {node["class_type"] for node in api.values()}
            self.assertIn("T8_FireRedAudio_VoiceProfile", class_types)
            self.assertIn("T8_FireRedAudio_VoiceBank", class_types)
            self.assertIn("T8_FireRedAudio_ScriptParser", class_types)
            self.assertIn("T8_FireRedAudio_BatchDubbing", class_types)
            self.assertIn("T8_FireRedAudio_TimelineRender", class_types)
            bank_node = next(node for node in api.values() if node["class_type"] == "T8_FireRedAudio_VoiceBank")
            self.assertTrue(any(key.startswith("profiles.profile_") for key in bank_node["inputs"]), name)

    def test_reference_cleanup_example_is_linked_and_non_destructive(self) -> None:
        name = "15_reference_cleanup"
        ui = json.loads(
            (ROOT / "example_workflows" / "ui" / f"{name}.json").read_text(
                encoding="utf-8"
            )
        )
        api = json.loads(
            (ROOT / "example_workflows" / "api" / f"{name}.json").read_text(
                encoding="utf-8"
            )
        )
        node_ids = {node["id"] for node in ui["nodes"]}
        self.assertEqual(len(node_ids), len(ui["nodes"]))
        for _link_id, source, _slot, target, _target_slot, _kind in ui["links"]:
            self.assertIn(source, node_ids)
            self.assertIn(target, node_ids)
        quality_id = next(
            node_id
            for node_id, value in api.items()
            if value["class_type"] == "T8_FireRedAudio_ReferenceQuality"
        )
        cleanup_id = next(
            node_id
            for node_id, value in api.items()
            if value["class_type"] == "T8_FireRedAudio_PrepareReference"
        )
        self.assertEqual(api[cleanup_id]["inputs"]["audio"], [quality_id, 0])
        self.assertTrue(api[cleanup_id]["inputs"]["trim_silence"])
        self.assertFalse(api[cleanup_id]["inputs"]["normalize_loudness"])

    def test_srt_delivery_example_connects_room_tone_and_named_preset(self) -> None:
        api = json.loads(
            (ROOT / "example_workflows" / "api" / "14_srt_dubbing_pipeline.json").read_text(
                encoding="utf-8"
            )
        )
        preset_id = next(
            node_id
            for node_id, value in api.items()
            if value["class_type"] == "T8_FireRedAudio_DeliveryPreset"
        )
        timeline = next(
            value for value in api.values() if value["class_type"] == "T8_FireRedAudio_TimelineRender"
        )
        save_audio = next(
            value for value in api.values() if value["class_type"] == "T8_FireRedAudio_SaveAudio"
        )
        self.assertTrue(timeline["inputs"]["auto_fill_gaps"])
        self.assertIn("room_tone_audio", timeline["inputs"])
        self.assertEqual(timeline["inputs"]["delivery_preset"], [preset_id, 0])
        self.assertEqual(save_audio["inputs"]["delivery_preset"], [preset_id, 0])

    def test_synchronized_ab_example_has_two_independent_saved_outputs(self) -> None:
        api = json.loads(
            (ROOT / "example_workflows" / "api" / "16_synchronized_ab.json").read_text(
                encoding="utf-8"
            )
        )
        compare_id = next(
            node_id
            for node_id, value in api.items()
            if value["class_type"] == "T8_FireRedAudio_SynchronizedAB"
        )
        saves = [
            value for value in api.values() if value["class_type"] == "T8_FireRedAudio_SaveAudio"
        ]
        self.assertEqual(len(saves), 2)
        self.assertEqual({tuple(value["inputs"]["audio"]) for value in saves}, {(compare_id, 0), (compare_id, 1)})

    def test_all_ui_examples_keep_auto_safe_in_model_loader_widget(self) -> None:
        for path in sorted((ROOT / "example_workflows" / "ui").glob("*.json")):
            ui = json.loads(path.read_text(encoding="utf-8"))
            loaders = [
                node
                for node in ui["nodes"]
                if node["type"] == "T8_FireRedAudio_ModelLoader"
            ]
            for loader in loaders:
                self.assertGreaterEqual(len(loader["widgets_values"]), 5, path.name)
                self.assertEqual(loader["widgets_values"][4], "auto_safe", path.name)


if __name__ == "__main__":
    unittest.main()
