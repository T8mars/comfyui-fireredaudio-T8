from __future__ import annotations

import json
import math
import tempfile
import unittest
import wave
from pathlib import Path

from runtime import creator_tools as TOOLS
from runtime import production as PRODUCTION

ROOT = Path(__file__).resolve().parents[1]


def write_tone(path: Path, *, seconds: float, sample_rate: int = 24000) -> None:
    frames = bytearray()
    for index in range(round(seconds * sample_rate)):
        value = round(math.sin(2 * math.pi * 440.0 * index / sample_rate) * 7000)
        frames.extend(int(value).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)


class CreatorToolTests(unittest.TestCase):
    def test_normalization_preserves_source_and_exposes_spoken_text(self) -> None:
        plan = PRODUCTION.ScriptPlan(
            "role_script",
            (
                PRODUCTION.ScriptLine(
                    line_id="line-1",
                    index=1,
                    speaker="旁白",
                    text="ＡＰＩ  于 2026-08-28 发布",
                    language="zh",
                ),
            ),
            (),
        )
        normalized, report = TOOLS.normalize_script_plan(
            plan,
            replacements={"API": "A P I"},
            expand_zh_dates=True,
        )
        line = normalized.lines[0]
        self.assertEqual(line.source_text, "ＡＰＩ  于 2026-08-28 发布")
        self.assertEqual(line.text, "A P I 于 二零二六年八月二十八日 发布")
        self.assertIn("unicode_nfkc", line.normalization)
        self.assertIn("dictionary:API", line.normalization)
        self.assertIn("zh_dates", line.normalization)
        self.assertEqual(report["changed"], 1)
        self.assertEqual(plan.lines[0].text, "ＡＰＩ  于 2026-08-28 发布")

    def test_review_combines_qa_suggestions_and_manual_override(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            items = []
            for index in range(1, 4):
                path = root / f"line-{index}.wav"
                write_tone(path, seconds=0.1)
                items.append(
                    {
                        "line_id": f"line-{index}",
                        "index": index,
                        "speaker": "旁白",
                        "text": f"第{index}句",
                        "language": "zh",
                        "status": "complete",
                        "output_path": str(path),
                    }
                )
            qa = {
                "items": [
                    {"line_id": "line-1", "passed": True, "checks": {"text": True}},
                    {
                        "line_id": "line-2",
                        "passed": False,
                        "checks": {
                            "text": True,
                            "silence": False,
                            "clipping": True,
                            "cue_duration": True,
                        },
                    },
                    {
                        "line_id": "line-3",
                        "passed": False,
                        "checks": {
                            "text": False,
                            "silence": True,
                            "clipping": True,
                            "cue_duration": True,
                        },
                    },
                ]
            }
            source = PRODUCTION.AudioBatch("source.json", tuple(items))
            reviewed, delivery, report = TOOLS.build_line_review(
                source,
                qa=qa,
                decisions={"line-2": "approve"},
                ratings={"line-2": 5},
                notes={"line-2": "停顿符合表演"},
            )
            self.assertEqual(report["approved_line_ids"], ["line-1", "line-2"])
            self.assertEqual(report["retry_line_ids"], ["line-3"])
            self.assertEqual(report["review_line_ids"], [])
            self.assertEqual(len(delivery.successful_items()), 2)
            self.assertEqual(reviewed.items[1]["human_review"]["rating"], 5.0)
            self.assertTrue(reviewed.items[1]["human_review"]["manual_override"])
            self.assertEqual(source.items[1].get("human_review"), None)

    def test_manifest_resume_marks_missing_and_preserves_review_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            existing = root / "existing.wav"
            write_tone(existing, seconds=0.1)
            manifest = root / "review-manifest.json"
            PRODUCTION.write_manifest(
                manifest,
                {
                    "manifest_version": PRODUCTION.MANIFEST_VERSION,
                    "kind": "line_production_review",
                    "items": [
                        {
                            "line_id": "line-1",
                            "status": "complete",
                            "output_path": "existing.wav",
                            "human_review": {"effective_decision": "approve"},
                        },
                        {
                            "line_id": "line-2",
                            "status": "complete",
                            "output_path": "missing.wav",
                        },
                    ],
                },
            )
            batch, report = TOOLS.load_audio_batch_from_manifest(
                manifest,
                allowed_root=root,
                missing_policy="mark_missing",
            )
            self.assertEqual(batch.items[0]["output_path"], str(existing.resolve()))
            self.assertEqual(
                batch.items[0]["human_review"]["effective_decision"], "approve"
            )
            self.assertEqual(batch.items[1]["status"], "missing")
            self.assertEqual(report["missing"], ["line-2"])
            self.assertEqual(report["reviewed"], 1)

    def test_manifest_resume_rejects_audio_outside_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "output"
            output.mkdir()
            external = root / "external.wav"
            write_tone(external, seconds=0.1)
            manifest = output / "review-manifest.json"
            PRODUCTION.write_manifest(
                manifest,
                {
                    "manifest_version": PRODUCTION.MANIFEST_VERSION,
                    "items": [
                        {
                            "line_id": "line-1",
                            "status": "complete",
                            "output_path": str(external),
                        }
                    ],
                },
            )
            with self.assertRaisesRegex(ValueError, "音频必须位于"):
                TOOLS.load_audio_batch_from_manifest(
                    manifest,
                    allowed_root=output,
                )

    def test_duration_fit_is_non_destructive_and_flags_unsafe_overrun(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            safe = root / "safe.wav"
            unsafe = root / "unsafe.wav"
            write_tone(safe, seconds=1.10)
            write_tone(unsafe, seconds=1.50)
            safe_hash = PRODUCTION.file_digest(safe)
            unsafe_hash = PRODUCTION.file_digest(unsafe)
            batch = PRODUCTION.AudioBatch(
                "source.json",
                (
                    {
                        "line_id": "safe",
                        "index": 1,
                        "speaker": "旁白",
                        "status": "complete",
                        "output_path": str(safe),
                        "start_seconds": 0.0,
                        "end_seconds": 1.0,
                    },
                    {
                        "line_id": "unsafe",
                        "index": 2,
                        "speaker": "旁白",
                        "status": "complete",
                        "output_path": str(unsafe),
                        "start_seconds": 1.0,
                        "end_seconds": 2.0,
                    },
                ),
            )
            fitted, report = TOOLS.fit_audio_batch_to_cues(
                batch,
                root / "fit",
                maximum_speed=1.15,
                tolerance_seconds=0.05,
            )
            fitted_path = Path(fitted.items[0]["output_path"])
            self.assertTrue(fitted_path.is_file())
            self.assertNotEqual(fitted_path, safe)
            self.assertAlmostEqual(
                PRODUCTION.wav_metrics(fitted_path)["duration_seconds"], 1.0, delta=0.03
            )
            self.assertEqual(fitted.items[1]["output_path"], str(unsafe))
            self.assertEqual(report["adapted_line_ids"], ["safe"])
            self.assertEqual(report["retry_line_ids"], ["unsafe"])
            self.assertEqual(PRODUCTION.file_digest(safe), safe_hash)
            self.assertEqual(PRODUCTION.file_digest(unsafe), unsafe_hash)
            self.assertFalse(report["source_files_overwritten"])

    def test_frontend_review_widget_is_packaged_and_binds_serialized_inputs(
        self,
    ) -> None:
        source = (ROOT / "web" / "line_review_v014.js").read_text(encoding="utf-8")
        self.assertIn("T8_FireRedAudio_LineReview", source)
        self.assertIn("decisions_json", source)
        self.assertIn("ratings_json", source)
        self.assertIn("notes_json", source)
        self.assertIn('note.addEventListener("input", sync)', source)
        self.assertIn("addDOMWidget", source)
        self.assertIn("fireredaudio_review", source)

    def test_v014_workflows_close_review_repair_and_resume_loops(self) -> None:
        production = json.loads(
            (
                ROOT / "example_workflows" / "api" / "26_production_review_loop.json"
            ).read_text(encoding="utf-8")
        )
        resume = json.loads(
            (
                ROOT / "example_workflows" / "api" / "27_resume_review_session.json"
            ).read_text(encoding="utf-8")
        )
        types = [value["class_type"] for value in production.values()]
        self.assertIn("T8_FireRedAudio_TextNormalizer", types)
        self.assertIn("T8_FireRedAudio_DurationFit", types)
        self.assertEqual(types.count("T8_FireRedAudio_LineReview"), 2)
        self.assertIn("T8_FireRedAudio_BatchRetry", types)
        self.assertEqual(production["12"]["inputs"]["audio_batch"], ["11", 0])
        self.assertEqual(production["12"]["inputs"]["failed_line_ids"], ["11", 2])
        self.assertTrue(production["12"]["inputs"]["enforce_cue_duration"])
        self.assertTrue(production["10"]["inputs"]["use_asr_cache"])
        self.assertFalse(production["10"]["inputs"]["refresh_asr_cache"])
        self.assertTrue(production["13"]["inputs"]["use_asr_cache"])
        self.assertFalse(production["13"]["inputs"]["refresh_asr_cache"])
        self.assertEqual(
            production["12"]["inputs"]["max_cue_overrun_seconds"], 0.5
        )
        self.assertEqual(production["14"]["inputs"]["audio_batch"], ["12", 0])
        self.assertEqual(production["15"]["inputs"]["audio_batch"], ["14", 1])
        self.assertEqual(resume["1"]["class_type"], "T8_FireRedAudio_AudioBatchResume")
        self.assertEqual(resume["2"]["inputs"]["audio_batch"], ["1", 0])
        self.assertEqual(resume["3"]["inputs"]["audio_batch"], ["2", 1])


if __name__ == "__main__":
    unittest.main()
