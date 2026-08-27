from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fireredaudio_reference_candidates",
    ROOT / "runtime" / "reference_candidates.py",
)
REFERENCE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = REFERENCE
SPEC.loader.exec_module(REFERENCE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_long_recording(path: Path, *, sample_rate: int = 8000) -> None:
    frames = bytearray()
    for index in range(sample_rate * 20):
        seconds = index / sample_rate
        if 2.0 <= seconds < 10.0:
            left = round(math.sin(2 * math.pi * 180.0 * seconds) * 7000)
            right = round(math.sin(2 * math.pi * 220.0 * seconds) * 9000)
        elif 12.0 <= seconds < 18.0:
            sign = 1 if math.sin(2 * math.pi * 160.0 * seconds) >= 0 else -1
            left = right = sign * 32767
        else:
            left = right = 0
        frames.extend(int(left).to_bytes(2, "little", signed=True))
        frames.extend(int(right).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)


class ReferenceCandidateTests(unittest.TestCase):
    def test_discovers_ranked_candidates_and_preserves_source_format(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "long.wav"
            output = root / "candidates"
            write_long_recording(source)
            before = digest(source)
            items, report = REFERENCE.discover_reference_candidates(
                source,
                output,
                min_seconds=3.0,
                preferred_seconds=6.0,
                max_seconds=8.0,
                max_candidates=4,
            )
            self.assertGreaterEqual(len(items), 2)
            self.assertLessEqual(len(items), 4)
            self.assertEqual(before, digest(source))
            self.assertTrue(report["source_preserved"])
            self.assertEqual(report["sample_rate"], 8000)
            self.assertEqual(report["channels"], 2)
            self.assertGreaterEqual(
                float(items[0]["signal_score"]),
                float(items[-1]["signal_score"]),
            )
            self.assertLess(float(items[0]["metrics"]["clipping_ratio"]), 0.001)
            for item in items:
                target = Path(item["output_path"])
                self.assertTrue(target.is_file())
                with wave.open(str(target), "rb") as reader:
                    self.assertEqual(reader.getframerate(), 8000)
                    self.assertEqual(reader.getnchannels(), 2)

    def test_asr_proxy_is_explicit_and_rewards_plausible_text(self) -> None:
        empty = REFERENCE.asr_intelligibility_proxy("", 5.0, "zh")
        clear = REFERENCE.asr_intelligibility_proxy("今天我们开始录制这一段清晰的参考声音", 5.0, "zh")
        self.assertEqual(empty["score"], 0.0)
        self.assertGreater(clear["score"], empty["score"])
        self.assertIn("不是", clear["notice"])

    def test_rejects_recording_shorter_than_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "short.wav"
            frames = b"\x00\x00" * 8000
            with wave.open(str(source), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(8000)
                writer.writeframes(frames)
            with self.assertRaisesRegex(ValueError, "短于"):
                REFERENCE.discover_reference_candidates(
                    source,
                    root / "out",
                    min_seconds=3.0,
                )


if __name__ == "__main__":
    unittest.main()
