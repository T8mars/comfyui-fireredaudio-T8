from __future__ import annotations

import json
import math
import tempfile
import unittest
import wave
from pathlib import Path

from runtime import asr_cache as CACHE


def write_tone(path: Path, frequency: float = 440.0) -> None:
    sample_rate = 8000
    frames = bytearray()
    for index in range(sample_rate // 10):
        value = round(math.sin(2 * math.pi * frequency * index / sample_rate) * 5000)
        frames.extend(int(value).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)


class ASRCacheTests(unittest.TestCase):
    def test_round_trip_and_identity_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            audio = root / "line.wav"
            write_tone(audio)
            descriptor = CACHE.build_asr_cache_descriptor(
                audio,
                model_revision="model-r1",
                model_fingerprint="fingerprint-a",
                prompt="Transcribe speech to text.",
                max_new_tokens=512,
            )
            self.assertIsNone(CACHE.load_cached_transcript(root / "cache", descriptor))
            target = CACHE.store_cached_transcript(root / "cache", descriptor, "测试转写")
            cached = CACHE.load_cached_transcript(root / "cache", descriptor)
            self.assertEqual(cached["transcript"], "测试转写")
            self.assertEqual(Path(cached["cache_path"]), target)

            changed = CACHE.build_asr_cache_descriptor(
                audio,
                model_revision="model-r2",
                model_fingerprint="fingerprint-a",
                prompt="Transcribe speech to text.",
                max_new_tokens=512,
            )
            self.assertNotEqual(descriptor["cache_key"], changed["cache_key"])
            self.assertIsNone(CACHE.load_cached_transcript(root / "cache", changed))

    def test_modified_audio_and_corrupt_entry_do_not_hit(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            audio = root / "line.wav"
            write_tone(audio)
            descriptor = CACHE.build_asr_cache_descriptor(
                audio,
                model_revision="model-r1",
                model_fingerprint="fingerprint-a",
                prompt="Transcribe speech to text.",
                max_new_tokens=512,
            )
            target = CACHE.store_cached_transcript(root / "cache", descriptor, "原转写")
            payload = json.loads(target.read_text(encoding="utf-8"))
            payload["descriptor"]["audio_sha256"] = "0" * 64
            target.write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(CACHE.load_cached_transcript(root / "cache", descriptor))

            write_tone(audio, frequency=880.0)
            changed = CACHE.build_asr_cache_descriptor(
                audio,
                model_revision="model-r1",
                model_fingerprint="fingerprint-a",
                prompt="Transcribe speech to text.",
                max_new_tokens=512,
            )
            self.assertNotEqual(descriptor["cache_key"], changed["cache_key"])

    def test_cache_path_rejects_untrusted_keys(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(ValueError):
                CACHE.cache_path(raw, "../escape")


if __name__ == "__main__":
    unittest.main()
