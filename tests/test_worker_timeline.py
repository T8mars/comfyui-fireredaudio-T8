from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker_bundle"))

from fireredaudio_t8.timeline import render_timeline


def write_stereo(path: Path, frames: int = 2400) -> None:
    values = array("h")
    for index in range(frames):
        values.extend((4000 if index % 2 else -4000, 9000 if index % 3 else -9000))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(values.tobytes())


class BundledWorkerTimelineTests(unittest.TestCase):
    def test_project_timeline_preserves_stereo(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "stereo.wav"
            target = root / "render.wav"
            write_stereo(source)
            result = render_timeline([{"id": "clip", "path": str(source)}], target)
            self.assertEqual(result.channels, 2)
            with wave.open(str(target), "rb") as reader:
                self.assertEqual(reader.getnchannels(), 2)
                samples = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2").reshape(-1, 2)
            self.assertFalse(np.array_equal(samples[:, 0], samples[:, 1]))


if __name__ == "__main__":
    unittest.main()
