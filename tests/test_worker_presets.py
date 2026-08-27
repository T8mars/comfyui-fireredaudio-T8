from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker_bundle"))

from fireredaudio_t8.errors import WorkerProtocolError
from fireredaudio_t8.presets import apply_quality_preset


class WorkerPresetTests(unittest.TestCase):
    def test_custom_quality_parameters_are_accepted_and_normalized(self) -> None:
        result = apply_quality_preset(
            {
                "quality_preset": "custom",
                "max_new_audio_steps": "300",
                "min_new_audio_steps": 6,
                "max_new_text_tokens": 128,
                "n_timesteps": 6,
                "inference_cfg": "1.8",
            }
        )
        self.assertEqual(result["max_new_audio_steps"], 300)
        self.assertEqual(result["inference_cfg"], 1.8)

    def test_custom_quality_rejects_missing_or_inverted_ranges(self) -> None:
        with self.assertRaisesRegex(WorkerProtocolError, "缺少有效"):
            apply_quality_preset({"quality_preset": "custom"})
        with self.assertRaisesRegex(WorkerProtocolError, "不能大于"):
            apply_quality_preset(
                {
                    "quality_preset": "custom",
                    "max_new_audio_steps": 10,
                    "min_new_audio_steps": 20,
                    "max_new_text_tokens": 128,
                    "n_timesteps": 6,
                    "inference_cfg": 1.8,
                }
            )


if __name__ == "__main__":
    unittest.main()
