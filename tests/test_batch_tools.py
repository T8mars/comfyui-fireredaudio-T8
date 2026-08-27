from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PRODUCTION = load_module("fireredaudio_batch_production", ROOT / "runtime" / "production.py")
AUDIO = load_module("fireredaudio_batch_audio", ROOT / "runtime" / "audio_adapter.py")


def write_tone(path: Path, *, frequency: float = 440.0) -> None:
    sample_rate = 24_000
    frames = bytearray()
    for index in range(2400):
        value = round(math.sin(2 * math.pi * frequency * index / sample_rate) * 6000)
        frames.extend(int(value).to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)


class BatchToolTests(unittest.TestCase):
    def test_parse_line_ids_accepts_json_and_delimited_text(self) -> None:
        self.assertEqual(PRODUCTION.parse_line_ids('["a", "b", "a"]'), ("a", "b"))
        self.assertEqual(PRODUCTION.parse_line_ids("a, b\nc；d"), ("a", "b", "c", "d"))
        self.assertEqual(PRODUCTION.parse_line_ids(""), ())

    def test_select_and_merge_are_ordered_and_non_destructive(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "first.wav"
            second = root / "second.wav"
            replacement = root / "replacement.wav"
            write_tone(first, frequency=330.0)
            write_tone(second, frequency=440.0)
            write_tone(replacement, frequency=550.0)
            items = (
                {"line_id": "a", "speaker": "旁白", "status": "complete", "output_path": str(first)},
                {"line_id": "b", "speaker": "角色", "status": "complete", "output_path": str(second)},
            )
            batch = PRODUCTION.AudioBatch(str(root / "source.json"), items)
            self.assertEqual(
                PRODUCTION.select_audio_batch_item(batch, mode="position", position=2)["line_id"],
                "b",
            )
            self.assertEqual(
                PRODUCTION.select_audio_batch_item(batch, mode="speaker", speaker="角色")["line_id"],
                "b",
            )
            merged = PRODUCTION.merge_audio_batch_items(
                batch,
                [
                    {
                        **items[1],
                        "output_path": str(replacement),
                        "repaired": True,
                    }
                ],
                root / "merged.json",
            )
            self.assertEqual([item["line_id"] for item in merged.items], ["a", "b"])
            self.assertEqual(merged.items[0]["output_path"], str(first))
            self.assertEqual(merged.items[1]["output_path"], str(replacement))
            self.assertEqual(batch.items[1]["output_path"], str(second))

    def test_export_audio_path_and_multi_asset_ui(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "output"
            output.mkdir()
            source = root / "source.wav"
            target_a = output / "batch" / "a.wav"
            target_b = output / "batch" / "b.wav"
            write_tone(source)
            AUDIO.export_audio_path(source, target_a, audio_format="wav")
            AUDIO.export_audio_path(source, target_b, audio_format="wav")
            previous = sys.modules.get("folder_paths")
            sys.modules["folder_paths"] = types.SimpleNamespace(
                get_output_directory=lambda: str(output)
            )
            try:
                ui = AUDIO.saved_audio_files_ui([target_a, target_b])
            finally:
                if previous is None:
                    sys.modules.pop("folder_paths", None)
                else:
                    sys.modules["folder_paths"] = previous
            self.assertEqual(len(ui["audio"]), 2)
            self.assertEqual(ui["audio"][0]["subfolder"], "batch")
            self.assertEqual(ui["audio"][1]["filename"], "b.wav")


if __name__ == "__main__":
    unittest.main()
