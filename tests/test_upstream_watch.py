from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fireredaudio_upstream_watch", ROOT / "scripts" / "check_upstream_revisions.py"
)
WATCH = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = WATCH
SPEC.loader.exec_module(WATCH)


class UpstreamWatchTests(unittest.TestCase):
    def test_manifest_pins_are_complete(self) -> None:
        pins = WATCH.load_pins()
        self.assertEqual(pins["codeRepository"], "FireRedTeam/FireRedAudio")
        self.assertEqual(len(pins["codeRevision"]), 40)
        self.assertEqual(len(pins["modelRevision"]), 40)

    def test_revision_comparison_reports_each_changed_source(self) -> None:
        pins = WATCH.load_pins()
        current = WATCH.compare_revisions(
            pins, pins["codeRevision"], pins["modelRevision"]
        )
        self.assertFalse(current["updates_available"])
        changed = WATCH.compare_revisions(
            pins, "a" * 40, pins["modelRevision"]
        )
        self.assertTrue(changed["updates_available"])
        self.assertTrue(changed["code"]["changed"])
        self.assertFalse(changed["model"]["changed"])
        self.assertIn(pins["codeRevision"], changed["code"]["compare_url"])


if __name__ == "__main__":
    unittest.main()
