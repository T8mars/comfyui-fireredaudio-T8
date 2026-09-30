from __future__ import annotations

import importlib.util
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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
        self.assertFalse(current["audit_required"])
        changed = WATCH.compare_revisions(
            pins, "a" * 40, pins["modelRevision"]
        )
        self.assertTrue(changed["updates_available"])
        self.assertTrue(changed["audit_required"])
        self.assertTrue(changed["code"]["changed"])
        self.assertFalse(changed["model"]["changed"])
        self.assertIn(pins["codeRevision"], changed["code"]["compare_url"])

    def test_reviewed_changes_do_not_change_pins_or_require_repeat_audit(self) -> None:
        pins = WATCH.load_pins()
        original = dict(pins)
        audit = WATCH.load_audit()
        report = WATCH.compare_revisions(
            pins, audit["code"]["auditedRevision"], audit["model"]["auditedRevision"], audit
        )
        self.assertTrue(report["updates_available"])
        self.assertEqual(report["status"], "audited")
        self.assertFalse(report["audit_required"])
        self.assertEqual(pins, original)

    def test_new_revision_in_either_source_requires_fresh_audit(self) -> None:
        pins = WATCH.load_pins()
        audit = WATCH.load_audit()
        for name in ("code", "model"):
            with self.subTest(source=name):
                latest = {key: audit[key]["auditedRevision"] for key in ("code", "model")}
                latest[name] = "a" * 40
                report = WATCH.compare_revisions(pins, latest["code"], latest["model"], audit)
                self.assertTrue(report["audit_required"])
                self.assertTrue(report[name]["audit_required"])
                other = "model" if name == "code" else "code"
                self.assertFalse(report[other]["audit_required"])
                if name == "code":
                    self.assertIn(audit[name]["auditedRevision"], report[name]["audit_compare_url"])

    def test_acknowledgement_cannot_suppress_changes_for_other_repository_or_pin(self) -> None:
        pins = WATCH.load_pins()
        audit = WATCH.load_audit()
        for name in ("code", "model"):
            for key, value in (("repository", "other/repo"), ("pinnedRevision", "a" * 40)):
                with self.subTest(source=name, field=key):
                    stale = copy.deepcopy(audit)
                    stale[name][key] = value
                    report = WATCH.compare_revisions(
                        pins, audit["code"]["auditedRevision"], audit["model"]["auditedRevision"], stale
                    )
                    self.assertTrue(report[name]["audit_required"])
                    self.assertIsNone(report[name]["audited"])

    def test_return_to_runtime_pins_requires_no_audit(self) -> None:
        pins = WATCH.load_pins()
        report = WATCH.compare_revisions(
            pins, pins["codeRevision"], pins["modelRevision"], WATCH.load_audit()
        )
        self.assertEqual(report["status"], "current")
        self.assertFalse(report["audit_required"])

    def test_missing_baseline_does_not_suppress_notifications(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            audit = WATCH.load_audit(Path(raw) / "absent.json")
        pins = WATCH.load_pins()
        report = WATCH.compare_revisions(pins, "a" * 40, "b" * 40, audit)
        self.assertTrue(report["audit_required"])

    def test_malformed_baseline_fails_loudly(self) -> None:
        for key, value in (("auditedRevision", "main"), ("decision", "automatic_upgrade"), ("evidence", "")):
            with self.subTest(field=key), tempfile.TemporaryDirectory() as raw:
                audit = WATCH.load_audit()
                audit["code"][key] = value
                path = Path(raw) / "audit.json"
                path.write_text(json.dumps(audit), encoding="utf-8")
                with self.assertRaises(ValueError):
                    WATCH.load_audit(path)

    def test_fail_on_update_still_reports_pins_differ_after_audit(self) -> None:
        audit = WATCH.load_audit()
        argv = ["watch", "--github-sha", audit["code"]["auditedRevision"],
                "--model-sha", audit["model"]["auditedRevision"], "--fail-on-update"]
        with patch.object(sys, "argv", argv), patch.object(sys, "stdout", io.StringIO()) as output:
            self.assertEqual(WATCH.main(), 2)
            report = json.loads(output.getvalue())
        self.assertTrue(report["updates_available"])
        self.assertFalse(report["audit_required"])

    def test_cli_missing_audit_file_falls_back_to_unreviewed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            argv = ["watch", "--audit", str(Path(raw) / "absent.json"),
                    "--github-sha", "a" * 40, "--model-sha", "b" * 40]
            with patch.object(sys, "argv", argv), patch.object(sys, "stdout", io.StringIO()) as output:
                self.assertEqual(WATCH.main(), 0)
                report = json.loads(output.getvalue())
        self.assertTrue(report["audit_required"])


if __name__ == "__main__":
    unittest.main()
