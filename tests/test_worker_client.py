from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fireredaudio_worker_client", ROOT / "runtime" / "worker_client.py"
)
CLIENT_MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = CLIENT_MODULE
SPEC.loader.exec_module(CLIENT_MODULE)


class WorkerClientTests(unittest.TestCase):
    def test_tts_batch_uses_dedicated_route_and_long_timeout(self) -> None:
        client = CLIENT_MODULE.WorkerClient("http://127.0.0.1:1234", "token")
        requests = [{"task": "tts", "seed": 42}, {"task": "tts", "seed": 43}]
        with patch.object(CLIENT_MODULE.WorkerClient, "request", return_value={"completed": 2}) as mocked:
            result = client.infer_tts_batch(requests)
        self.assertEqual(result["completed"], 2)
        mocked.assert_called_once_with(
            "v1/infer/tts-batch", {"requests": requests}, timeout=7200.0
        )


if __name__ == "__main__":
    unittest.main()
