from __future__ import annotations

import io
import threading

from runtime.worker_manager import WorkerManager


class FakeProcess:
    stdout = io.StringIO("startup failure detail\n")


def test_log_capture_does_not_wait_for_worker_lifecycle_lock() -> None:
    manager = WorkerManager()
    manager._process = FakeProcess()  # type: ignore[assignment]
    with manager._lock:
        thread = threading.Thread(target=manager._capture_logs)
        thread.start()
        thread.join(timeout=1.0)
        assert not thread.is_alive(), "log capture must not block behind startup polling"
        assert manager._log_lines == ["startup failure detail\n"]
    manager._process = None
