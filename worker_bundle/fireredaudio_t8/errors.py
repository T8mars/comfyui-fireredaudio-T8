class FireRedAudioT8Error(RuntimeError):
    """Base error for actionable runtime failures."""


class ModelValidationError(FireRedAudioT8Error):
    """The selected model directory is incomplete or corrupt."""


class WorkerProtocolError(FireRedAudioT8Error):
    """A worker request or response is invalid."""


class TaskCancelledError(FireRedAudioT8Error):
    """The active inference task was cancelled by the caller."""
