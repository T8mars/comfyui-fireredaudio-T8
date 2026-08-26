"""Shared runtime used by the FireRedAudio T8 desktop and ComfyUI clients."""

from .constants import CODE_REVISION, MODEL_REVISION, RUNTIME_VERSION

__version__ = RUNTIME_VERSION

__all__ = ["CODE_REVISION", "MODEL_REVISION", "RUNTIME_VERSION", "__version__"]
