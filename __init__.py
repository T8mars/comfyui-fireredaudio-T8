"""ComfyUI V3 nodes for FireRedAudio, by T8star-Aix."""

try:
    from .nodes_v3 import comfy_entrypoint
except ModuleNotFoundError as exc:
    if not (exc.name or "").startswith("comfy_api"):
        raise
    _COMFY_IMPORT_ERROR = exc

    async def comfy_entrypoint():
        raise RuntimeError(
            "comfy_api.latest is required; install comfyui-fireredaudio-T8 in a current ComfyUI build."
        ) from _COMFY_IMPORT_ERROR

__version__ = "0.4.0"
__all__ = ["comfy_entrypoint", "__version__"]
