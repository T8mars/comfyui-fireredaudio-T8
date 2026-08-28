"""ComfyUI V3 nodes for FireRedAudio, by T8star-Aix."""

if __package__:
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
else:
    async def comfy_entrypoint():
        raise RuntimeError(
            "Load comfyui-fireredaudio-T8 as a ComfyUI package, not as a standalone __init__.py module."
        )

WEB_DIRECTORY = "./web"
__version__ = "0.17.0"
__all__ = ["WEB_DIRECTORY", "comfy_entrypoint", "__version__"]
