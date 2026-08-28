# Model and isolated runtime

Creates the worker handle shared by all FireRedAudio nodes without changing ComfyUI's Python or Transformers packages.

## How to use

1. Place models under `ComfyUI/models/TTS/FireRedAudio`, or select a custom model directory.
2. Use `auto_safe` first on a normal single GPU; use `off` as the troubleshooting baseline.
3. Benchmark DeepSpeed, FLA/Liger, and Torch Compile before selecting them manually.

## Important

Lite supports recognition and understanding only. Generation, voice design, and editing require Full.
