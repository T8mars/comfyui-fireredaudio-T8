# Third-party notices

## FireRedAudio

- Project: https://github.com/FireRedTeam/FireRedAudio
- Model: https://huggingface.co/FireRedTeam/FireRedAudio
- Pinned code revision: `88b826378023eb9a49b297214568398a300e5c32`
- Pinned model revision: `3abf43d7539a2fc05991b9f89295c892f0a034f0`
- License: Apache License 2.0. The complete license text is retained in `LICENSE`.

FireRedAudio acknowledges Qwen3.5, Whisper-large-v3, x-transformers, and vocos. Their packages and binary distributions retain their respective metadata and license files in the packaged Python environment.

## Distribution runtime

The portable build may include CPython, PyTorch, Torchaudio, TorchCodec, Transformers, Electron, and FFmpeg components. The release builder must generate a software bill of materials from the exact packaged environment and copy all bundled license files into `resources/licenses`. Optional acceleration wheels are distributed only after their exact Python/Torch/CUDA ABI and license have been audited.

The isolated Windows runtime can install FlashAttention 2.8.3, DeepSpeed 0.17.5, Triton-Windows, Flash Linear Attention and Liger Kernel from the exact prebuilt wheel URLs and SHA-256 values in `manifests/desktop_acceleration_manifest.json`. It does not compile FlashAttention or DeepSpeed from source and does not install these packages into the ComfyUI host.
