# Third-party notices

## FireRedAudio

- Project: https://github.com/FireRedTeam/FireRedAudio
- Upstream model: https://huggingface.co/FireRedTeam/FireRedAudio
- T8star-Aix distribution mirror: https://huggingface.co/t8star/Firered-Audio-Comfy
- Pinned code revision: `88b826378023eb9a49b297214568398a300e5c32`
- Upstream source model revision: `3abf43d7539a2fc05991b9f89295c892f0a034f0`
- Pinned T8star-Aix mirror revision: `ffd450cf336a929f4ceb1eaaff4c7ba4e4af34bb`
- License: Apache License 2.0. The complete license text is retained in `LICENSE`.

FireRedAudio acknowledges Qwen3.5, Whisper-large-v3, x-transformers, and vocos. Their packages and binary distributions retain their respective metadata and license files in the packaged Python environment.

## Quantization runtime

The optional INT8 model formats use TorchAO 0.16.0 and Comfy-Kitchen 0.2.31. TorchAO supplies the stable weight-only INT8 path. Comfy-Kitchen supplies the experimental ConvRot tensor layout and runtime; on Windows the package prefers its Triton/eager backends when the optional CUDA extension is not compatible with the installed display driver. Their license and package metadata remain in the packaged Python environment and generated SBOM.

## Distribution runtime

The Windows portable build includes CPython, PyTorch, Torchaudio, Transformers, Electron, and FFmpeg components. TorchCodec is deliberately omitted on Windows because the pinned release has no `win_amd64` wheel. `manifests/sbom-runtime.json` is generated from the exact packaged Python environment; Python distribution and package license files remain beside their binaries in packaged resources. Optional acceleration wheels are distributed only after their exact Python/Torch/CUDA ABI and license have been audited.

The v0.6 Windows acceleration set includes FlashAttention 2.8.3 from `kingbri1/flash-attention`, DeepSpeed 0.17.5 Windows wheels from `6Morpheus6/deepspeed-windows-wheels`, Triton-Windows, Flash Linear Attention and Liger Kernel. Exact URLs and SHA-256 values are recorded in `manifests/desktop_acceleration_manifest.json`. FlashAttention and DeepSpeed are installed from prebuilt `win_amd64` wheels; the installer does not build them from source.

The desktop package and Windows ComfyUI node archive use the `ffmpeg-static` npm distribution for a self-contained decoder executable. That binary is redistributed under its accompanying GPL/LGPL build terms; its license and source-offer information are retained beside the executable and in these release notices.

The Windows ComfyUI node archive includes the `uv` 0.12.6 executable to create its isolated runtime without installing package-management tools into the ComfyUI host. Its Apache-2.0 and MIT license texts are retained under `tools/licenses/uv`.
