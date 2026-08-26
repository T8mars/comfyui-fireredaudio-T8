from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import sys
import threading
import time
import uuid
from typing import Any

from comfy_api.latest import ComfyExtension, io

from .runtime.audio_adapter import audio_to_wav, output_wav_path, wav_to_audio
from .runtime.model_discovery import (
    MISSING_MODEL_OPTION,
    fingerprint,
    manifest,
    model_options,
    register_model_paths,
    resolve_model,
    validate_sizes,
)
from .runtime.types import GenerationSettings, RuntimeHandle
from .runtime.worker_manager import WORKER_MANAGER

LOGGER = logging.getLogger(__name__)
CATEGORY = "T8star-Aix/Audio/FireRedAudio"
ModelType = io.Custom("T8_FIREREDAUDIO_MODEL")
SettingsType = io.Custom("T8_FIREREDAUDIO_SETTINGS")


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _client(handle: RuntimeHandle):
    return WORKER_MANAGER.client_for(handle)


def _settings(value: GenerationSettings | None) -> GenerationSettings:
    return value or GenerationSettings()


def _infer(handle: RuntimeHandle, request: dict[str, Any]) -> dict[str, Any]:
    """Run inference while forwarding Worker progress and ComfyUI interruption."""
    client = _client(handle)
    task_id = str(request.get("task_id") or uuid.uuid4())
    request["task_id"] = task_id
    result_box: dict[str, Any] = {}
    error_box: list[BaseException] = []

    def run() -> None:
        try:
            result_box.update(client.infer(request))
        except BaseException as exc:
            error_box.append(exc)

    thread = threading.Thread(target=run, name=f"fireredaudio-{request['task']}", daemon=True)
    thread.start()
    progress_bar = None
    try:
        from comfy.utils import ProgressBar

        progress_bar = ProgressBar(100)
    except Exception:
        pass
    while thread.is_alive():
        thread.join(timeout=0.5)
        try:
            health = client.health()
            status = health.get("status", {})
            if progress_bar is not None:
                progress_bar.update_absolute(
                    int(max(0.0, min(1.0, float(status.get("progress", 0.0)))) * 100),
                    100,
                )
        except Exception:
            pass
        try:
            from comfy.model_management import throw_exception_if_processing_interrupted

            throw_exception_if_processing_interrupted()
        except ImportError:
            pass
        except BaseException:
            try:
                client.cancel(task_id)
            finally:
                thread.join(timeout=3.0)
            raise
    if error_box:
        raise error_box[0]
    if progress_bar is not None:
        progress_bar.update_absolute(100, 100)
    return result_box


def _base_request(
    handle: RuntimeHandle, task: str, settings: GenerationSettings | None = None
) -> dict[str, Any]:
    request = {
        "task": task,
        "model_root": handle.model_root,
        "device": handle.device,
        "memory_mode": handle.memory_mode,
        "release_after": handle.release_after,
    }
    if settings is not None:
        request.update(settings.to_dict())
    return request


class T8FireRedAudioModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        options = model_options()
        return io.Schema(
            node_id="T8_FireRedAudio_ModelLoader",
            display_name="FireRedAudio 模型/隔离运行时 · T8star-Aix",
            category=CATEGORY,
            search_aliases=["FireRedAudio", "T8star-Aix", "audio language model"],
            description="发现外置模型并连接独立 Python 3.10 / Transformers 5.8 Worker；不会修改 ComfyUI 环境。",
            inputs=[
                io.Combo.Input("model_name", display_name="模型", options=options, default=options[0]),
                io.String.Input("custom_model_path", display_name="自定义模型根目录", default="", optional=True),
                io.Combo.Input("device", display_name="推理设备", options=["cuda:0", "cuda:1", "cpu"], default="cuda:0"),
                io.Combo.Input("memory_mode", display_name="显存模式", options=["auto", "full_gpu", "sequential", "decoder_cpu"], default="auto", tooltip="auto 在低于 36GB 显存时让主模型与解码器顺序上卡。"),
                io.Combo.Input("profile", display_name="校验配置", options=["full", "lite"], default="full"),
                io.Combo.Input("worker_mode", display_name="Worker 模式", options=["managed", "external"], default="managed"),
                io.String.Input("runtime_python", display_name="隔离 Python.exe", default="", optional=True, advanced=True),
                io.String.Input("worker_url", display_name="外部 Worker URL", default="", optional=True, advanced=True),
                io.String.Input("worker_token", display_name="外部 Worker Token", default="", optional=True, advanced=True),
                io.Boolean.Input("verify_hashes", display_name="完整 SHA-256 校验", default=False, advanced=True),
                io.Boolean.Input("release_after_run", display_name="每次任务后释放模型", default=False, advanced=True),
            ],
            outputs=[
                ModelType.Output("model", display_name="FireRedAudio 运行时"),
                io.String.Output("model_info", display_name="模型信息"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls, model_name: str, custom_model_path: str = "", **kwargs
    ) -> str:
        try:
            return fingerprint(resolve_model(model_name, custom_model_path))
        except Exception as exc:
            return f"missing:{model_name}:{custom_model_path}:{exc}"

    @classmethod
    def validate_inputs(
        cls,
        model_name: str,
        custom_model_path: str = "",
        worker_mode: str = "managed",
        worker_url: str = "",
        worker_token: str = "",
        **kwargs,
    ) -> bool | str:
        if model_name == MISSING_MODEL_OPTION and not custom_model_path.strip():
            return "未找到 FireRedAudio 模型；请运行 scripts/download_models.py。"
        if worker_mode == "external" and (not worker_url.strip() or not worker_token.strip()):
            return "external 模式必须填写 Worker URL 和 Token。"
        return True

    @classmethod
    def execute(
        cls,
        model_name: str,
        custom_model_path: str,
        device: str,
        memory_mode: str,
        profile: str,
        worker_mode: str,
        runtime_python: str,
        worker_url: str,
        worker_token: str,
        verify_hashes: bool,
        release_after_run: bool,
    ) -> io.NodeOutput:
        root = resolve_model(model_name, custom_model_path)
        quick = validate_sizes(root, profile)
        if not quick["valid"]:
            details = "; ".join(f"{item['path']}:{item['problem']}" for item in quick["issues"][:8])
            raise RuntimeError(f"模型不完整：{details}")
        handle = RuntimeHandle(
            model_root=str(root),
            device=device,
            memory_mode=memory_mode,
            runtime_python=runtime_python.strip() if worker_mode == "managed" else "",
            worker_url=worker_url.strip() if worker_mode == "external" else "",
            worker_token=worker_token.strip() if worker_mode == "external" else "",
            verify_hashes=bool(verify_hashes),
            release_after=bool(release_after_run),
        )
        verification: dict[str, Any] = quick
        if verify_hashes:
            verification = _client(handle).validate(
                {"model_root": str(root), "profile": profile, "verify_hashes": True}
            )
            if not verification.get("valid"):
                raise RuntimeError("模型 SHA-256 校验失败：" + _json(verification.get("issues")))
        definition = manifest()
        info = {
            "model_root": str(root),
            "device": device,
            "memory_mode": memory_mode,
            "profile": profile,
            "worker_mode": worker_mode,
            "host_environment_untouched": True,
            "code_revision": definition["codeRevision"],
            "model_revision": definition["modelRevision"],
            "verification": verification,
        }
        return io.NodeOutput(handle, _json(info))


class T8FireRedAudioGenerationSettings(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_GenerationSettings",
            display_name="FireRedAudio 生成参数 · T8star-Aix",
            category=CATEGORY,
            description="TTS、声音设计和语音编辑共用的官方生成参数。",
            inputs=[
                io.Combo.Input("quality_preset", display_name="质量预设", options=["fast", "balanced", "high_quality", "custom"], default="balanced"),
                io.Int.Input("seed", display_name="Seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("max_new_audio_steps", display_name="最大音频步数", default=750, min=6, max=3000),
                io.Int.Input("min_new_audio_steps", display_name="最小音频步数", default=6, min=1, max=750, advanced=True),
                io.Int.Input("max_new_text_tokens", display_name="最大文本 Token", default=512, min=1, max=4096, advanced=True),
                io.Int.Input("n_timesteps", display_name="扩散步数", default=10, min=1, max=100),
                io.Float.Input("inference_cfg", display_name="CFG", default=2.0, min=0.0, max=10.0, step=0.1),
            ],
            outputs=[SettingsType.Output("settings", display_name="生成参数")],
        )

    @classmethod
    def execute(cls, quality_preset: str, seed: int, max_new_audio_steps: int, min_new_audio_steps: int, max_new_text_tokens: int, n_timesteps: int, inference_cfg: float) -> io.NodeOutput:
        return io.NodeOutput(GenerationSettings(quality_preset, seed, max_new_audio_steps, min_new_audio_steps, max_new_text_tokens, n_timesteps, inference_cfg))


class T8FireRedAudioASR(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_ASR",
            display_name="FireRedAudio 语音识别 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="使用 FireRedAudio 进行确定性多语言 ASR。",
            inputs=[
                ModelType.Input("model"),
                io.Audio.Input("audio", display_name="输入音频"),
                io.String.Input("prompt", display_name="提示词", default="Transcribe speech to text.", multiline=True),
                io.Int.Input("max_new_tokens", display_name="最大新 Token", default=300, min=1, max=4096),
            ],
            outputs=[io.String.Output("transcript", display_name="识别文本"), io.String.Output("report", display_name="运行报告")],
        )

    @classmethod
    def execute(cls, model: RuntimeHandle, audio: dict, prompt: str, max_new_tokens: int) -> io.NodeOutput:
        path = audio_to_wav(audio, "asr")
        request = _base_request(model, "asr")
        request.update({"audio_path": str(path), "prompt": prompt, "max_new_tokens": max_new_tokens})
        result = _infer(model, request)
        return io.NodeOutput(result.get("answer", ""), _json(result))


class T8FireRedAudioLongASR(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_LongASR",
            display_name="FireRedAudio 长音频分段转写 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="固定窗口分段转写长录音，输出全文、SRT 和带时间戳的 JSON。",
            inputs=[
                ModelType.Input("model"),
                io.Audio.Input("audio", display_name="长音频"),
                io.String.Input("prompt", display_name="提示词", default="Transcribe speech to text.", multiline=True),
                io.Float.Input("chunk_seconds", display_name="每段秒数", default=30.0, min=5.0, max=300.0, step=1.0),
                io.Float.Input("overlap_seconds", display_name="重叠秒数", default=1.0, min=0.0, max=9.0, step=0.5),
                io.Int.Input("max_new_tokens", display_name="每段最大新 Token", default=300, min=1, max=4096),
            ],
            outputs=[
                io.String.Output("transcript", display_name="完整转写"),
                io.String.Output("srt", display_name="SRT 字幕"),
                io.String.Output("segments_json", display_name="时间戳 JSON"),
                io.String.Output("report", display_name="运行报告"),
            ],
        )

    @classmethod
    def execute(cls, model: RuntimeHandle, audio: dict, prompt: str, chunk_seconds: float, overlap_seconds: float, max_new_tokens: int) -> io.NodeOutput:
        request = _base_request(model, "long_asr")
        request.update({
            "audio_path": str(audio_to_wav(audio, "long-asr")),
            "prompt": prompt,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "max_new_tokens": max_new_tokens,
        })
        result = _infer(model, request)
        return io.NodeOutput(
            result.get("answer", ""),
            result.get("srt", ""),
            _json(result.get("segments", [])),
            _json(result),
        )


class T8FireRedAudioUnderstand(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_Understand",
            display_name="FireRedAudio 音频理解 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="对一个或两个音频提问，可选输出思考过程。",
            inputs=[
                ModelType.Input("model"),
                io.Audio.Input("audio", display_name="音频 1"),
                io.String.Input("prompt", display_name="问题", multiline=True, default="请总结音频内容。", dynamic_prompts=True),
                io.Boolean.Input("enable_thinking", display_name="输出思考过程", default=False),
                io.Int.Input("max_new_tokens", display_name="最大新 Token", default=1024, min=1, max=10240),
                io.Audio.Input("audio_2", display_name="可选音频 2", optional=True),
            ],
            outputs=[io.String.Output("answer", display_name="回答"), io.String.Output("reasoning", display_name="思考过程"), io.String.Output("report", display_name="运行报告")],
        )

    @classmethod
    def execute(cls, model: RuntimeHandle, audio: dict, prompt: str, enable_thinking: bool, max_new_tokens: int, audio_2: dict | None = None) -> io.NodeOutput:
        paths = [str(audio_to_wav(audio, "understand-1"))]
        if audio_2 is not None:
            paths.append(str(audio_to_wav(audio_2, "understand-2")))
        request = _base_request(model, "understand")
        request.update({"audio_paths": paths, "prompt": prompt, "enable_thinking": enable_thinking, "max_new_tokens": max_new_tokens})
        result = _infer(model, request)
        return io.NodeOutput(result.get("answer", ""), result.get("reasoning") or "", _json(result))


class T8FireRedAudioTTS(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_TTS",
            display_name="FireRedAudio 零样本声音克隆 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="参考音频和逐字稿驱动的中英文零样本 TTS，输出 24kHz ComfyUI AUDIO。",
            inputs=[
                ModelType.Input("model"),
                io.Audio.Input("prompt_audio", display_name="参考音频"),
                io.String.Input("prompt_text", display_name="参考音频逐字稿", multiline=True, dynamic_prompts=True),
                io.String.Input("target_text", display_name="目标文本", multiline=True, dynamic_prompts=True),
                io.Combo.Input("language", display_name="语言", options=["zh", "en"], default="zh"),
                SettingsType.Input("settings", display_name="生成参数", optional=True),
            ],
            outputs=[io.Audio.Output("audio", display_name="生成音频"), io.String.Output("report", display_name="运行报告")],
        )

    @classmethod
    def validate_inputs(cls, prompt_text: str, target_text: str, **kwargs) -> bool | str:
        if not prompt_text.strip() or not target_text.strip():
            return "参考逐字稿和目标文本不能为空。"
        return True

    @classmethod
    def execute(cls, model: RuntimeHandle, prompt_audio: dict, prompt_text: str, target_text: str, language: str, settings: GenerationSettings | None = None) -> io.NodeOutput:
        config = _settings(settings)
        output = output_wav_path("tts")
        request = _base_request(model, "tts", config)
        request.update({"prompt_audio": str(audio_to_wav(prompt_audio, "tts-reference")), "prompt_text": prompt_text, "target_text": target_text, "language": language, "output_path": str(output)})
        result = _infer(model, request)
        return io.NodeOutput(wav_to_audio(result["output_path"]), _json(result))


class T8FireRedAudioVoiceDesign(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_VoiceDesign",
            display_name="FireRedAudio 声音设计 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="根据中英文自然语言音色描述合成声音。",
            inputs=[
                ModelType.Input("model"),
                io.String.Input("instruction", display_name="音色描述", multiline=True, dynamic_prompts=True),
                io.String.Input("text", display_name="朗读文本", multiline=True, dynamic_prompts=True),
                SettingsType.Input("settings", display_name="生成参数", optional=True),
            ],
            outputs=[io.Audio.Output("audio", display_name="生成音频"), io.String.Output("report", display_name="运行报告")],
        )

    @classmethod
    def execute(cls, model: RuntimeHandle, instruction: str, text: str, settings: GenerationSettings | None = None) -> io.NodeOutput:
        output = output_wav_path("voice-design")
        request = _base_request(model, "voice_design", _settings(settings))
        request.update({"instruction": instruction, "text": text, "output_path": str(output)})
        result = _infer(model, request)
        return io.NodeOutput(wav_to_audio(result["output_path"]), _json(result))


class T8FireRedAudioSpeechEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_SpeechEdit",
            display_name="FireRedAudio 语音编辑 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="语义插入/删除/替换，或按官方模板调整音高、速度和音量。",
            inputs=[
                ModelType.Input("model"),
                io.Audio.Input("audio", display_name="待编辑音频"),
                io.String.Input("instruction", display_name="编辑指令", multiline=True, dynamic_prompts=True),
                io.Combo.Input("edit_type", display_name="编辑类型", options=["semantic", "acoustic"], default="semantic"),
                SettingsType.Input("settings", display_name="生成参数", optional=True),
            ],
            outputs=[io.Audio.Output("audio", display_name="编辑后音频"), io.String.Output("edited_text", display_name="语义编辑文本"), io.String.Output("report", display_name="运行报告")],
        )

    @classmethod
    def execute(cls, model: RuntimeHandle, audio: dict, instruction: str, edit_type: str, settings: GenerationSettings | None = None) -> io.NodeOutput:
        output = output_wav_path("speech-edit")
        request = _base_request(model, "edit", _settings(settings))
        request.update({"audio_path": str(audio_to_wav(audio, "edit-input")), "instruction": instruction, "edit_type": edit_type, "output_path": str(output)})
        result = _infer(model, request)
        return io.NodeOutput(wav_to_audio(result["output_path"]), result.get("text", ""), _json(result))


class T8FireRedAudioRuntimeControl(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_RuntimeControl",
            display_name="FireRedAudio 运行时控制 · T8star-Aix",
            category=CATEGORY,
            description="查看隔离 Worker 状态、取消当前任务、释放模型显存或停止托管 Worker。",
            inputs=[
                ModelType.Input("model"),
                io.Combo.Input("action", display_name="操作", options=["status", "cancel_current", "unload_model", "stop_worker"], default="status"),
            ],
            outputs=[io.String.Output("status", display_name="运行时状态")],
        )

    @classmethod
    def execute(cls, model: RuntimeHandle, action: str) -> io.NodeOutput:
        if action == "unload_model":
            result = WORKER_MANAGER.unload(model)
        elif action == "cancel_current":
            result = _client(model).cancel()
        elif action == "stop_worker":
            WORKER_MANAGER.stop()
            result = {"stopped": True}
        else:
            client = _client(model)
            result = {"manager": WORKER_MANAGER.status(), "worker": client.health()}
        return io.NodeOutput(_json(result))


class T8FireRedAudioEnvironment(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_Environment",
            display_name="FireRedAudio 环境诊断 · T8star-Aix",
            category=CATEGORY,
            description="证明宿主 ComfyUI 依赖未被节点覆盖，并显示隔离架构要求。",
            inputs=[],
            outputs=[io.String.Output("report", display_name="环境报告")],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        names = ["torch", "torchaudio", "transformers", "numpy", "comfyui-frontend-package"]
        packages = {}
        for name in names:
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        return io.NodeOutput(_json({
            "python": sys.version,
            "platform": platform.platform(),
            "host_packages": packages,
            "node_host_dependencies": [],
            "isolated_worker_required": {"python": "3.10.x", "torch": "2.11.0+cu128", "transformers": "5.8.0"},
            "manager": WORKER_MANAGER.status(),
        }))


class T8FireRedAudioExtension(ComfyExtension):
    async def on_load(self) -> None:
        register_model_paths()
        LOGGER.info("Loaded comfyui-fireredaudio-T8 (isolated FireRedAudio worker)")

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            T8FireRedAudioModelLoader,
            T8FireRedAudioGenerationSettings,
            T8FireRedAudioASR,
            T8FireRedAudioLongASR,
            T8FireRedAudioUnderstand,
            T8FireRedAudioTTS,
            T8FireRedAudioVoiceDesign,
            T8FireRedAudioSpeechEdit,
            T8FireRedAudioRuntimeControl,
            T8FireRedAudioEnvironment,
        ]


async def comfy_entrypoint() -> T8FireRedAudioExtension:
    return T8FireRedAudioExtension()


__all__ = [name for name in globals() if name.startswith("T8FireRedAudio") or name == "comfy_entrypoint"]
