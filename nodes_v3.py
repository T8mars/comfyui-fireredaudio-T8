from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import shutil
import statistics
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from comfy_api.latest import ComfyExtension, io

from .runtime.acoustic import acoustic_instruction
from .runtime.audio_adapter import (
    _safe_name,
    _safe_output_dir,
    audio_to_wav,
    output_wav_path,
    saved_audio_ui,
    save_audio_file,
    save_text_file,
    wav_to_audio,
)
from .runtime.model_discovery import (
    MISSING_MODEL_OPTION,
    fingerprint,
    manifest,
    model_options,
    register_model_paths,
    resolve_model,
    validate_sizes,
)
from .runtime.evidence import (
    extract_evidence_ranges,
    parse_structured_json,
    render_evidence_clips,
)
from .runtime.postproduction import prepare_synchronized_ab
from .runtime.production import (
    MANIFEST_VERSION,
    AudioBatch,
    ScriptPlan,
    VoiceBank,
    can_reuse_manifest_item,
    create_voice_bank,
    create_voice_profile,
    line_fingerprint,
    load_manifest,
    load_project_exchange,
    manifest_items_by_id,
    parse_script,
    render_timeline_to_wav,
    stable_digest,
    text_error_rate,
    wav_metrics,
    write_manifest,
)
from .runtime.types import (
    DELIVERY_PRESETS,
    DeliveryPreset,
    GenerationSettings,
    RuntimeHandle,
    delivery_preset,
)
from .runtime.worker_manager import WORKER_MANAGER

LOGGER = logging.getLogger(__name__)
CATEGORY = "T8star-Aix/Audio/FireRedAudio"
ModelType = io.Custom("T8_FIREREDAUDIO_MODEL")
SettingsType = io.Custom("T8_FIREREDAUDIO_SETTINGS")
VoiceProfileType = io.Custom("T8_FIREREDAUDIO_VOICE_PROFILE")
VoiceBankType = io.Custom("T8_FIREREDAUDIO_VOICE_BANK")
ScriptPlanType = io.Custom("T8_FIREREDAUDIO_SCRIPT_PLAN")
AudioBatchType = io.Custom("T8_FIREREDAUDIO_AUDIO_BATCH")
SpeechQAType = io.Custom("T8_FIREREDAUDIO_SPEECH_QA")
DeliveryPresetType = io.Custom("T8_FIREREDAUDIO_DELIVERY_PRESET")

LONG_LOCATE_PROMPTS = {
    "timeline_summary": (
        "Analyze the full recording and return a chronological timeline as JSON. "
        "Each item must contain start_time, end_time, topic, and summary. "
        "Use HH:MM:SS timestamps and preserve important transitions."
    ),
    "time_to_content": (
        "Locate what happens at the requested time or interval in the recording. "
        "Return JSON with query, start_time, end_time, transcript_or_event, and evidence."
    ),
    "content_to_time": (
        "Find every time interval in the recording that matches the requested content. "
        "Return a JSON array with start_time, end_time, match, and confidence_reason."
    ),
    "structured_summary": (
        "Summarize the complete recording as JSON with title, duration_context, speakers, "
        "topics, key_events, decisions, and action_items. Add timestamps where possible."
    ),
}


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _set_official_progress(value: float) -> bool:
    try:
        from comfy_api.latest import ComfyAPISync

        ComfyAPISync().execution.set_progress(
            max(0.0, min(1.0, float(value))), 1.0
        )
        return True
    except Exception:
        return False


def _autogrow_values(values: dict) -> list[Any]:
    def order(item: tuple[str, Any]) -> tuple[int, str]:
        key = item[0]
        suffix = key.rsplit("_", 1)[-1]
        return (int(suffix) if suffix.isdigit() else 10_000, key)

    return [value for _key, value in sorted(values.items(), key=order) if value is not None]


def _client(handle: RuntimeHandle):
    return WORKER_MANAGER.client_for(handle)


def _settings(value: GenerationSettings | None) -> GenerationSettings:
    return value or GenerationSettings()


def _audio_batch_state(value: AudioBatch) -> list[dict[str, Any]]:
    state: list[dict[str, Any]] = []
    if not isinstance(value, AudioBatch):
        return state
    for item in value.items:
        path = Path(str(item.get("output_path") or ""))
        file_state: dict[str, Any] = {"exists": path.is_file()}
        if file_state["exists"]:
            stat = path.stat()
            file_state.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        state.append(
            {
                "line_id": item.get("line_id"),
                "fingerprint": item.get("fingerprint"),
                "status": item.get("status"),
                "output_path": str(path),
                "file": file_state,
            }
        )
    return state


def _is_processing_interrupt(exc: BaseException) -> bool:
    try:
        from comfy.model_management import InterruptProcessingException

        return isinstance(exc, InterruptProcessingException)
    except ImportError:
        return False


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
            official = _set_official_progress(float(status.get("progress", 0.0)))
            if progress_bar is not None and not official:
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
    _set_official_progress(1.0)
    return result_box


def _infer_tts_batch(
    handle: RuntimeHandle, requests: list[dict[str, Any]]
) -> dict[str, Any]:
    """Run the Worker's latent-first TTS batch while forwarding ComfyUI cancellation."""
    client = _client(handle)
    result_box: dict[str, Any] = {}
    error_box: list[BaseException] = []

    def run() -> None:
        try:
            result_box.update(client.infer_tts_batch(requests))
        except BaseException as exc:
            error_box.append(exc)

    thread = threading.Thread(target=run, name="fireredaudio-tts-audition", daemon=True)
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
            status = client.health().get("status", {})
            progress = float(status.get("progress", 0.0))
            official = _set_official_progress(progress)
            if progress_bar is not None and not official:
                progress_bar.update_absolute(int(max(0.0, min(1.0, progress)) * 100), 100)
        except Exception:
            pass
        try:
            from comfy.model_management import throw_exception_if_processing_interrupted

            throw_exception_if_processing_interrupted()
        except ImportError:
            pass
        except BaseException:
            try:
                client.cancel()
            finally:
                thread.join(timeout=3.0)
            raise
    if error_box:
        raise error_box[0]
    if progress_bar is not None:
        progress_bar.update_absolute(100, 100)
    _set_official_progress(1.0)
    return result_box


def _transcribe_reference(handle: RuntimeHandle, audio_path: str | Path) -> tuple[str, dict[str, Any]]:
    request = _base_request(handle, "asr")
    request.update(
        {
            "audio_path": str(audio_path),
            "prompt": "Transcribe speech to text.",
            "max_new_tokens": 512,
            "release_after": False,
        }
    )
    result = _infer(handle, request)
    transcript = str(result.get("answer") or "").strip()
    if not transcript:
        raise RuntimeError("参考音频 ASR 未返回逐字稿，请检查录音是否包含清晰语音")
    return transcript, result


def _base_request(
    handle: RuntimeHandle, task: str, settings: GenerationSettings | None = None
) -> dict[str, Any]:
    request = {
        "task": task,
        "model_root": handle.model_root,
        "device": handle.device,
        "memory_mode": handle.memory_mode,
        "acceleration_mode": handle.acceleration_mode,
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
                io.Combo.Input("device", display_name="推理设备", options=["auto", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"], default="auto", tooltip="auto 由隔离 Worker 选择第一张可用 NVIDIA GPU；运行时状态会显示真实显存。"),
                io.Combo.Input("memory_mode", display_name="显存模式", options=["auto", "full_gpu", "sequential", "decoder_cpu"], default="auto", tooltip="auto 在低于 36GB 显存时让主模型与解码器顺序上卡。"),
                io.Combo.Input("acceleration_mode", display_name="加速模式", options=["auto_safe", "off", "flash_attention", "deepspeed", "fla_liger", "torch_compile"], default="auto_safe", tooltip="auto_safe 默认使用预编译 FlashAttention；DeepSpeed 为单卡 BF16 实验模式。失败会显式回退，且不会修改 ComfyUI 宿主环境。"),
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
        acceleration_mode: str,
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
            acceleration_mode=acceleration_mode,
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
            "acceleration_mode": acceleration_mode,
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


class T8FireRedAudioDeliveryPreset(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_DeliveryPreset",
            display_name="FireRedAudio 交付预设 · T8star-Aix",
            category=CATEGORY,
            description="一次设置有声书、播客或视频对白的排列、交叉淡化、EBU R128 响度、True Peak、采样率和保存格式。",
            inputs=[
                io.Combo.Input(
                    "preset_name",
                    display_name="交付场景",
                    options=list(DELIVERY_PRESETS),
                    default="audiobook",
                )
            ],
            outputs=[
                DeliveryPresetType.Output("delivery_preset", display_name="交付预设"),
                io.String.Output("preset_report", display_name="预设详情"),
            ],
        )

    @classmethod
    def execute(cls, preset_name: str) -> io.NodeOutput:
        preset = delivery_preset(preset_name)
        return io.NodeOutput(preset, _json(preset.to_dict()))


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


class T8FireRedAudioReferenceTranscript(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_ReferenceTranscript",
            display_name="FireRedAudio 参考音频 ASR 逐字稿 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="一键转写声音克隆参考音频，并原样传递音频；把逐字稿输出连接到 TTS 或音色档案即可自动带入。",
            inputs=[
                ModelType.Input("model"),
                io.Audio.Input("reference_audio", display_name="参考音频"),
                io.Int.Input("max_new_tokens", display_name="最大新 Token", default=512, min=32, max=4096),
            ],
            outputs=[
                io.Audio.Output("reference_audio", display_name="参考音频"),
                io.String.Output("transcript", display_name="自动逐字稿"),
                io.String.Output("report", display_name="ASR 报告"),
            ],
        )

    @classmethod
    def execute(
        cls, model: RuntimeHandle, reference_audio: dict, max_new_tokens: int
    ) -> io.NodeOutput:
        path = audio_to_wav(reference_audio, "reference-transcript")
        request = _base_request(model, "asr")
        request.update(
            {
                "audio_path": str(path),
                "prompt": "Transcribe speech to text.",
                "max_new_tokens": max_new_tokens,
            }
        )
        result = _infer(model, request)
        transcript = str(result.get("answer") or "").strip()
        if not transcript:
            raise RuntimeError("参考音频 ASR 未返回逐字稿")
        return io.NodeOutput(reference_audio, transcript, _json(result))


class T8FireRedAudioLongASR(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_LongASR",
            display_name="FireRedAudio 长音频分段转写 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="静音附近智能切段、重叠文本去重，输出分段级近似时间的 SRT/VTT/JSONL。",
            inputs=[
                ModelType.Input("model"),
                io.Audio.Input("audio", display_name="长音频"),
                io.String.Input("prompt", display_name="提示词", default="Transcribe speech to text.", multiline=True),
                io.Float.Input("chunk_seconds", display_name="每段秒数", default=30.0, min=5.0, max=300.0, step=1.0),
                io.Float.Input("overlap_seconds", display_name="重叠秒数", default=1.0, min=0.0, max=9.0, step=0.5),
                io.Int.Input("max_new_tokens", display_name="每段最大新 Token", default=300, min=1, max=4096),
                io.Float.Input("silence_search_seconds", display_name="静音切点搜索范围", default=1.5, min=0.0, max=5.0, step=0.25, advanced=True),
            ],
            outputs=[
                io.String.Output("transcript", display_name="完整转写"),
                io.String.Output("srt", display_name="SRT 字幕"),
                io.String.Output("segments_json", display_name="时间戳 JSON"),
                io.String.Output("report", display_name="运行报告"),
                io.String.Output("vtt", display_name="WebVTT 字幕"),
                io.String.Output("jsonl", display_name="JSONL 分段"),
            ],
        )

    @classmethod
    def execute(cls, model: RuntimeHandle, audio: dict, prompt: str, chunk_seconds: float, overlap_seconds: float, max_new_tokens: int, silence_search_seconds: float = 1.5) -> io.NodeOutput:
        request = _base_request(model, "long_asr")
        request.update({
            "audio_path": str(audio_to_wav(audio, "long-asr")),
            "prompt": prompt,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "max_new_tokens": max_new_tokens,
            "silence_search_seconds": silence_search_seconds,
        })
        result = _infer(model, request)
        return io.NodeOutput(
            result.get("answer", ""),
            result.get("srt", ""),
            _json(result.get("segments", [])),
            _json(result),
            result.get("vtt", ""),
            result.get("jsonl", ""),
        )


class T8FireRedAudioLongLocator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_LongLocator",
            display_name="FireRedAudio 长音频时间定位 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="调用 FireRedAudio 原生长录音理解能力，完成时间线、时间找内容、内容找时间和结构化摘要。",
            inputs=[
                ModelType.Input("model"),
                io.Audio.Input("audio", display_name="长音频"),
                io.Combo.Input("mode", display_name="定位模式", options=list(LONG_LOCATE_PROMPTS), default="timeline_summary"),
                io.String.Input("query", display_name="时间、内容或补充要求", default="", multiline=True, dynamic_prompts=True),
                io.Boolean.Input("enable_thinking", display_name="输出思考过程", default=True),
                io.Int.Input("max_new_tokens", display_name="最大新 Token", default=2048, min=128, max=10240),
            ],
            outputs=[
                io.String.Output("answer", display_name="定位结果"),
                io.String.Output("structured_json", display_name="结构化 JSON"),
                io.String.Output("reasoning", display_name="思考过程"),
                io.String.Output("report", display_name="运行报告"),
            ],
        )

    @classmethod
    def execute(cls, model: RuntimeHandle, audio: dict, mode: str, query: str, enable_thinking: bool, max_new_tokens: int) -> io.NodeOutput:
        base = LONG_LOCATE_PROMPTS.get(mode, LONG_LOCATE_PROMPTS["timeline_summary"])
        prompt = base + (f"\nUser query: {query.strip()}" if query.strip() else "")
        request = _base_request(model, "long_locate")
        request.update({
            "audio_path": str(audio_to_wav(audio, "long-locator")),
            "mode": mode,
            "prompt": prompt,
            "enable_thinking": enable_thinking,
            "max_new_tokens": max_new_tokens,
        })
        result = _infer(model, request)
        return io.NodeOutput(
            result.get("answer", ""),
            _json(result.get("structured")),
            result.get("reasoning") or "",
            _json(result),
        )


class T8FireRedAudioEvidenceClips(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_EvidenceClips",
            display_name="FireRedAudio 定位证据片段/剪辑清单 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="把长音频定位 JSON 转成可试听证据片段、时间范围和可继续进入批量制作的 AudioBatch。",
            inputs=[
                io.Audio.Input("source_audio", display_name="原始长音频"),
                io.String.Input("structured_json", display_name="定位结构化 JSON", multiline=True, force_input=True),
                io.String.Input("project_name", display_name="项目名", default="evidence-project"),
                io.String.Input("subfolder", display_name="输出子目录", default="fireredaudio/evidence"),
                io.Float.Input("padding_seconds", display_name="片段前后留白", default=0.25, min=0.0, max=10.0, step=0.05),
                io.Float.Input("default_clip_seconds", display_name="仅有时间点时的默认长度", default=8.0, min=0.5, max=120.0, step=0.5),
                io.Int.Input("max_clips", display_name="最多片段", default=20, min=1, max=100),
            ],
            outputs=[
                io.Audio.Output("first_clip", display_name="首个证据片段"),
                AudioBatchType.Output("evidence_batch", display_name="证据片段批次"),
                io.String.Output("cut_list_json", display_name="剪辑清单 JSON"),
                io.String.Output("manifest_path", display_name="Manifest 路径"),
            ],
        )

    @classmethod
    def execute(
        cls,
        source_audio: dict,
        structured_json: str,
        project_name: str,
        subfolder: str,
        padding_seconds: float,
        default_clip_seconds: float,
        max_clips: int,
    ) -> io.NodeOutput:
        structured = parse_structured_json(structured_json)
        ranges = extract_evidence_ranges(
            structured,
            default_clip_seconds=default_clip_seconds,
            max_clips=max_clips,
        )
        if not ranges:
            raise ValueError("定位结果中没有可识别的开始/结束时间")
        project = _safe_name(project_name, "evidence-project")
        clean_subfolder = subfolder.rstrip("/\\") or "fireredaudio/evidence"
        project_dir = _safe_output_dir(f"{clean_subfolder}/{project}")
        source_path = audio_to_wav(source_audio, "evidence-source")
        items = render_evidence_clips(
            source_path,
            ranges,
            project_dir,
            filename_prefix=project,
            padding_seconds=padding_seconds,
        )
        manifest_path = project_dir / "manifest.json"
        manifest_payload = {
            "manifest_version": MANIFEST_VERSION,
            "kind": "long_audio_evidence",
            "project_name": project,
            "source_audio": str(source_path),
            "items": items,
        }
        write_manifest(manifest_path, manifest_payload)
        batch = AudioBatch(str(manifest_path), tuple(items))
        return io.NodeOutput(
            wav_to_audio(items[0]["output_path"]),
            batch,
            _json(items),
            str(manifest_path),
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


class T8FireRedAudioMultiUnderstand(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_MultiUnderstand",
            display_name="FireRedAudio 多音频比较理解 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="自动增长 1–8 个音频输入，用于说话人比较、事件对照和多音频问答。",
            inputs=[
                ModelType.Input("model"),
                io.Autogrow.Input(
                    "audios",
                    display_name="音频",
                    template=io.Autogrow.TemplatePrefix(
                        io.Audio.Input("audio"), prefix="audio_", min=1, max=8
                    ),
                ),
                io.String.Input("prompt", display_name="问题", multiline=True, default="Compare these recordings and explain the relevant similarities and differences.", dynamic_prompts=True),
                io.Boolean.Input("enable_thinking", display_name="输出思考过程", default=False),
                io.Int.Input("max_new_tokens", display_name="最大新 Token", default=1024, min=1, max=10240),
            ],
            outputs=[
                io.String.Output("answer", display_name="回答"),
                io.String.Output("reasoning", display_name="思考过程"),
                io.String.Output("report", display_name="运行报告"),
            ],
        )

    @classmethod
    def execute(cls, model: RuntimeHandle, audios: dict, prompt: str, enable_thinking: bool, max_new_tokens: int) -> io.NodeOutput:
        values = _autogrow_values(audios)
        if not values:
            raise ValueError("至少连接一个音频")
        paths = [str(audio_to_wav(value, f"multi-understand-{index}")) for index, value in enumerate(values, 1)]
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
            description="参考音频驱动的中英文零样本 TTS；逐字稿可手工连接，留空时可自动 ASR 后继续生成。",
            inputs=[
                ModelType.Input("model"),
                io.Audio.Input("prompt_audio", display_name="参考音频"),
                io.String.Input("prompt_text", display_name="参考音频逐字稿（可留空自动 ASR）", default="", multiline=True, dynamic_prompts=True),
                io.String.Input("target_text", display_name="目标文本", multiline=True, dynamic_prompts=True),
                io.Combo.Input("language", display_name="语言", options=["zh", "en"], default="zh"),
                io.Boolean.Input("auto_transcribe_reference", display_name="逐字稿为空时自动 ASR", default=True),
                SettingsType.Input("settings", display_name="生成参数", optional=True),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="生成音频"),
                io.String.Output("report", display_name="运行报告"),
                io.String.Output("reference_transcript", display_name="实际参考逐字稿"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        prompt_text: str,
        target_text: str,
        auto_transcribe_reference: bool = True,
        **kwargs,
    ) -> bool | str:
        if not target_text.strip():
            return "目标文本不能为空。"
        if not prompt_text.strip() and not auto_transcribe_reference:
            return "参考逐字稿为空；请填写逐字稿或启用自动 ASR。"
        return True

    @classmethod
    def execute(
        cls,
        model: RuntimeHandle,
        prompt_audio: dict,
        prompt_text: str,
        target_text: str,
        language: str,
        auto_transcribe_reference: bool = True,
        settings: GenerationSettings | None = None,
    ) -> io.NodeOutput:
        config = _settings(settings)
        reference_path = audio_to_wav(prompt_audio, "tts-reference")
        transcript = str(prompt_text or "").strip()
        transcript_report: dict[str, Any] | None = None
        if not transcript:
            if not auto_transcribe_reference:
                raise ValueError("参考逐字稿为空且未启用自动 ASR")
            transcript, transcript_report = _transcribe_reference(model, reference_path)
        output = output_wav_path("tts")
        request = _base_request(model, "tts", config)
        request.update({"prompt_audio": str(reference_path), "prompt_text": transcript, "target_text": target_text, "language": language, "output_path": str(output)})
        result = _infer(model, request)
        result["reference_transcript"] = transcript
        result["reference_transcript_source"] = "automatic_asr" if transcript_report else "user"
        if transcript_report:
            result["reference_asr_performance"] = transcript_report.get("performance")
        return io.NodeOutput(wav_to_audio(result["output_path"]), _json(result), transcript)


class T8FireRedAudioSeedAudition(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_SeedAudition",
            display_name="FireRedAudio 多 Seed 试音/推荐 Take · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="一次生成 2–8 个 Seed 候选；可选 ASR 回读质检，输出推荐 Take、全部持久化候选与审计报告。",
            inputs=[
                ModelType.Input("model"),
                io.Audio.Input("prompt_audio", display_name="参考音频"),
                io.String.Input("prompt_text", display_name="参考逐字稿（可留空自动 ASR）", default="", multiline=True, dynamic_prompts=True),
                io.String.Input("target_text", display_name="目标文本", multiline=True, dynamic_prompts=True),
                io.Combo.Input("language", display_name="语言", options=["zh", "en"], default="zh"),
                io.Int.Input("seed_start", display_name="起始 Seed", default=42, min=0, max=0xFFFFFFFF - 8),
                io.Int.Input("take_count", display_name="候选数量", default=4, min=2, max=8),
                io.Boolean.Input("run_asr_qa", display_name="逐个 ASR 回读并参与推荐", default=True),
                io.String.Input("project_name", display_name="试音项目名", default="seed-audition"),
                io.String.Input("subfolder", display_name="输出子目录", default="fireredaudio/auditions"),
                SettingsType.Input("settings", display_name="生成参数", optional=True),
            ],
            outputs=[
                io.Audio.Output("recommended_audio", display_name="推荐 Take"),
                AudioBatchType.Output("all_takes", display_name="全部候选"),
                io.String.Output("manifest_path", display_name="Manifest 路径"),
                io.String.Output("reference_transcript", display_name="实际参考逐字稿"),
                io.String.Output("audition_report", display_name="试音与推荐报告"),
            ],
        )

    @classmethod
    def validate_inputs(cls, target_text: str, project_name: str, **kwargs) -> bool | str:
        if not target_text.strip():
            return "目标文本不能为空。"
        if not project_name.strip():
            return "试音项目名不能为空。"
        return True

    @classmethod
    def execute(
        cls,
        model: RuntimeHandle,
        prompt_audio: dict,
        prompt_text: str,
        target_text: str,
        language: str,
        seed_start: int,
        take_count: int,
        run_asr_qa: bool,
        project_name: str,
        subfolder: str,
        settings: GenerationSettings | None = None,
    ) -> io.NodeOutput:
        reference_path = audio_to_wav(prompt_audio, "audition-reference")
        transcript = str(prompt_text or "").strip()
        reference_source = "user"
        if not transcript:
            transcript, _asr_report = _transcribe_reference(model, reference_path)
            reference_source = "automatic_asr"
        project = _safe_name(project_name, "seed-audition")
        clean_subfolder = subfolder.rstrip("/\\") or "fireredaudio/auditions"
        project_dir = _safe_output_dir(f"{clean_subfolder}/{project}-{uuid.uuid4().hex[:8]}")
        config = _settings(settings)
        requests: list[dict[str, Any]] = []
        for offset in range(int(take_count)):
            seed = int(seed_start) + offset
            request = _base_request(model, "tts", config)
            request.update(
                {
                    "task_id": f"audition-{seed}-{uuid.uuid4().hex[:8]}",
                    "seed": seed,
                    "prompt_audio": str(reference_path),
                    "prompt_text": transcript,
                    "target_text": target_text,
                    "language": language,
                    "output_path": str(project_dir / f"take-seed-{seed}.wav"),
                }
            )
            requests.append(request)
        batch_result = _infer_tts_batch(model, requests)
        items: list[dict[str, Any]] = []
        for outcome in batch_result.get("outcomes", []):
            if not outcome.get("ok"):
                items.append(
                    {
                        "line_id": f"take-{int(outcome.get('index', 0)) + 1:03d}",
                        "index": int(outcome.get("index", 0)) + 1,
                        "status": "failed",
                        "error": outcome.get("error"),
                    }
                )
                continue
            result = outcome["result"]
            index = int(outcome.get("index", 0))
            path = Path(result["output_path"])
            metrics = wav_metrics(path)
            item: dict[str, Any] = {
                "line_id": f"take-{index + 1:03d}",
                "index": index + 1,
                "status": "complete",
                "output_path": str(path),
                "seed": requests[index]["seed"],
                "metrics": metrics,
                "worker_report": result,
            }
            if run_asr_qa:
                qa_request = _base_request(model, "asr")
                qa_request.update(
                    {
                        "audio_path": str(path),
                        "prompt": "Transcribe speech to text.",
                        "max_new_tokens": 1024,
                        "release_after": False,
                    }
                )
                qa_result = _infer(model, qa_request)
                hypothesis = str(qa_result.get("answer") or "").strip()
                metric_name, error_rate = text_error_rate(target_text, hypothesis, language)
                item["asr_qa"] = {
                    "hypothesis": hypothesis,
                    "metric": metric_name,
                    "error_rate": error_rate,
                }
            items.append(item)
        successful = [item for item in items if item.get("status") == "complete"]
        if not successful:
            raise RuntimeError("多 Seed 试音没有生成任何有效候选：" + _json(items))
        median_duration = statistics.median(
            float(item["metrics"]["duration_seconds"]) for item in successful
        )
        for item in successful:
            metrics = item["metrics"]
            duration_penalty = abs(float(metrics["duration_seconds"]) - median_duration) / max(median_duration, 0.001)
            item["ranking_score"] = round(
                float(item.get("asr_qa", {}).get("error_rate", 0.0)) * 1000.0
                + float(metrics.get("clipping_ratio", 0.0)) * 500.0
                + float(metrics.get("silence_ratio", 0.0)) * 10.0
                + duration_penalty,
                6,
            )
        recommended = min(successful, key=lambda item: (item["ranking_score"], item["seed"]))
        manifest_path = project_dir / "manifest.json"
        manifest_payload = {
            "manifest_version": MANIFEST_VERSION,
            "kind": "seed_audition",
            "project_name": project,
            "reference_transcript": transcript,
            "reference_transcript_source": reference_source,
            "target_text": target_text,
            "language": language,
            "recommended_line_id": recommended["line_id"],
            "items": items,
            "performance": batch_result.get("performance"),
        }
        write_manifest(manifest_path, manifest_payload)
        batch = AudioBatch(str(manifest_path), tuple(items))
        return io.NodeOutput(
            wav_to_audio(recommended["output_path"]),
            batch,
            str(manifest_path),
            transcript,
            _json(manifest_payload),
        )


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


class T8FireRedAudioAcousticEdit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_AcousticEdit",
            display_name="FireRedAudio 参数化声学编辑 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="用安全控件生成上游严格模板，避免手写音高、速度和音量英文指令。",
            inputs=[
                ModelType.Input("model"),
                io.Audio.Input("audio", display_name="待编辑音频"),
                io.Combo.Input("operation", display_name="操作", options=["pitch", "speed", "volume"], default="pitch"),
                io.Int.Input("pitch_steps", display_name="音高步数（不能为 0）", default=3, min=-6, max=6),
                io.Float.Input("speed", display_name="速度", default=1.2, min=0.5, max=2.0, step=0.1),
                io.Float.Input("volume", display_name="音量", default=1.0, min=0.3, max=2.0, step=0.1),
                SettingsType.Input("settings", display_name="生成参数", optional=True),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="编辑后音频"),
                io.String.Output("instruction", display_name="实际指令"),
                io.String.Output("report", display_name="运行报告"),
            ],
        )

    @classmethod
    def execute(cls, model: RuntimeHandle, audio: dict, operation: str, pitch_steps: int, speed: float, volume: float, settings: GenerationSettings | None = None) -> io.NodeOutput:
        instruction = acoustic_instruction(
            operation, pitch_steps=pitch_steps, speed=speed, volume=volume
        )
        output = output_wav_path("acoustic-edit")
        request = _base_request(model, "edit", _settings(settings))
        request.update({
            "audio_path": str(audio_to_wav(audio, "acoustic-edit-input")),
            "instruction": instruction,
            "edit_type": "acoustic",
            "output_path": str(output),
        })
        result = _infer(model, request)
        return io.NodeOutput(wav_to_audio(result["output_path"]), instruction, _json(result))


class T8FireRedAudioReferenceQuality(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_ReferenceQuality",
            display_name="FireRedAudio 参考音频质检 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="检查参考音频时长、采样率、声道、削波、静音比例、响度和直流偏移。",
            inputs=[ModelType.Input("model"), io.Audio.Input("audio", display_name="参考音频")],
            outputs=[io.Audio.Output("audio", display_name="原始音频"), io.String.Output("quality_report", display_name="质检报告 JSON")],
        )

    @classmethod
    def execute(cls, model: RuntimeHandle, audio: dict) -> io.NodeOutput:
        path = audio_to_wav(audio, "reference-quality")
        result = _client(model).analyze_audio(str(path))
        return io.NodeOutput(audio, _json(result))


class T8FireRedAudioPrepareReference(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_PrepareReference",
            display_name="FireRedAudio 参考音频清理副本 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description=(
                "非破坏式生成 24 kHz 单声道参考副本，可裁首尾静音并规范响度。"
                "不会覆盖原音频，也不会执行可能改变音色的降噪、去混响或削波伪修复。"
            ),
            inputs=[
                ModelType.Input("model"),
                io.Audio.Input("audio", display_name="参考音频"),
                io.Boolean.Input("trim_silence", display_name="裁首尾静音", default=True),
                io.Boolean.Input("normalize_loudness", display_name="规范到 -23 LUFS", default=False),
                io.Boolean.Input("speech_highpass", display_name="60 Hz 语音高通", default=True),
            ],
            outputs=[
                io.Audio.Output("clean_audio", display_name="清理副本"),
                io.String.Output("cleanup_report", display_name="清理报告 JSON"),
                io.String.Output("output_path", display_name="副本路径"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: RuntimeHandle,
        audio: dict,
        trim_silence: bool,
        normalize_loudness: bool,
        speech_highpass: bool,
    ) -> io.NodeOutput:
        source = audio_to_wav(audio, "reference-clean-source")
        output = output_wav_path("reference-clean")
        result = _client(model).prepare_reference(
            str(source),
            str(output),
            trim_silence=trim_silence,
            normalize_loudness=normalize_loudness,
            target_lufs=-23.0,
            highpass_hz=60.0 if speech_highpass else None,
        )
        return io.NodeOutput(
            wav_to_audio(result["output_path"]),
            _json(result),
            str(result["output_path"]),
        )


class T8FireRedAudioProjectExchange(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_ProjectExchange",
            display_name="FireRedAudio 桌面项目交换 · T8star-Aix",
            category=CATEGORY,
            description="载入桌面整合包导出的项目 JSON，并还原角色音色库、脚本计划和已采用 take。",
            inputs=[
                io.String.Input(
                    "exchange_path",
                    display_name="项目交换 JSON 路径",
                    multiline=False,
                    default="",
                )
            ],
            outputs=[
                VoiceBankType.Output("voice_bank", display_name="角色音色库"),
                ScriptPlanType.Output("script_plan", display_name="脚本计划"),
                AudioBatchType.Output("audio_batch", display_name="已采用 Take"),
                io.String.Output("report", display_name="载入报告"),
            ],
        )

    @classmethod
    def execute(cls, exchange_path: str) -> io.NodeOutput:
        bank, plan, batch, report = load_project_exchange(exchange_path)
        return io.NodeOutput(bank, plan, batch, _json(report))


class T8FireRedAudioVoiceProfile(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_VoiceProfile",
            display_name="FireRedAudio 音色档案 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="把参考音频、逐字稿、语言和标签封装为可复用音色档案。参考音频使用内容哈希缓存。",
            inputs=[
                io.Audio.Input("audio", display_name="参考音频"),
                io.String.Input("name", display_name="角色/音色名称", default="旁白"),
                io.String.Input("prompt_text", display_name="参考音频逐字稿", multiline=True, dynamic_prompts=True),
                io.Combo.Input("language", display_name="语言", options=["zh", "en"], default="zh"),
                io.String.Input("tags", display_name="标签（逗号分隔）", default="", optional=True),
            ],
            outputs=[
                VoiceProfileType.Output("profile", display_name="音色档案"),
                io.String.Output("profile_json", display_name="档案 JSON"),
            ],
        )

    @classmethod
    def validate_inputs(cls, name: str, prompt_text: str, **kwargs) -> bool | str:
        if not name.strip():
            return "音色名称不能为空。"
        if not prompt_text.strip():
            return "参考音频逐字稿不能为空。"
        return True

    @classmethod
    def execute(
        cls,
        audio: dict,
        name: str,
        prompt_text: str,
        language: str,
        tags: str = "",
    ) -> io.NodeOutput:
        profile = create_voice_profile(
            name,
            audio_to_wav(audio, f"voice-profile-{_safe_name(name, 'voice')}"),
            prompt_text,
            language,
            tags,
        )
        return io.NodeOutput(profile, _json(profile.to_dict()))


class T8FireRedAudioVoiceBank(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_VoiceBank",
            display_name="FireRedAudio 音色库（1–8）· T8star-Aix",
            category=CATEGORY,
            description="聚合 1–8 个唯一命名的音色档案，供角色脚本和批量配音复用。",
            inputs=[
                io.Autogrow.Input(
                    "profiles",
                    display_name="音色档案",
                    template=io.Autogrow.TemplatePrefix(
                        VoiceProfileType.Input("profile"), prefix="profile_", min=1, max=8
                    ),
                ),
            ],
            outputs=[
                VoiceBankType.Output("voice_bank", display_name="音色库"),
                io.String.Output("voice_bank_json", display_name="音色库 JSON"),
            ],
        )

    @classmethod
    def execute(cls, profiles: dict) -> io.NodeOutput:
        bank = create_voice_bank(_autogrow_values(profiles))
        return io.NodeOutput(bank, _json(bank.to_dict()))


class T8FireRedAudioScriptParser(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_ScriptParser",
            display_name="FireRedAudio 角色脚本/SRT 预检 · T8star-Aix",
            category=CATEGORY,
            description=(
                "解析 SRT、角色: 台词、带时间码角色脚本或 JSON；检查角色绑定、空台词、语言和时间范围。"
            ),
            inputs=[
                VoiceBankType.Input("voice_bank", display_name="音色库"),
                io.String.Input("script", display_name="脚本", multiline=True, dynamic_prompts=True),
                io.Combo.Input(
                    "source_format",
                    display_name="脚本格式",
                    options=["auto", "srt", "role_script", "json"],
                    default="auto",
                ),
                io.String.Input("default_speaker", display_name="默认角色", default="", optional=True),
                io.Boolean.Input("strict_validation", display_name="发现错误时终止", default=False),
            ],
            outputs=[
                ScriptPlanType.Output("script_plan", display_name="脚本计划"),
                io.String.Output("normalized_json", display_name="标准化脚本 JSON"),
                io.String.Output("preflight_report", display_name="预检报告"),
            ],
        )

    @classmethod
    def validate_inputs(cls, script: str, **kwargs) -> bool | str:
        return True if script.strip() else "脚本不能为空。"

    @classmethod
    def execute(
        cls,
        voice_bank: VoiceBank,
        script: str,
        source_format: str,
        default_speaker: str,
        strict_validation: bool,
    ) -> io.NodeOutput:
        plan = parse_script(script, source_format, voice_bank, default_speaker)
        report = {
            "valid": plan.valid,
            "format": plan.source_format,
            "line_count": len(plan.lines),
            "error_count": sum(item.get("severity") == "error" for item in plan.issues),
            "warning_count": sum(item.get("severity") == "warning" for item in plan.issues),
            "issues": list(plan.issues),
        }
        if strict_validation and not plan.valid:
            raise ValueError("脚本预检失败：" + _json(report))
        return io.NodeOutput(plan, _json(plan.to_dict()), _json(report))


class T8FireRedAudioBatchDubbing(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_BatchDubbing",
            display_name="FireRedAudio 可恢复批量配音 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description=(
                "按脚本顺序逐条调用隔离 Worker。每条成功后原子写入 manifest；重新执行时只跳过指纹一致且文件存在的条目。"
            ),
            inputs=[
                ModelType.Input("model"),
                ScriptPlanType.Input("script_plan", display_name="脚本计划"),
                VoiceBankType.Input("voice_bank", display_name="音色库"),
                io.String.Input("project_name", display_name="项目名", default="dubbing-project"),
                io.String.Input("subfolder", display_name="输出子目录", default="fireredaudio/projects"),
                io.Boolean.Input("resume", display_name="从 manifest 恢复", default=True),
                io.Boolean.Input("continue_on_error", display_name="单条失败后继续", default=True),
                SettingsType.Input("settings", display_name="生成参数", optional=True),
            ],
            outputs=[
                AudioBatchType.Output("audio_batch", display_name="批量音频"),
                io.String.Output("manifest_path", display_name="Manifest 路径"),
                io.String.Output("batch_report", display_name="批量报告"),
            ],
        )

    @classmethod
    def validate_inputs(cls, project_name: str, **kwargs) -> bool | str:
        return True if project_name.strip() else "项目名不能为空。"

    @classmethod
    def fingerprint_inputs(
        cls,
        model: RuntimeHandle,
        script_plan: ScriptPlan,
        voice_bank: VoiceBank,
        project_name: str,
        subfolder: str,
        resume: bool,
        continue_on_error: bool,
        settings: GenerationSettings | None = None,
        **kwargs,
    ) -> str:
        try:
            project = _safe_name(project_name, "dubbing-project")
            clean_subfolder = subfolder.rstrip("/\\") or "fireredaudio/projects"
            manifest_path = _safe_output_dir(f"{clean_subfolder}/{project}") / "manifest.json"
            manifest_state: Any = "missing"
            if manifest_path.is_file():
                loaded = load_manifest(manifest_path)
                manifest_state = [
                    {
                        "line_id": item.get("line_id"),
                        "fingerprint": item.get("fingerprint"),
                        "status": item.get("status"),
                        "output_path": item.get("output_path"),
                        "file_exists": Path(str(item.get("output_path") or "")).is_file(),
                    }
                    for item in (loaded or {}).get("items", [])
                    if isinstance(item, dict)
                ]
            return stable_digest(
                {
                    "model": model.to_dict(),
                    "script": script_plan.to_dict(),
                    "voice_bank": voice_bank.to_dict(),
                    "settings": _settings(settings).to_dict(),
                    "project": project,
                    "subfolder": clean_subfolder,
                    "resume": resume,
                    "continue_on_error": continue_on_error,
                    "manifest_state": manifest_state,
                }
            )
        except Exception as exc:
            return f"batch-invalid:{type(exc).__name__}:{exc}"

    @classmethod
    def execute(
        cls,
        model: RuntimeHandle,
        script_plan: ScriptPlan,
        voice_bank: VoiceBank,
        project_name: str,
        subfolder: str,
        resume: bool,
        continue_on_error: bool,
        settings: GenerationSettings | None = None,
    ) -> io.NodeOutput:
        if not isinstance(script_plan, ScriptPlan) or not isinstance(voice_bank, VoiceBank):
            raise TypeError("批量配音必须连接脚本计划和音色库")
        if not script_plan.valid:
            raise ValueError("脚本预检存在错误，不能开始批量配音：" + _json(list(script_plan.issues)))
        project = _safe_name(project_name, "dubbing-project")
        clean_subfolder = subfolder.rstrip("/\\") or "fireredaudio/projects"
        project_dir = _safe_output_dir(f"{clean_subfolder}/{project}")
        manifest_path = project_dir / "manifest.json"
        previous = load_manifest(manifest_path) if resume and manifest_path.is_file() else None
        previous_items = manifest_items_by_id(previous)
        config = _settings(settings).to_dict()
        resolved_model_root = Path(model.model_root).resolve()
        model_identity = stable_digest(
            {
                "model_root": str(resolved_model_root),
                "model_fingerprint": fingerprint(resolved_model_root),
                "device": model.device,
                "memory_mode": model.memory_mode,
                "acceleration_mode": model.acceleration_mode,
            }
        )
        manifest_payload: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "project_name": project,
            "script_digest": script_plan.to_dict()["digest"],
            "voice_bank_digest": voice_bank.to_dict()["digest"],
            "settings": config,
            "model_identity": model_identity,
            "items": [],
        }
        generated = cached = failed = 0
        total = len(script_plan.lines)
        for position, line in enumerate(script_plan.lines, 1):
            profile = voice_bank.resolve(line.speaker)
            if profile is None:
                raise ValueError(f"角色没有音色档案：{line.speaker}")
            fingerprint_value = line_fingerprint(line, profile, config, model_identity)
            filename = _safe_name(f"{line.index:04d}-{line.speaker}-{line.line_id}", f"line-{line.index:04d}") + ".wav"
            target = (project_dir / filename).resolve()
            try:
                target.relative_to(project_dir.resolve())
            except ValueError as exc:
                raise ValueError("批量输出路径越界") from exc
            old = previous_items.get(line.line_id)
            item = {
                **line.to_dict(),
                "profile_id": profile.profile_id,
                "fingerprint": fingerprint_value,
                "output_path": str(target),
            }
            if resume and can_reuse_manifest_item(old, fingerprint_value, target):
                item.update(status="complete", cache_hit=True, worker_report=old.get("worker_report", {}))
                cached += 1
            else:
                try:
                    request = _base_request(model, "tts", _settings(settings))
                    request.update(
                        {
                            "prompt_audio": profile.prompt_audio,
                            "prompt_text": profile.prompt_text,
                            "target_text": line.text,
                            "language": line.language,
                            "output_path": str(target),
                        }
                    )
                    result = _infer(model, request)
                    actual = Path(str(result.get("output_path") or target))
                    if not actual.is_file():
                        raise RuntimeError("Worker 未产生预期音频文件")
                    if actual.resolve() != target:
                        shutil.copy2(actual, target)
                    item.update(status="complete", cache_hit=False, worker_report=result)
                    generated += 1
                except Exception as exc:
                    if _is_processing_interrupt(exc):
                        raise
                    item.update(status="failed", cache_hit=False, error=f"{type(exc).__name__}: {exc}")
                    failed += 1
                    manifest_payload["items"].append(item)
                    write_manifest(manifest_path, manifest_payload)
                    _set_official_progress(position / max(1, total))
                    if not continue_on_error:
                        raise
                    continue
            manifest_payload["items"].append(item)
            write_manifest(manifest_path, manifest_payload)
            _set_official_progress(position / max(1, total))
        batch = AudioBatch(str(manifest_path), tuple(manifest_payload["items"]))
        report = {
            "manifest_path": str(manifest_path),
            "total": total,
            "generated": generated,
            "cache_hits": cached,
            "failed": failed,
            "execution_model": "sequential_worker_calls",
            "native_tensor_batch": False,
        }
        return io.NodeOutput(batch, str(manifest_path), _json(report))


class T8FireRedAudioSynchronizedAB(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_SynchronizedAB",
            display_name="FireRedAudio 同步 A/B 对比 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="非破坏地对齐两个候选的有效声音起点、匹配 EBU R128 响度并补齐相同长度，便于公平盲听。",
            inputs=[
                io.Audio.Input("audio_a", display_name="候选 A"),
                io.Audio.Input("audio_b", display_name="候选 B"),
                io.Boolean.Input("synchronize_onset", display_name="同步有效声音起点", default=True),
                io.Boolean.Input("match_loudness", display_name="匹配 EBU R128 响度", default=True),
                io.Float.Input("target_lufs", display_name="目标 LUFS", default=-20.0, min=-35.0, max=-12.0, step=0.5),
                io.Float.Input("onset_threshold_dbfs", display_name="起点阈值 dBFS", default=-42.0, min=-70.0, max=-10.0, step=1.0, advanced=True),
                io.Int.Input("preroll_ms", display_name="起点前预留（毫秒）", default=20, min=0, max=500, advanced=True),
            ],
            outputs=[
                io.Audio.Output("audio_a_synced", display_name="同步候选 A"),
                io.Audio.Output("audio_b_synced", display_name="同步候选 B"),
                io.String.Output("comparison_report", display_name="A/B 对比报告"),
            ],
        )

    @classmethod
    def execute(
        cls,
        audio_a: dict,
        audio_b: dict,
        synchronize_onset: bool,
        match_loudness: bool,
        target_lufs: float,
        onset_threshold_dbfs: float,
        preroll_ms: int,
    ) -> io.NodeOutput:
        source_a = audio_to_wav(audio_a, "ab-source-a")
        source_b = audio_to_wav(audio_b, "ab-source-b")
        output_a = output_wav_path("ab-synced-a")
        output_b = output_wav_path("ab-synced-b")
        report = prepare_synchronized_ab(
            source_a,
            source_b,
            output_a,
            output_b,
            synchronize_onset=synchronize_onset,
            match_loudness=match_loudness,
            target_lufs=target_lufs,
            onset_threshold_dbfs=onset_threshold_dbfs,
            preroll_ms=preroll_ms,
        )
        return io.NodeOutput(wav_to_audio(output_a), wav_to_audio(output_b), _json(report))


class T8FireRedAudioTimelineRender(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_TimelineRender",
            display_name="FireRedAudio 时间线渲染 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="把批量配音按顺序、脚本时间码或全轨叠加渲染；支持相邻对白交叉淡化、room tone 自动补空隙和 EBU R128 交付预设。",
            inputs=[
                AudioBatchType.Input("audio_batch", display_name="批量音频"),
                io.Combo.Input("mode", display_name="排列模式", options=["sequence", "timeline", "overlay"], default="timeline"),
                io.Int.Input("gap_ms", display_name="顺序间隔（毫秒）", default=120, min=0, max=10000),
                io.Int.Input("crossfade_ms", display_name="交叉淡化（毫秒）", default=0, min=0, max=2000, tooltip="sequence 中大于 0 时用相邻重叠替代顺序间隔；timeline 中只处理实际重叠区域。"),
                io.Boolean.Input("auto_fill_gaps", display_name="用 room tone 自动补空隙", default=False),
                io.Combo.Input("peak_policy", display_name="峰值策略", options=["limit", "clip", "none"], default="limit"),
                io.Int.Input("sample_rate", display_name="输出采样率", default=24000, min=8000, max=192000, advanced=True),
                io.Audio.Input("room_tone_audio", display_name="可选 room tone", optional=True),
                DeliveryPresetType.Input("delivery_preset", display_name="可选交付预设", optional=True),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="时间线音频"),
                io.String.Output("timeline_report", display_name="时间线报告"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        audio_batch: AudioBatch,
        mode: str,
        gap_ms: int,
        crossfade_ms: int,
        auto_fill_gaps: bool,
        peak_policy: str,
        sample_rate: int,
        room_tone_audio: dict | None = None,
        delivery_preset: DeliveryPreset | None = None,
        **kwargs,
    ) -> str:
        return stable_digest(
            {
                "audio_batch": _audio_batch_state(audio_batch),
                "mode": mode,
                "gap_ms": gap_ms,
                "crossfade_ms": crossfade_ms,
                "auto_fill_gaps": auto_fill_gaps,
                "peak_policy": peak_policy,
                "sample_rate": sample_rate,
                "room_tone": (
                    str(audio_to_wav(room_tone_audio, "timeline-room-tone"))
                    if room_tone_audio is not None
                    else None
                ),
                "delivery_preset": delivery_preset.to_dict() if delivery_preset else None,
            }
        )

    @classmethod
    def execute(
        cls,
        audio_batch: AudioBatch,
        mode: str,
        gap_ms: int,
        crossfade_ms: int,
        auto_fill_gaps: bool,
        peak_policy: str,
        sample_rate: int,
        room_tone_audio: dict | None = None,
        delivery_preset: DeliveryPreset | None = None,
    ) -> io.NodeOutput:
        if not isinstance(audio_batch, AudioBatch):
            raise TypeError("时间线渲染必须连接批量音频")
        if delivery_preset is not None and not isinstance(delivery_preset, DeliveryPreset):
            raise TypeError("交付预设输入类型无效")
        effective_mode = delivery_preset.mode if delivery_preset else mode
        effective_gap = delivery_preset.gap_ms if delivery_preset else gap_ms
        effective_crossfade = delivery_preset.crossfade_ms if delivery_preset else crossfade_ms
        effective_sample_rate = delivery_preset.sample_rate if delivery_preset else sample_rate
        room_tone_path = (
            audio_to_wav(room_tone_audio, "timeline-room-tone")
            if room_tone_audio is not None
            else None
        )
        output = output_wav_path("timeline-render")
        report = render_timeline_to_wav(
            audio_batch.items,
            output,
            mode=effective_mode,
            gap_ms=effective_gap,
            crossfade_ms=effective_crossfade,
            peak_policy=peak_policy,
            sample_rate=effective_sample_rate,
            gap_fill_path=room_tone_path,
            auto_fill_gaps=auto_fill_gaps,
            target_lufs=(delivery_preset.target_lufs if delivery_preset else None),
            loudness_range_lu=(delivery_preset.loudness_range_lu if delivery_preset else 7.0),
            true_peak_dbfs=(delivery_preset.true_peak_dbfs if delivery_preset else -1.0),
        )
        report["delivery_preset"] = delivery_preset.to_dict() if delivery_preset else None
        return io.NodeOutput(wav_to_audio(output), _json(report))


class T8FireRedAudioSpeechQA(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_SpeechQA",
            display_name="FireRedAudio 成品语音 QA · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="逐条 ASR 回读并结合时长、静音、削波做成品质检；中文使用 CER，英文使用 WER。",
            inputs=[
                ModelType.Input("model"),
                AudioBatchType.Input("audio_batch", display_name="批量音频"),
                io.Float.Input("max_text_error_rate", display_name="最大 CER/WER", default=0.20, min=0.0, max=1.0, step=0.01),
                io.Float.Input("max_clipping_ratio", display_name="最大削波比例", default=0.001, min=0.0, max=1.0, step=0.0001),
                io.Float.Input("max_silence_ratio", display_name="最大静音比例", default=0.80, min=0.0, max=1.0, step=0.01),
                io.Float.Input("max_cue_overrun_seconds", display_name="最大超出时间槽（秒）", default=0.50, min=0.0, max=60.0, step=0.1),
                io.Int.Input("max_new_tokens", display_name="ASR 最大新 Token", default=512, min=1, max=4096),
            ],
            outputs=[
                SpeechQAType.Output("qa", display_name="语音 QA"),
                io.String.Output("qa_report", display_name="QA 报告"),
                io.String.Output("failed_line_ids", display_name="未通过条目"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        model: RuntimeHandle,
        audio_batch: AudioBatch,
        max_text_error_rate: float,
        max_clipping_ratio: float,
        max_silence_ratio: float,
        max_cue_overrun_seconds: float,
        max_new_tokens: int,
        **kwargs,
    ) -> str:
        return stable_digest(
            {
                "model": model.to_dict(),
                "audio_batch": _audio_batch_state(audio_batch),
                "thresholds": {
                    "max_text_error_rate": max_text_error_rate,
                    "max_clipping_ratio": max_clipping_ratio,
                    "max_silence_ratio": max_silence_ratio,
                    "max_cue_overrun_seconds": max_cue_overrun_seconds,
                    "max_new_tokens": max_new_tokens,
                },
            }
        )

    @classmethod
    def execute(
        cls,
        model: RuntimeHandle,
        audio_batch: AudioBatch,
        max_text_error_rate: float,
        max_clipping_ratio: float,
        max_silence_ratio: float,
        max_cue_overrun_seconds: float,
        max_new_tokens: int,
    ) -> io.NodeOutput:
        if not isinstance(audio_batch, AudioBatch):
            raise TypeError("语音 QA 必须连接批量音频")
        results: list[dict[str, Any]] = []
        failed_ids: list[str] = []
        items = audio_batch.successful_items()
        for position, item in enumerate(items, 1):
            line_id = str(item.get("line_id") or position)
            try:
                path = Path(str(item["output_path"]))
                metrics = wav_metrics(path)
                request = _base_request(model, "asr")
                request.update(
                    {
                        "audio_path": str(path),
                        "prompt": "Transcribe speech to text.",
                        "max_new_tokens": max_new_tokens,
                    }
                )
                inference = _infer(model, request)
                transcript = str(inference.get("answer") or "")
                metric_name, error_rate = text_error_rate(
                    str(item.get("text") or ""), transcript, str(item.get("language") or "zh")
                )
                cue_overrun = 0.0
                if item.get("start_seconds") is not None and item.get("end_seconds") is not None:
                    available = float(item["end_seconds"]) - float(item["start_seconds"])
                    cue_overrun = max(0.0, metrics["duration_seconds"] - available)
                checks = {
                    "text": error_rate <= max_text_error_rate,
                    "clipping": metrics["clipping_ratio"] <= max_clipping_ratio,
                    "silence": metrics["silence_ratio"] <= max_silence_ratio,
                    "cue_duration": cue_overrun <= max_cue_overrun_seconds,
                }
                passed = all(checks.values())
                result = {
                    "line_id": line_id,
                    "speaker": item.get("speaker"),
                    "reference_text": item.get("text"),
                    "transcript": transcript,
                    "metric": metric_name,
                    "text_error_rate": error_rate,
                    "cue_overrun_seconds": cue_overrun,
                    "audio_metrics": metrics,
                    "checks": checks,
                    "passed": passed,
                }
            except Exception as exc:
                if _is_processing_interrupt(exc):
                    raise
                passed = False
                result = {"line_id": line_id, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
            if not passed:
                failed_ids.append(line_id)
            results.append(result)
            _set_official_progress(position / max(1, len(items)))
        qa = {
            "passed": bool(results) and not failed_ids,
            "total": len(results),
            "passed_count": len(results) - len(failed_ids),
            "failed_count": len(failed_ids),
            "thresholds": {
                "max_text_error_rate": max_text_error_rate,
                "max_clipping_ratio": max_clipping_ratio,
                "max_silence_ratio": max_silence_ratio,
                "max_cue_overrun_seconds": max_cue_overrun_seconds,
            },
            "items": results,
        }
        return io.NodeOutput(qa, _json(qa), "\n".join(failed_ids))


class T8FireRedAudioSaveAudio(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_SaveAudio",
            display_name="FireRedAudio 保存音频 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="安全保存 WAV/FLAC/MP3/OGG 到 ComfyUI output 目录。",
            inputs=[
                io.Audio.Input("audio"),
                io.Combo.Input("audio_format", display_name="格式", options=["wav", "flac", "mp3", "ogg"], default="wav"),
                io.String.Input("filename_prefix", display_name="文件名前缀", default="fireredaudio"),
                io.String.Input("subfolder", display_name="输出子目录", default="fireredaudio"),
                DeliveryPresetType.Input("delivery_preset", display_name="可选交付预设", optional=True),
            ],
            outputs=[
                io.String.Output("saved_path", display_name="保存路径"),
                io.Audio.Output("audio", display_name="已保存音频"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        audio: dict,
        audio_format: str,
        filename_prefix: str,
        subfolder: str,
        delivery_preset: DeliveryPreset | None = None,
    ) -> io.NodeOutput:
        if delivery_preset is not None and not isinstance(delivery_preset, DeliveryPreset):
            raise TypeError("交付预设输入类型无效")
        effective_format = delivery_preset.audio_format if delivery_preset else audio_format
        target = save_audio_file(
            audio,
            filename_prefix=filename_prefix,
            subfolder=subfolder,
            audio_format=effective_format,
        )
        return io.NodeOutput(str(target), audio, ui=saved_audio_ui(target))


class T8FireRedAudioSaveSubtitle(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_SaveSubtitle",
            display_name="FireRedAudio 保存字幕/文本 · T8star-Aix",
            category=CATEGORY,
            description="安全保存 SRT/VTT/TXT/JSONL 到 ComfyUI output 目录。",
            inputs=[
                io.String.Input("content", display_name="字幕或文本", multiline=True, force_input=True),
                io.Combo.Input("text_format", display_name="格式", options=["srt", "vtt", "txt", "jsonl"], default="srt"),
                io.String.Input("filename_prefix", display_name="文件名前缀", default="fireredaudio"),
                io.String.Input("subfolder", display_name="输出子目录", default="fireredaudio"),
            ],
            outputs=[io.String.Output("saved_path", display_name="保存路径")],
        )

    @classmethod
    def execute(cls, content: str, text_format: str, filename_prefix: str, subfolder: str) -> io.NodeOutput:
        target = save_text_file(content, filename_prefix=filename_prefix, subfolder=subfolder, text_format=text_format)
        return io.NodeOutput(str(target))


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
                io.Combo.Input("action", display_name="操作", options=["status", "cache_status", "clear_audio_cache", "cancel_current", "unload_model", "stop_worker"], default="status"),
            ],
            outputs=[io.String.Output("status", display_name="运行时状态")],
        )

    @classmethod
    def execute(cls, model: RuntimeHandle, action: str) -> io.NodeOutput:
        if action == "unload_model":
            result = WORKER_MANAGER.unload(model)
        elif action == "cache_status":
            result = _client(model).cache_status()
        elif action == "clear_audio_cache":
            result = _client(model).cleanup_cache(clear_all=True)
        elif action == "cancel_current":
            result = _client(model).cancel()
        elif action == "stop_worker":
            WORKER_MANAGER.stop()
            result = {"stopped": True}
        else:
            client = _client(model)
            result = {"manager": WORKER_MANAGER.status(), "worker": client.health()}
        return io.NodeOutput(_json(result))


class T8FireRedAudioPerformanceReport(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_PerformanceReport",
            display_name="FireRedAudio 分阶段性能分析 · T8star-Aix",
            category=CATEGORY,
            description="解析生成节点报告，显示冷/热启动、各阶段耗时、实时率和 CUDA 峰值显存，用于选择真实有效的加速模式。",
            inputs=[
                io.String.Input("generation_report", display_name="生成报告 JSON", multiline=True, force_input=True),
                io.Float.Input("target_rtf", display_name="目标 RTF", default=1.0, min=0.0, max=1000.0, step=0.1),
            ],
            outputs=[
                io.String.Output("summary", display_name="性能摘要"),
                io.String.Output("performance_json", display_name="性能 JSON"),
                io.Float.Output("rtf", display_name="RTF"),
                io.Float.Output("total_seconds", display_name="总耗时（秒）"),
                io.Float.Output("peak_vram_gib", display_name="峰值显存 GiB"),
            ],
        )

    @classmethod
    def execute(cls, generation_report: str, target_rtf: float) -> io.NodeOutput:
        try:
            payload = json.loads(generation_report)
        except (TypeError, ValueError) as exc:
            raise ValueError("生成报告不是有效 JSON") from exc
        performance = payload.get("performance") if isinstance(payload, dict) else None
        if not isinstance(performance, dict):
            raise ValueError("生成报告中没有 performance 字段；请连接 v0.10 生成节点报告")
        total = float(performance.get("total_seconds") or payload.get("elapsed_seconds") or 0.0)
        rtf = float(performance.get("rtf") or 0.0)
        peak_bytes = int(performance.get("gpu_peak_allocated_bytes") or 0)
        peak_gib = peak_bytes / (1024 ** 3)
        phases = performance.get("phase_seconds") or {}
        phase_copy = "、".join(
            f"{name} {float(seconds):.2f}s"
            for name, seconds in sorted(phases.items(), key=lambda item: float(item[1]), reverse=True)
        ) or "无阶段数据"
        target_met = rtf > 0 and rtf <= float(target_rtf)
        summary = "\n".join(
            [
                f"{'冷启动' if performance.get('cold_start') else '热运行'} · {performance.get('device') or 'unknown'} · {performance.get('acceleration_mode') or 'unknown'}",
                f"总耗时 {total:.3f}s · RTF {rtf:.3f} · 峰值显存 {peak_gib:.3f} GiB",
                f"目标 RTF ≤ {float(target_rtf):.3f}：{'达标' if target_met else '未达标或无输出时长'}",
                f"阶段：{phase_copy}",
            ]
        )
        enriched = {**performance, "target_rtf": float(target_rtf), "target_met": target_met}
        return io.NodeOutput(summary, _json(enriched), rtf, total, peak_gib)


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
            "isolated_worker_required": {"python": "3.10.x", "torch": "2.8.0+cu128", "transformers": "5.8.0"},
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
            T8FireRedAudioDeliveryPreset,
            T8FireRedAudioASR,
            T8FireRedAudioReferenceTranscript,
            T8FireRedAudioLongASR,
            T8FireRedAudioLongLocator,
            T8FireRedAudioEvidenceClips,
            T8FireRedAudioUnderstand,
            T8FireRedAudioMultiUnderstand,
            T8FireRedAudioTTS,
            T8FireRedAudioSeedAudition,
            T8FireRedAudioVoiceDesign,
            T8FireRedAudioSpeechEdit,
            T8FireRedAudioAcousticEdit,
            T8FireRedAudioReferenceQuality,
            T8FireRedAudioPrepareReference,
            T8FireRedAudioProjectExchange,
            T8FireRedAudioVoiceProfile,
            T8FireRedAudioVoiceBank,
            T8FireRedAudioScriptParser,
            T8FireRedAudioBatchDubbing,
            T8FireRedAudioSynchronizedAB,
            T8FireRedAudioTimelineRender,
            T8FireRedAudioSpeechQA,
            T8FireRedAudioSaveAudio,
            T8FireRedAudioSaveSubtitle,
            T8FireRedAudioRuntimeControl,
            T8FireRedAudioPerformanceReport,
            T8FireRedAudioEnvironment,
        ]


async def comfy_entrypoint() -> T8FireRedAudioExtension:
    return T8FireRedAudioExtension()


__all__ = [name for name in globals() if name.startswith("T8FireRedAudio") or name == "comfy_entrypoint"]
