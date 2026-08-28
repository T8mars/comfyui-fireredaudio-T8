from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import shutil
import statistics
import sys
import threading
import time
import uuid
import zipfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from comfy_api.latest import ComfyExtension, io

from .runtime.acoustic import acoustic_instruction
from .runtime.asr_cache import (
    build_asr_cache_descriptor,
    load_cached_transcript,
    store_cached_transcript,
)
from .runtime.audio_adapter import (
    _safe_name,
    _safe_output_dir,
    audio_to_wav,
    export_audio_path,
    output_root,
    output_wav_path,
    save_audio_file,
    save_text_file,
    saved_audio_files_ui,
    saved_audio_ui,
    wav_to_audio,
)
from .runtime.creator_tools import (
    build_line_review,
    fit_audio_batch_to_cues,
    load_audio_batch_from_manifest,
    normalize_script_plan,
    parse_json_mapping,
)
from .runtime.evidence import (
    extract_evidence_ranges,
    parse_structured_json,
    render_evidence_clips,
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
from .runtime.postproduction import prepare_synchronized_ab
from .runtime.production import (
    MANIFEST_VERSION,
    AudioBatch,
    ScriptPlan,
    VoiceBank,
    acoustic_signature_distance,
    build_batch_subtitles,
    can_reuse_manifest_item,
    create_voice_bank,
    create_voice_profile,
    crop_wav_region,
    file_digest,
    line_fingerprint,
    load_manifest,
    load_project_exchange,
    manifest_items_by_id,
    merge_audio_batch_items,
    parse_line_ids,
    parse_script,
    render_grouped_stems,
    render_timeline_to_wav,
    replace_wav_region,
    select_audio_batch_item,
    stable_digest,
    text_error_rate,
    wav_acoustic_signature,
    wav_metrics,
    write_manifest,
)
from .runtime.reference_candidates import (
    asr_intelligibility_proxy,
    discover_reference_candidates,
)
from .runtime.types import (
    DELIVERY_PRESETS,
    DeliveryPreset,
    GenerationSettings,
    LocalRepairPlan,
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
LocalRepairPlanType = io.Custom("T8_FIREREDAUDIO_LOCAL_REPAIR_PLAN")
NODE_VERSION = "0.18.0"

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
                    # Keep filenames opaque so the native preview list can be used
                    # for a real blind listen; the seed remains in the manifest.
                    "output_path": str(
                        project_dir / f"blind-{uuid.uuid4().hex[:12]}.wav"
                    ),
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


class T8FireRedAudioLocalRepairRange(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_LocalRepairRange",
            display_name="FireRedAudio 局部修复范围 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description=(
                "从原音频按手工时间或长音频定位 JSON 非破坏裁出修复片段；"
                "原版、片段与不可变修复计划会一起传给编辑和回填节点。"
            ),
            inputs=[
                io.Audio.Input("audio", display_name="原始音频"),
                io.Combo.Input("range_mode", display_name="范围来源", options=["manual", "locator_json"], default="manual"),
                io.Float.Input("start_seconds", display_name="手工开始（秒）", default=0.0, min=0.0, max=86400.0, step=0.01),
                io.Float.Input("end_seconds", display_name="手工结束（秒）", default=5.0, min=0.01, max=86400.0, step=0.01),
                io.Int.Input("range_index", display_name="定位结果序号", default=1, min=1, max=100),
                io.Int.Input("context_ms", display_name="两侧上下文（毫秒）", default=250, min=0, max=5000),
                io.String.Input("locator_json", display_name="可选定位 JSON", multiline=True, force_input=True, optional=True),
            ],
            outputs=[
                io.Audio.Output("original_audio", display_name="原版 A"),
                io.Audio.Output("repair_clip", display_name="待编辑片段"),
                LocalRepairPlanType.Output("repair_plan", display_name="局部修复计划"),
                io.String.Output("range_report", display_name="范围报告"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        audio: dict,
        range_mode: str,
        start_seconds: float,
        end_seconds: float,
        range_index: int,
        context_ms: int,
        locator_json: str = "",
        **kwargs,
    ) -> str:
        source = audio_to_wav(audio, "local-repair-source")
        return stable_digest(
            {
                "source_sha256": file_digest(source),
                "range_mode": range_mode,
                "start_seconds": float(start_seconds),
                "end_seconds": float(end_seconds),
                "range_index": int(range_index),
                "context_ms": int(context_ms),
                "locator_json": str(locator_json or ""),
            }
        )

    @classmethod
    def execute(
        cls,
        audio: dict,
        range_mode: str,
        start_seconds: float,
        end_seconds: float,
        range_index: int,
        context_ms: int,
        locator_json: str = "",
    ) -> io.NodeOutput:
        selected_start = float(start_seconds)
        selected_end = float(end_seconds)
        range_label = "手工范围"
        if range_mode == "locator_json":
            if not str(locator_json or "").strip():
                raise ValueError("locator_json 模式必须连接长音频时间定位 JSON")
            ranges = extract_evidence_ranges(
                parse_structured_json(locator_json), default_clip_seconds=8.0, max_clips=100
            )
            if not ranges:
                raise ValueError("定位 JSON 中没有可用时间范围")
            position = int(range_index) - 1
            if position < 0 or position >= len(ranges):
                raise IndexError(f"定位结果序号超出范围：1–{len(ranges)}")
            selected = ranges[position]
            selected_start = float(selected["start_seconds"])
            selected_end = float(selected["end_seconds"])
            range_label = str(selected.get("label") or f"定位结果 {range_index}")
        elif range_mode != "manual":
            raise ValueError(f"不支持的局部修复范围来源：{range_mode}")
        source = audio_to_wav(audio, "local-repair-source")
        clip = output_wav_path("local-repair-clip")
        report = crop_wav_region(
            source,
            clip,
            start_seconds=selected_start,
            end_seconds=selected_end,
            context_ms=context_ms,
        )
        report.update(range_source=range_mode, range_label=range_label)
        plan = LocalRepairPlan(
            source_path=str(source),
            source_sha256=str(report["source_sha256"]),
            sample_rate=int(report["sample_rate"]),
            channels=int(report["channels"]),
            source_frames=int(report["source_frames"]),
            requested_start_seconds=selected_start,
            requested_end_seconds=selected_end,
            replace_start_frame=int(report["replace_start_frame"]),
            replace_end_frame=int(report["replace_end_frame"]),
            context_ms=int(context_ms),
            range_source=range_mode,
            range_label=range_label,
        )
        return io.NodeOutput(audio, wav_to_audio(clip), plan, _json(report))


class T8FireRedAudioLocalRepairApply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_LocalRepairApply",
            display_name="FireRedAudio 局部修复回填 A/B · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description=(
                "把语义或声学编辑后的片段非破坏回填到原音频；保留原声道和采样率，"
                "使用等功率交叉淡化并同时输出原版/修复版与哈希审计报告。"
            ),
            inputs=[
                LocalRepairPlanType.Input("repair_plan", display_name="局部修复计划"),
                io.Audio.Input("edited_clip", display_name="编辑后片段"),
                io.Int.Input("crossfade_ms", display_name="边缘交叉淡化（毫秒）", default=40, min=0, max=2000),
            ],
            outputs=[
                io.Audio.Output("original_audio", display_name="原版 A"),
                io.Audio.Output("repaired_audio", display_name="修复版 B"),
                io.String.Output("replacement_report", display_name="替换审计报告"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        repair_plan: LocalRepairPlan,
        edited_clip: dict,
        crossfade_ms: int,
        **kwargs,
    ) -> str:
        if not isinstance(repair_plan, LocalRepairPlan):
            return "invalid-local-repair-plan"
        edited = audio_to_wav(edited_clip, "local-repair-edited")
        return stable_digest(
            {
                "repair_plan": repair_plan.to_dict(),
                "edited_sha256": file_digest(edited),
                "crossfade_ms": int(crossfade_ms),
            }
        )

    @classmethod
    def execute(
        cls,
        repair_plan: LocalRepairPlan,
        edited_clip: dict,
        crossfade_ms: int,
    ) -> io.NodeOutput:
        if not isinstance(repair_plan, LocalRepairPlan):
            raise TypeError("局部修复回填必须连接局部修复计划")
        edited = audio_to_wav(edited_clip, "local-repair-edited")
        output = output_wav_path("local-repair-result")
        report = replace_wav_region(
            repair_plan.source_path,
            edited,
            output,
            replace_start_frame=repair_plan.replace_start_frame,
            replace_end_frame=repair_plan.replace_end_frame,
            crossfade_ms=crossfade_ms,
            expected_source_sha256=repair_plan.source_sha256,
        )
        report["repair_plan"] = repair_plan.to_dict()
        return io.NodeOutput(
            wav_to_audio(repair_plan.source_path),
            wav_to_audio(output),
            _json(report),
        )


class T8FireRedAudioReferenceCandidates(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_ReferenceCandidates",
            display_name="FireRedAudio 长录音参考候选 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description=(
                "从长录音非破坏提取 3–15 秒候选，按削波、静音、能量对比、"
                "语音活动和时长排序；可选 ASR 可懂度代理，最终必须人工试听。"
            ),
            inputs=[
                io.Audio.Input("source_audio", display_name="长录音"),
                io.String.Input("project_name", display_name="候选项目名", default="reference-search"),
                io.String.Input("subfolder", display_name="输出子目录", default="fireredaudio/reference-candidates"),
                io.Float.Input("min_seconds", display_name="最短候选（秒）", default=3.0, min=1.0, max=30.0, step=0.5),
                io.Float.Input("preferred_seconds", display_name="推荐候选（秒）", default=8.0, min=1.0, max=30.0, step=0.5),
                io.Float.Input("max_seconds", display_name="最长候选（秒）", default=15.0, min=1.0, max=30.0, step=0.5),
                io.Float.Input("padding_seconds", display_name="语音前后留白（秒）", default=0.2, min=0.0, max=2.0, step=0.05),
                io.Int.Input("max_candidates", display_name="候选数量", default=8, min=1, max=20),
                io.Boolean.Input("run_asr_check", display_name="使用 ASR 可懂度代理复排", default=False),
                io.Combo.Input("language", display_name="ASR 语言", options=["zh", "en"], default="zh"),
                ModelType.Input("model", display_name="可选 FireRedAudio 运行时", optional=True),
            ],
            outputs=[
                io.Audio.Output("recommended_audio", display_name="信号/ASR 推荐候选"),
                AudioBatchType.Output("candidates", display_name="全部候选"),
                io.String.Output("recommended_line_id", display_name="推荐 line ID"),
                io.String.Output("manifest_path", display_name="候选 Manifest 路径"),
                io.String.Output("ranking_report", display_name="排序报告"),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        min_seconds: float,
        preferred_seconds: float,
        max_seconds: float,
        run_asr_check: bool,
        model: RuntimeHandle | None = None,
        **kwargs,
    ) -> bool | str:
        if not 1.0 <= float(min_seconds) <= float(preferred_seconds) <= float(max_seconds) <= 30.0:
            return "候选时长必须满足 1 ≤ 最短 ≤ 推荐 ≤ 最长 ≤ 30 秒。"
        if run_asr_check and model is None:
            return "启用 ASR 可懂度代理时必须连接 FireRedAudio 运行时。"
        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        source_audio: dict,
        project_name: str,
        subfolder: str,
        min_seconds: float,
        preferred_seconds: float,
        max_seconds: float,
        padding_seconds: float,
        max_candidates: int,
        run_asr_check: bool,
        language: str,
        model: RuntimeHandle | None = None,
        **kwargs,
    ) -> str:
        source = audio_to_wav(source_audio, "reference-candidates-source")
        return stable_digest(
            {
                "source_sha256": file_digest(source),
                "project_name": project_name,
                "subfolder": subfolder,
                "min_seconds": float(min_seconds),
                "preferred_seconds": float(preferred_seconds),
                "max_seconds": float(max_seconds),
                "padding_seconds": float(padding_seconds),
                "max_candidates": int(max_candidates),
                "run_asr_check": bool(run_asr_check),
                "language": language,
                "model": model.to_dict() if isinstance(model, RuntimeHandle) else None,
            }
        )

    @classmethod
    def execute(
        cls,
        source_audio: dict,
        project_name: str,
        subfolder: str,
        min_seconds: float,
        preferred_seconds: float,
        max_seconds: float,
        padding_seconds: float,
        max_candidates: int,
        run_asr_check: bool,
        language: str,
        model: RuntimeHandle | None = None,
    ) -> io.NodeOutput:
        if run_asr_check and model is None:
            raise ValueError("启用 ASR 可懂度代理时必须连接 FireRedAudio 运行时")
        source = audio_to_wav(source_audio, "reference-candidates-source")
        project = _safe_name(project_name, "reference-search")
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        clean_subfolder = subfolder.rstrip("/\\") or "fireredaudio/reference-candidates"
        output_dir = _safe_output_dir(f"{clean_subfolder}/{project}-{stamp}")
        items, report = discover_reference_candidates(
            source,
            output_dir,
            min_seconds=min_seconds,
            preferred_seconds=preferred_seconds,
            max_seconds=max_seconds,
            padding_seconds=padding_seconds,
            max_candidates=max_candidates,
        )
        if run_asr_check:
            assert model is not None
            for position, item in enumerate(items, 1):
                transcript, asr_report = _transcribe_reference(model, item["output_path"])
                proxy = asr_intelligibility_proxy(
                    transcript,
                    float(item["duration_seconds"]),
                    language,
                )
                item["text"] = transcript
                item["language"] = language
                item["asr_intelligibility_proxy"] = proxy
                item["asr_performance"] = asr_report.get("performance")
                item["ranking_score"] = round(
                    float(item["signal_score"]) * 0.8 + float(proxy["score"]) * 0.2,
                    3,
                )
                _set_official_progress(position / max(1, len(items)))
        else:
            for item in items:
                item["ranking_score"] = float(item["signal_score"])
        items.sort(key=lambda item: (-float(item["ranking_score"]), int(item["index"])))
        recommended = items[0]
        manifest_path = output_dir / "reference-candidates-manifest.json"
        report.update(
            {
                "manifest_version": MANIFEST_VERSION,
                "node_version": NODE_VERSION,
                "run_asr_check": bool(run_asr_check),
                "asr_metric_is_proxy_not_accuracy": bool(run_asr_check),
                "recommended_line_id": recommended["line_id"],
                "items": items,
            }
        )
        write_manifest(manifest_path, report)
        batch = AudioBatch(str(manifest_path), tuple(items))
        return io.NodeOutput(
            wav_to_audio(recommended["output_path"]),
            batch,
            str(recommended["line_id"]),
            str(manifest_path),
            _json(report),
            ui=saved_audio_files_ui([item["output_path"] for item in items]),
        )


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


class T8FireRedAudioAudioBatchResume(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_AudioBatchResume",
            display_name="FireRedAudio 恢复批次/审核会话 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description=(
                "从既有批量、返修、时长适配或审核 Manifest 恢复 AudioBatch。"
                "默认只读取 ComfyUI output，缺失文件会显式标记。"
            ),
            inputs=[
                io.String.Input("manifest_path", display_name="Manifest 路径", default=""),
                io.Combo.Input(
                    "missing_policy",
                    display_name="缺失文件处理",
                    options=["mark_missing", "error"],
                    default="mark_missing",
                ),
                io.Boolean.Input("verify_hashes", display_name="校验已记录 SHA-256", default=False),
                io.Boolean.Input(
                    "allow_external_manifest",
                    display_name="允许读取 output 外部 Manifest",
                    default=False,
                    advanced=True,
                ),
            ],
            outputs=[
                AudioBatchType.Output("audio_batch", display_name="已恢复 AudioBatch"),
                io.String.Output("resolved_manifest_path", display_name="Manifest 绝对路径"),
                io.String.Output("resume_report", display_name="恢复报告"),
            ],
        )

    @classmethod
    def validate_inputs(cls, manifest_path: str, **kwargs) -> bool | str:
        return True if str(manifest_path).strip() else "Manifest 路径不能为空。"

    @classmethod
    def fingerprint_inputs(
        cls,
        manifest_path: str,
        missing_policy: str,
        verify_hashes: bool,
        allow_external_manifest: bool,
        **kwargs,
    ) -> str:
        raw = Path(str(manifest_path).strip())
        target = raw if raw.is_absolute() else output_root() / raw
        target = target.resolve()
        state: dict[str, Any] = {"exists": target.is_file()}
        if target.is_file():
            stat = target.stat()
            state.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        return stable_digest(
            {
                "path": str(target),
                "state": state,
                "missing_policy": missing_policy,
                "verify_hashes": bool(verify_hashes),
                "allow_external_manifest": bool(allow_external_manifest),
            }
        )

    @classmethod
    def execute(
        cls,
        manifest_path: str,
        missing_policy: str,
        verify_hashes: bool,
        allow_external_manifest: bool,
    ) -> io.NodeOutput:
        raw = Path(str(manifest_path).strip())
        target = raw if raw.is_absolute() else output_root() / raw
        target = target.resolve()
        batch, report = load_audio_batch_from_manifest(
            target,
            allowed_root=None if allow_external_manifest else output_root(),
            missing_policy=missing_policy,
            verify_hashes=verify_hashes,
        )
        return io.NodeOutput(
            batch,
            str(target),
            _json(report),
            ui={"fireredaudio_resume_dashboard": [report]},
        )


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


class T8FireRedAudioTextNormalizer(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_TextNormalizer",
            display_name="FireRedAudio 朗读文本规范化 · T8star-Aix",
            category=CATEGORY,
            description=(
                "在不丢失原文的前提下生成实际送入 TTS 的朗读文本；支持自定义词典、"
                "Unicode/空白清理、中文日期与可选数字展开。"
            ),
            inputs=[
                ScriptPlanType.Input("script_plan", display_name="原脚本计划"),
                io.String.Input(
                    "replacement_dictionary_json",
                    display_name='替换词典 JSON，例如 {"API":"A P I"}',
                    default="{}",
                    multiline=True,
                ),
                io.Boolean.Input("normalize_unicode", display_name="统一全角/兼容字符", default=True),
                io.Boolean.Input("normalize_whitespace", display_name="清理空白与标点前空格", default=True),
                io.Boolean.Input("expand_zh_dates", display_name="展开中文日期", default=True),
                io.Boolean.Input(
                    "expand_zh_numbers",
                    display_name="展开中文数字（可能改变专有编号读法）",
                    default=False,
                    advanced=True,
                ),
            ],
            outputs=[
                ScriptPlanType.Output("script_plan", display_name="规范化脚本计划"),
                io.String.Output("normalized_json", display_name="原文/朗读文本对照 JSON"),
                io.String.Output("changed_line_ids", display_name="发生变化的 line ID"),
                io.String.Output("normalization_report", display_name="规范化报告"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        script_plan: ScriptPlan,
        replacement_dictionary_json: str,
        normalize_unicode: bool,
        normalize_whitespace: bool,
        expand_zh_dates: bool,
        expand_zh_numbers: bool,
        **kwargs,
    ) -> str:
        return stable_digest(
            {
                "script_plan": script_plan.to_dict() if isinstance(script_plan, ScriptPlan) else None,
                "replacement_dictionary_json": replacement_dictionary_json,
                "normalize_unicode": bool(normalize_unicode),
                "normalize_whitespace": bool(normalize_whitespace),
                "expand_zh_dates": bool(expand_zh_dates),
                "expand_zh_numbers": bool(expand_zh_numbers),
            }
        )

    @classmethod
    def execute(
        cls,
        script_plan: ScriptPlan,
        replacement_dictionary_json: str,
        normalize_unicode: bool,
        normalize_whitespace: bool,
        expand_zh_dates: bool,
        expand_zh_numbers: bool,
    ) -> io.NodeOutput:
        dictionary = parse_json_mapping(replacement_dictionary_json, "替换词典")
        plan, report = normalize_script_plan(
            script_plan,
            replacements=dictionary,
            normalize_unicode=normalize_unicode,
            normalize_whitespace=normalize_whitespace,
            expand_zh_dates=expand_zh_dates,
            expand_zh_numbers=expand_zh_numbers,
        )
        changed_ids = [str(item["line_id"]) for item in report["items"]]
        comparison = {
            "source_format": plan.source_format,
            "valid": plan.valid,
            "lines": [
                {
                    "line_id": line.line_id,
                    "index": line.index,
                    "speaker": line.speaker,
                    "source_text": line.source_text or line.text,
                    "spoken_text": line.text,
                    "normalization": list(line.normalization),
                }
                for line in plan.lines
            ],
        }
        return io.NodeOutput(plan, _json(comparison), "\n".join(changed_ids), _json(report))


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
                io.Int.Input("batch_size", display_name="每批条数", default=8, min=1, max=32, tooltip="每批先生成全部 latent，再统一切换解码器；24GB 显存建议 4–8。"),
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
        batch_size: int = 8,
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
                    "batch_size": int(batch_size),
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
        batch_size: int = 8,
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
        effective_batch_size = max(1, min(32, int(batch_size)))
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
        pending: list[dict[str, Any]] = []
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
                item.update(status="pending", cache_hit=False)
                request = _base_request(model, "tts", _settings(settings))
                request.update(
                    {
                        "task_id": f"dubbing-{project}-{line.line_id}",
                        "prompt_audio": profile.prompt_audio,
                        "prompt_text": profile.prompt_text,
                        "target_text": line.text,
                        "language": line.language,
                        "output_path": str(target),
                        "release_after": False,
                    }
                )
                pending.append(
                    {
                        "position": position,
                        "item": item,
                        "target": target,
                        "request": request,
                    }
                )
            manifest_payload["items"].append(item)
        write_manifest(manifest_path, manifest_payload)

        batch_reports: list[dict[str, Any]] = []
        hard_error: BaseException | None = None
        for chunk_start in range(0, len(pending), effective_batch_size):
            chunk = pending[chunk_start : chunk_start + effective_batch_size]
            requests = [dict(entry["request"]) for entry in chunk]
            is_last_chunk = chunk_start + len(chunk) >= len(pending)
            if is_last_chunk and model.release_after:
                for request in requests:
                    request["release_after"] = True
            try:
                batch_result = _infer_tts_batch(model, requests)
                outcomes = {
                    int(outcome.get("index", -1)): outcome
                    for outcome in batch_result.get("outcomes", [])
                    if isinstance(outcome, dict)
                }
                batch_reports.append(dict(batch_result.get("performance") or {}))
                for local_index, entry in enumerate(chunk):
                    item = entry["item"]
                    target = entry["target"]
                    outcome = outcomes.get(local_index)
                    try:
                        if not outcome:
                            raise RuntimeError("Worker 批量结果缺少对应条目")
                        if not outcome.get("ok"):
                            raise RuntimeError(str(outcome.get("error") or "Worker 批量 TTS 失败"))
                        result = dict(outcome.get("result") or {})
                        actual = Path(str(result.get("output_path") or target))
                        if not actual.is_file():
                            raise RuntimeError("Worker 未产生预期音频文件")
                        if actual.resolve() != target:
                            shutil.copy2(actual, target)
                        item.update(
                            status="complete",
                            cache_hit=False,
                            worker_report=result,
                        )
                        item.pop("error", None)
                        generated += 1
                    except Exception as exc:
                        item.update(
                            status="failed",
                            cache_hit=False,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                        failed += 1
                        if not continue_on_error and hard_error is None:
                            hard_error = exc
                    write_manifest(manifest_path, manifest_payload)
                    completed_positions = cached + generated + failed
                    _set_official_progress(completed_positions / max(1, total))
            except BaseException as exc:
                if _is_processing_interrupt(exc) or not isinstance(exc, Exception):
                    write_manifest(manifest_path, manifest_payload)
                    if model.release_after:
                        try:
                            _client(model).unload()
                        except Exception as unload_exc:
                            LOGGER.warning("批量取消后释放 Worker 模型失败：%s", unload_exc)
                    raise
                for entry in chunk:
                    item = entry["item"]
                    item.update(
                        status="failed",
                        cache_hit=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    failed += 1
                write_manifest(manifest_path, manifest_payload)
                if not continue_on_error:
                    if model.release_after:
                        try:
                            _client(model).unload()
                        except Exception as unload_exc:
                            LOGGER.warning("批量失败后释放 Worker 模型失败：%s", unload_exc)
                    raise
            if hard_error is not None:
                break

        if hard_error is not None:
            write_manifest(manifest_path, manifest_payload)
            if model.release_after:
                try:
                    _client(model).unload()
                except Exception as unload_exc:
                    LOGGER.warning("批量失败后释放 Worker 模型失败：%s", unload_exc)
            raise RuntimeError(f"批量配音存在失败条目：{hard_error}") from hard_error
        batch = AudioBatch(str(manifest_path), tuple(manifest_payload["items"]))
        execution_models = sorted(
            {
                str(report.get("execution_model"))
                for report in batch_reports
                if report.get("execution_model")
            }
        )
        report = {
            "manifest_path": str(manifest_path),
            "total": total,
            "generated": generated,
            "cache_hits": cached,
            "failed": failed,
            "batch_size": effective_batch_size,
            "batch_count": len(batch_reports),
            "execution_model": (
                "manifest_cache_only"
                if not batch_reports
                else execution_models[0]
                if len(execution_models) == 1
                else execution_models
            ),
            "worker_batch_route": True,
            "native_tensor_batch": False,
            "performance": batch_reports,
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
                io.Boolean.Input("use_asr_cache", display_name="复用 ASR 转写缓存", default=True, advanced=True),
                io.Boolean.Input("refresh_asr_cache", display_name="强制刷新 ASR 缓存", default=False, advanced=True),
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
        use_asr_cache: bool = True,
        refresh_asr_cache: bool = False,
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
                "asr_cache": {
                    "enabled": bool(use_asr_cache),
                    "refresh": bool(refresh_asr_cache),
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
        use_asr_cache: bool = True,
        refresh_asr_cache: bool = False,
    ) -> io.NodeOutput:
        if not isinstance(audio_batch, AudioBatch):
            raise TypeError("语音 QA 必须连接批量音频")
        results: list[dict[str, Any]] = []
        failed_ids: list[str] = []
        items = audio_batch.successful_items()
        prompt = "Transcribe speech to text."
        definition = manifest()
        model_identity = fingerprint(Path(model.model_root).resolve())
        cache_root = output_root() / "fireredaudio" / "qa-cache" / "asr"
        cache_hits = 0
        cache_misses = 0
        for position, item in enumerate(items, 1):
            line_id = str(item.get("line_id") or position)
            try:
                path = Path(str(item["output_path"]))
                metrics = wav_metrics(path)
                descriptor = build_asr_cache_descriptor(
                    path,
                    model_revision=str(definition["modelRevision"]),
                    model_fingerprint=model_identity,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )
                cached = None
                if use_asr_cache and not refresh_asr_cache:
                    cached = load_cached_transcript(cache_root, descriptor)
                if cached is not None:
                    transcript = str(cached["transcript"])
                    transcript_cache_path = str(cached["cache_path"])
                    asr_cache_hit = True
                    cache_hits += 1
                else:
                    request = _base_request(model, "asr")
                    request.update(
                        {
                            "audio_path": str(path),
                            "prompt": prompt,
                            "max_new_tokens": max_new_tokens,
                        }
                    )
                    inference = _infer(model, request)
                    transcript = str(inference.get("answer") or "")
                    transcript_cache_path = ""
                    if use_asr_cache:
                        transcript_cache_path = str(
                            store_cached_transcript(cache_root, descriptor, transcript)
                        )
                    asr_cache_hit = False
                    cache_misses += 1
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
                    "asr_cache_hit": asr_cache_hit,
                    "asr_cache_key": descriptor["cache_key"],
                    "asr_cache_path": transcript_cache_path,
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
            "asr_cache": {
                "enabled": bool(use_asr_cache),
                "refresh": bool(refresh_asr_cache),
                "hits": cache_hits,
                "misses": cache_misses,
                "directory": str(cache_root),
                "identity": {
                    "model_revision": str(definition["modelRevision"]),
                    "model_fingerprint": model_identity,
                    "prompt": prompt,
                    "max_new_tokens": int(max_new_tokens),
                },
            },
            "thresholds": {
                "max_text_error_rate": max_text_error_rate,
                "max_clipping_ratio": max_clipping_ratio,
                "max_silence_ratio": max_silence_ratio,
                "max_cue_overrun_seconds": max_cue_overrun_seconds,
            },
            "items": results,
        }
        return io.NodeOutput(qa, _json(qa), "\n".join(failed_ids))


class T8FireRedAudioDurationFit(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_DurationFit",
            display_name="FireRedAudio 字幕时长适配 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description=(
                "按 SRT 时间槽报告超时；语音感知模式先裁掉首尾多余静音，再只对剩余超时"
                "做安全范围内的 FFmpeg atempo。超过上限的条目进入重做清单，始终保留源文件。"
            ),
            inputs=[
                AudioBatchType.Input("audio_batch", display_name="批量音频"),
                io.Combo.Input(
                    "strategy",
                    display_name="处理策略",
                    options=["speech_aware", "safe_stretch", "report_only"],
                    default="speech_aware",
                ),
                io.Float.Input(
                    "tolerance_seconds",
                    display_name="允许时间差（秒）",
                    default=0.10,
                    min=0.0,
                    max=5.0,
                    step=0.05,
                ),
                io.Float.Input(
                    "maximum_speed",
                    display_name="最大安全加速倍率",
                    default=1.15,
                    min=1.0,
                    max=2.0,
                    step=0.01,
                ),
                io.Boolean.Input("fit_underrun", display_name="同时拉伸明显过短台词", default=False),
                io.Float.Input(
                    "minimum_speed",
                    display_name="最小安全减速倍率",
                    default=0.90,
                    min=0.5,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.String.Input("project_name", display_name="适配项目名", default="subtitle-fit"),
                io.String.Input("subfolder", display_name="输出子目录", default="fireredaudio/duration-fit"),
                io.Float.Input(
                    "edge_silence_threshold_db",
                    display_name="首尾静音阈值（dB）",
                    default=-40.0,
                    min=-80.0,
                    max=-20.0,
                    step=1.0,
                    advanced=True,
                ),
                io.Float.Input(
                    "edge_silence_min_seconds",
                    display_name="静音最短时长（秒）",
                    default=0.05,
                    min=0.01,
                    max=2.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "edge_padding_seconds",
                    display_name="保留首尾缓冲（秒）",
                    default=0.12,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "internal_pause_min_seconds",
                    display_name="保护内部停顿（秒以上）",
                    default=0.18,
                    min=0.05,
                    max=2.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "maximum_speech_speed",
                    display_name="自然语速上限",
                    default=1.12,
                    min=1.0,
                    max=2.0,
                    step=0.01,
                    advanced=True,
                ),
            ],
            outputs=[
                AudioBatchType.Output("audio_batch", display_name="适配后 AudioBatch"),
                io.String.Output("manifest_path", display_name="适配 Manifest 路径"),
                io.String.Output("retry_line_ids", display_name="建议重新生成的 line ID"),
                io.String.Output("fit_report", display_name="时长适配报告"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        audio_batch: AudioBatch,
        strategy: str,
        tolerance_seconds: float,
        maximum_speed: float,
        fit_underrun: bool,
        minimum_speed: float,
        project_name: str,
        subfolder: str,
        edge_silence_threshold_db: float = -40.0,
        edge_silence_min_seconds: float = 0.05,
        edge_padding_seconds: float = 0.12,
        internal_pause_min_seconds: float = 0.18,
        maximum_speech_speed: float = 1.12,
        **kwargs,
    ) -> str:
        return stable_digest(
            {
                "audio_batch": _audio_batch_state(audio_batch),
                "strategy": strategy,
                "tolerance_seconds": float(tolerance_seconds),
                "maximum_speed": float(maximum_speed),
                "fit_underrun": bool(fit_underrun),
                "minimum_speed": float(minimum_speed),
                "project_name": project_name,
                "subfolder": subfolder,
                "edge_silence_threshold_db": float(edge_silence_threshold_db),
                "edge_silence_min_seconds": float(edge_silence_min_seconds),
                "edge_padding_seconds": float(edge_padding_seconds),
                "internal_pause_min_seconds": float(internal_pause_min_seconds),
                "maximum_speech_speed": float(maximum_speech_speed),
            }
        )

    @classmethod
    def execute(
        cls,
        audio_batch: AudioBatch,
        strategy: str,
        tolerance_seconds: float,
        maximum_speed: float,
        fit_underrun: bool,
        minimum_speed: float,
        project_name: str,
        subfolder: str,
        edge_silence_threshold_db: float = -40.0,
        edge_silence_min_seconds: float = 0.05,
        edge_padding_seconds: float = 0.12,
        internal_pause_min_seconds: float = 0.18,
        maximum_speech_speed: float = 1.12,
    ) -> io.NodeOutput:
        project = _safe_name(project_name, "subtitle-fit")
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        clean_subfolder = subfolder.rstrip("/\\") or "fireredaudio/duration-fit"
        output_dir = _safe_output_dir(f"{clean_subfolder}/{project}-{stamp}")
        batch, report = fit_audio_batch_to_cues(
            audio_batch,
            output_dir,
            strategy=strategy,
            tolerance_seconds=tolerance_seconds,
            maximum_speed=maximum_speed,
            minimum_speed=minimum_speed,
            fit_underrun=fit_underrun,
            edge_silence_threshold_db=edge_silence_threshold_db,
            edge_silence_min_seconds=edge_silence_min_seconds,
            edge_padding_seconds=edge_padding_seconds,
            internal_pause_min_seconds=internal_pause_min_seconds,
            maximum_speech_speed=maximum_speech_speed,
        )
        manifest_path = output_dir / "duration-fit-manifest.json"
        payload = {
            "manifest_version": MANIFEST_VERSION,
            "kind": "duration_fit",
            "node_version": NODE_VERSION,
            **report,
            "items": list(batch.items),
        }
        write_manifest(manifest_path, payload)
        fitted = AudioBatch(str(manifest_path), batch.items)
        return io.NodeOutput(
            fitted,
            str(manifest_path),
            "\n".join(report["retry_line_ids"]),
            _json(report),
        )


class T8FireRedAudioLineReview(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_LineReview",
            display_name="FireRedAudio 逐句制作审核台 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description=(
                "逐行试听并合并自动 QA 与人工决定；输出通过批次、人工复核清单和定向重做清单。"
                "前端表格会把决定、评分和备注同步到可序列化 JSON 输入。"
            ),
            inputs=[
                AudioBatchType.Input("audio_batch", display_name="待审核 AudioBatch"),
                SpeechQAType.Input("qa", display_name="可选语音 QA", optional=True),
                io.String.Input(
                    "decisions_json",
                    display_name='决定 JSON：auto/approve/review/retry',
                    default="{}",
                    multiline=True,
                    advanced=True,
                ),
                io.String.Input(
                    "ratings_json",
                    display_name="评分 JSON（1–5）",
                    default="{}",
                    multiline=True,
                    advanced=True,
                ),
                io.String.Input(
                    "notes_json",
                    display_name="备注 JSON",
                    default="{}",
                    multiline=True,
                    advanced=True,
                ),
                io.String.Input("review_name", display_name="审核项目名", default="production-review"),
                io.String.Input("subfolder", display_name="审核记录目录", default="fireredaudio/line-reviews"),
                io.Int.Input("preview_limit", display_name="表格最多加载行数", default=40, min=1, max=200),
            ],
            outputs=[
                AudioBatchType.Output("reviewed_batch", display_name="带审核记录 AudioBatch"),
                AudioBatchType.Output("approved_batch", display_name="仅通过项 AudioBatch"),
                io.String.Output("retry_line_ids", display_name="建议重做 line ID"),
                io.String.Output("review_line_ids", display_name="待人工复核 line ID"),
                io.String.Output("review_manifest_path", display_name="审核 Manifest 路径"),
                io.String.Output("review_report", display_name="审核报告"),
            ],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        audio_batch: AudioBatch,
        decisions_json: str,
        ratings_json: str,
        notes_json: str,
        review_name: str,
        subfolder: str,
        preview_limit: int,
        qa: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        return stable_digest(
            {
                "audio_batch": _audio_batch_state(audio_batch),
                "qa": qa,
                "decisions_json": decisions_json,
                "ratings_json": ratings_json,
                "notes_json": notes_json,
                "review_name": review_name,
                "subfolder": subfolder,
                "preview_limit": int(preview_limit),
            }
        )

    @classmethod
    def execute(
        cls,
        audio_batch: AudioBatch,
        decisions_json: str,
        ratings_json: str,
        notes_json: str,
        review_name: str,
        subfolder: str,
        preview_limit: int,
        qa: dict[str, Any] | None = None,
    ) -> io.NodeOutput:
        decisions = parse_json_mapping(decisions_json, "决定 JSON")
        ratings = parse_json_mapping(ratings_json, "评分 JSON")
        notes = parse_json_mapping(notes_json, "备注 JSON")
        reviewed, approved, report = build_line_review(
            audio_batch,
            qa=qa,
            decisions=decisions,
            ratings=ratings,
            notes=notes,
        )
        project = _safe_name(review_name, "production-review")
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        clean_subfolder = subfolder.rstrip("/\\") or "fireredaudio/line-reviews"
        review_dir = _safe_output_dir(f"{clean_subfolder}/{project}-{stamp}")
        manifest_path = review_dir / "review-manifest.json"
        manifest_payload = {
            "manifest_version": MANIFEST_VERSION,
            "kind": "line_production_review",
            "node_version": NODE_VERSION,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            **{key: value for key, value in report.items() if key != "rows"},
            "items": list(reviewed.items),
        }
        write_manifest(manifest_path, manifest_payload)
        reviewed_batch = AudioBatch(str(manifest_path), reviewed.items)
        approved_batch = AudioBatch(str(manifest_path), approved.items)
        limit = max(1, min(200, int(preview_limit)))
        ui_rows: list[dict[str, Any]] = []
        audio_descriptors: list[dict[str, str]] = []
        for row in report["rows"][:limit]:
            copy = dict(row)
            path = Path(str(copy.get("output_path") or ""))
            descriptor = None
            if path.is_file():
                try:
                    descriptor = saved_audio_ui(path)["audio"][0]
                    audio_descriptors.append(descriptor)
                except ValueError:
                    descriptor = None
            copy["audio"] = descriptor
            ui_rows.append(copy)
        ui_payload = {
            "manifest_path": str(manifest_path),
            "source_manifest_path": audio_batch.manifest_path,
            "total": report["total"],
            "previewed": len(ui_rows),
            "rows": ui_rows,
        }
        ui: dict[str, Any] = {"fireredaudio_review": [ui_payload]}
        if audio_descriptors:
            ui["audio"] = audio_descriptors
        return io.NodeOutput(
            reviewed_batch,
            approved_batch,
            "\n".join(report["retry_line_ids"]),
            "\n".join(report["review_line_ids"]),
            str(manifest_path),
            _json({key: value for key, value in report.items() if key != "rows"}),
            ui=ui,
        )


class T8FireRedAudioBatchRetry(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_BatchRetry",
            display_name="FireRedAudio QA 失败项定向返修 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description=(
                "消费 SpeechQA 的失败 line ID，只返修失败台词；输出非破坏合并后的新 AudioBatch 和独立返修 manifest。"
            ),
            inputs=[
                ModelType.Input("model"),
                AudioBatchType.Input("audio_batch", display_name="原批量音频"),
                ScriptPlanType.Input("script_plan", display_name="原脚本计划"),
                VoiceBankType.Input("voice_bank", display_name="原音色库"),
                io.String.Input("failed_line_ids", display_name="失败 line ID", multiline=True, force_input=True),
                io.String.Input("project_name", display_name="返修项目名", default="qa-repair"),
                io.String.Input("subfolder", display_name="输出子目录", default="fireredaudio/repairs"),
                io.Combo.Input("seed_strategy", display_name="Seed 策略", options=["increment", "fixed"], default="increment"),
                io.Int.Input("seed_step", display_name="每次 Seed 增量", default=1, min=1, max=100000),
                io.Int.Input("max_attempts", display_name="最多尝试次数", default=2, min=1, max=5),
                io.Int.Input("batch_size", display_name="每批条数", default=8, min=1, max=32),
                SettingsType.Input("settings", display_name="生成参数", optional=True),
                io.Boolean.Input(
                    "enforce_cue_duration",
                    display_name="返修后仍须满足字幕时间槽",
                    default=True,
                    advanced=True,
                ),
                io.Float.Input(
                    "max_cue_overrun_seconds",
                    display_name="返修允许超出时间槽（秒）",
                    default=0.50,
                    min=0.0,
                    max=60.0,
                    step=0.1,
                    advanced=True,
                ),
            ],
            outputs=[
                AudioBatchType.Output("audio_batch", display_name="返修后批量音频"),
                io.String.Output("manifest_path", display_name="返修 Manifest 路径"),
                io.String.Output("repair_report", display_name="返修报告"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        model: RuntimeHandle,
        audio_batch: AudioBatch,
        script_plan: ScriptPlan,
        voice_bank: VoiceBank,
        failed_line_ids: str,
        project_name: str,
        subfolder: str,
        seed_strategy: str,
        seed_step: int,
        max_attempts: int,
        batch_size: int,
        settings: GenerationSettings | None = None,
        enforce_cue_duration: bool = True,
        max_cue_overrun_seconds: float = 0.50,
        **kwargs,
    ) -> str:
        return stable_digest(
            {
                "model": model.to_dict(),
                "audio_batch": _audio_batch_state(audio_batch),
                "script_plan": script_plan.to_dict() if isinstance(script_plan, ScriptPlan) else None,
                "voice_bank": voice_bank.to_dict() if isinstance(voice_bank, VoiceBank) else None,
                "failed_line_ids": parse_line_ids(failed_line_ids),
                "project_name": project_name,
                "subfolder": subfolder,
                "seed_strategy": seed_strategy,
                "seed_step": int(seed_step),
                "max_attempts": int(max_attempts),
                "batch_size": int(batch_size),
                "settings": _settings(settings).to_dict(),
                "enforce_cue_duration": bool(enforce_cue_duration),
                "max_cue_overrun_seconds": float(max_cue_overrun_seconds),
            }
        )

    @classmethod
    def execute(
        cls,
        model: RuntimeHandle,
        audio_batch: AudioBatch,
        script_plan: ScriptPlan,
        voice_bank: VoiceBank,
        failed_line_ids: str,
        project_name: str,
        subfolder: str,
        seed_strategy: str,
        seed_step: int,
        max_attempts: int,
        batch_size: int,
        settings: GenerationSettings | None = None,
        enforce_cue_duration: bool = True,
        max_cue_overrun_seconds: float = 0.50,
    ) -> io.NodeOutput:
        if not isinstance(audio_batch, AudioBatch):
            raise TypeError("定向返修必须连接原 AudioBatch")
        if not isinstance(script_plan, ScriptPlan) or not isinstance(voice_bank, VoiceBank):
            raise TypeError("定向返修必须连接原脚本计划和音色库")
        if not script_plan.valid:
            raise ValueError("原脚本计划预检存在错误，不能返修")
        target_ids = parse_line_ids(failed_line_ids)
        if not target_ids:
            report = {
                "manifest_path": audio_batch.manifest_path,
                "source_manifest_path": audio_batch.manifest_path,
                "requested": 0,
                "repaired": 0,
                "failed": 0,
                "repaired_line_ids": [],
                "failed_line_ids": [],
                "action": "passthrough_no_qa_failures",
                "source_files_overwritten": False,
            }
            return io.NodeOutput(audio_batch, audio_batch.manifest_path, _json(report))
        if seed_strategy not in {"increment", "fixed"}:
            raise ValueError(f"不支持的 Seed 策略：{seed_strategy}")
        original_by_id = {
            str(item.get("line_id") or ""): dict(item) for item in audio_batch.items
        }
        line_by_id = {line.line_id: line for line in script_plan.lines}
        unknown = [line_id for line_id in target_ids if line_id not in original_by_id or line_id not in line_by_id]
        if unknown:
            raise ValueError("失败 line ID 不属于当前脚本/AudioBatch：" + ", ".join(unknown))

        project = _safe_name(project_name, "qa-repair")
        clean_subfolder = subfolder.rstrip("/\\") or "fireredaudio/repairs"
        project_dir = _safe_output_dir(f"{clean_subfolder}/{project}")
        manifest_path = project_dir / "repair-manifest.json"
        base_settings = _settings(settings)
        base_config = base_settings.to_dict()
        effective_batch_size = max(1, min(32, int(batch_size)))
        effective_attempts = max(1, min(5, int(max_attempts)))
        cue_threshold = max(0.0, float(max_cue_overrun_seconds))
        replacements: dict[str, dict[str, Any]] = {}
        for line_id in target_ids:
            item = dict(original_by_id[line_id])
            item.update(
                status="repair_pending",
                original_output_path=str(item.get("output_path") or ""),
                repair_attempts=[],
            )
            replacements[line_id] = item

        def merged_items() -> list[dict[str, Any]]:
            return [
                replacements.get(str(item.get("line_id") or ""), dict(item))
                for item in audio_batch.items
            ]

        manifest_payload: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION,
            "project_name": project,
            "source_manifest_path": audio_batch.manifest_path,
            "target_line_ids": list(target_ids),
            "settings": base_config,
            "seed_strategy": seed_strategy,
            "seed_step": int(seed_step),
            "max_attempts": effective_attempts,
            "enforce_cue_duration": bool(enforce_cue_duration),
            "max_cue_overrun_seconds": cue_threshold,
            "items": merged_items(),
        }
        write_manifest(manifest_path, manifest_payload)
        remaining = list(target_ids)
        performance: list[dict[str, Any]] = []
        total_calls = 0
        cue_rejected: set[str] = set()
        duplicate_rejected: set[str] = set()
        try:
            for attempt in range(1, effective_attempts + 1):
                if not remaining:
                    break
                jobs: list[dict[str, Any]] = []
                attempt_seed = int(base_config.get("seed", 42))
                if seed_strategy == "increment":
                    attempt_seed += attempt * int(seed_step)
                for line_id in remaining:
                    line = line_by_id[line_id]
                    profile = voice_bank.resolve(line.speaker)
                    if profile is None:
                        raise ValueError(f"角色没有音色档案：{line.speaker}")
                    target = project_dir / (
                        _safe_name(
                            f"{line.index:04d}-{line.speaker}-{line.line_id}-repair-a{attempt:02d}",
                            f"line-{line.index:04d}-repair-a{attempt:02d}",
                        )
                        + ".wav"
                    )
                    request = _base_request(model, "tts", base_settings)
                    request.update(
                        {
                            "task_id": f"repair-{project}-{line.line_id}-a{attempt}",
                            "prompt_audio": profile.prompt_audio,
                            "prompt_text": profile.prompt_text,
                            "target_text": line.text,
                            "language": line.language,
                            "seed": attempt_seed,
                            "output_path": str(target),
                            "release_after": False,
                        }
                    )
                    original_path = Path(
                        str(original_by_id[line_id].get("output_path") or "")
                    )
                    original_sha256 = (
                        file_digest(original_path) if original_path.is_file() else ""
                    )
                    jobs.append(
                        {
                            "line_id": line_id,
                            "target": target,
                            "request": request,
                            "seed": attempt_seed,
                            "original_sha256": original_sha256,
                        }
                    )

                next_remaining: list[str] = []
                for chunk_start in range(0, len(jobs), effective_batch_size):
                    chunk = jobs[chunk_start : chunk_start + effective_batch_size]
                    total_calls += 1
                    try:
                        result = _infer_tts_batch(model, [dict(job["request"]) for job in chunk])
                        outcomes = {
                            int(outcome.get("index", -1)): outcome
                            for outcome in result.get("outcomes", [])
                            if isinstance(outcome, dict)
                        }
                        performance.append(dict(result.get("performance") or {}))
                    except BaseException as exc:
                        if _is_processing_interrupt(exc) or not isinstance(exc, Exception):
                            manifest_payload["items"] = merged_items()
                            write_manifest(manifest_path, manifest_payload)
                            raise
                        outcomes = {
                            index: {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                            for index in range(len(chunk))
                        }
                    for local_index, job in enumerate(chunk):
                        line_id = job["line_id"]
                        item = replacements[line_id]
                        outcome = outcomes.get(local_index)
                        attempt_record = {
                            "attempt": attempt,
                            "seed": job["seed"],
                            "output_path": str(job["target"]),
                        }
                        try:
                            if not outcome:
                                raise RuntimeError("Worker 批量结果缺少对应返修条目")
                            if not outcome.get("ok"):
                                raise RuntimeError(str(outcome.get("error") or "Worker 返修失败"))
                            worker_report = dict(outcome.get("result") or {})
                            actual = Path(str(worker_report.get("output_path") or job["target"]))
                            if not actual.is_file():
                                raise RuntimeError("Worker 未产生返修音频")
                            if actual.resolve() != job["target"]:
                                shutil.copy2(actual, job["target"])
                            output_sha256 = file_digest(job["target"])
                            attempt_record["output_sha256"] = output_sha256
                            if (
                                job.get("original_sha256")
                                and output_sha256 == job["original_sha256"]
                            ):
                                duplicate_rejected.add(line_id)
                                raise RuntimeError(
                                    "返修音频与原 Take 的 SHA-256 完全相同，未产生新候选"
                                )
                            line = line_by_id[line_id]
                            if (
                                line.start_seconds is not None
                                and line.end_seconds is not None
                            ):
                                cue_seconds = float(line.end_seconds) - float(
                                    line.start_seconds
                                )
                                duration_seconds = float(
                                    wav_metrics(job["target"])["duration_seconds"]
                                )
                                cue_overrun = max(0.0, duration_seconds - cue_seconds)
                                attempt_record.update(
                                    duration_seconds=duration_seconds,
                                    cue_seconds=cue_seconds,
                                    cue_overrun_seconds=cue_overrun,
                                )
                                if enforce_cue_duration and cue_overrun > cue_threshold:
                                    cue_rejected.add(line_id)
                                    raise RuntimeError(
                                        "返修音频仍超出字幕时间槽："
                                        f"{cue_overrun:.3f}s > {cue_threshold:.3f}s"
                                    )
                            attempt_record.update(status="complete", worker_report=worker_report)
                            item["repair_attempts"].append(attempt_record)
                            if isinstance(item.get("human_review"), dict):
                                item["previous_human_review"] = dict(item["human_review"])
                                item.pop("human_review", None)
                            item.update(
                                status="complete",
                                output_path=str(job["target"]),
                                cache_hit=False,
                                repaired=True,
                                repair_seed=job["seed"],
                                worker_report=worker_report,
                            )
                            item.pop("error", None)
                        except Exception as exc:
                            attempt_record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                            item["repair_attempts"].append(attempt_record)
                            item.update(status="repair_failed", error=attempt_record["error"])
                            next_remaining.append(line_id)
                        manifest_payload["items"] = merged_items()
                        write_manifest(manifest_path, manifest_payload)
                remaining = list(dict.fromkeys(next_remaining))
        finally:
            if model.release_after:
                try:
                    _client(model).unload()
                except Exception as exc:
                    LOGGER.warning("返修结束后释放 Worker 模型失败：%s", exc)

        for line_id in remaining:
            replacements[line_id]["status"] = "failed"
        merged_batch = merge_audio_batch_items(audio_batch, replacements.values(), manifest_path)
        manifest_payload["items"] = list(merged_batch.items)
        write_manifest(manifest_path, manifest_payload)
        repaired_ids = [line_id for line_id in target_ids if replacements[line_id].get("status") == "complete"]
        report = {
            "manifest_path": str(manifest_path),
            "source_manifest_path": audio_batch.manifest_path,
            "requested": len(target_ids),
            "repaired": len(repaired_ids),
            "failed": len(remaining),
            "repaired_line_ids": repaired_ids,
            "failed_line_ids": remaining,
            "seed_strategy": seed_strategy,
            "seed_step": int(seed_step),
            "max_attempts": effective_attempts,
            "batch_size": effective_batch_size,
            "enforce_cue_duration": bool(enforce_cue_duration),
            "max_cue_overrun_seconds": cue_threshold,
            "cue_rejected_line_ids": sorted(cue_rejected),
            "duplicate_rejected_line_ids": sorted(duplicate_rejected),
            "worker_batch_calls": total_calls,
            "performance": performance,
            "source_files_overwritten": False,
        }
        return io.NodeOutput(merged_batch, str(manifest_path), _json(report))


class T8FireRedAudioCreativeCandidatePool(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_CreativeCandidatePool",
            display_name="FireRedAudio 单句创意候选池 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description=(
                "为一条指定台词生成 2–7 个不同 Seed 的创意候选，并可把原 Take 一起匿名复制到候选池。"
                "它不自动覆盖成品；请连接多 Take 试听评审板，再用候选采用节点明确回填。"
            ),
            inputs=[
                ModelType.Input("model"),
                AudioBatchType.Input("audio_batch", display_name="原批量音频"),
                ScriptPlanType.Input("script_plan", display_name="原脚本计划"),
                VoiceBankType.Input("voice_bank", display_name="原音色库"),
                io.String.Input(
                    "target_line_id",
                    display_name="要探索的单个 line ID",
                    multiline=True,
                    force_input=True,
                ),
                io.Int.Input("candidate_count", display_name="新候选数量", default=3, min=2, max=7),
                io.Int.Input("seed_start", display_name="起始 Seed", default=1001, min=0, max=0xFFFFFFFF - 700000),
                io.Int.Input("seed_step", display_name="候选 Seed 间隔", default=97, min=1, max=100000),
                io.Boolean.Input("include_original", display_name="把原 Take 匿名加入盲听", default=True),
                io.Boolean.Input("run_asr_qa", display_name="逐个 ASR 回读（较慢）", default=False),
                io.String.Input("project_name", display_name="候选池项目名", default="creative-line-candidates"),
                io.String.Input("subfolder", display_name="输出子目录", default="fireredaudio/candidates"),
                SettingsType.Input("settings", display_name="生成参数", optional=True),
                io.Float.Input(
                    "minimum_acoustic_difference",
                    display_name="候选声学差异预筛阈值",
                    default=0.005,
                    min=0.0,
                    max=1.0,
                    step=0.005,
                    advanced=True,
                ),
            ],
            outputs=[
                AudioBatchType.Output("candidate_batch", display_name="匿名候选 AudioBatch"),
                io.String.Output("source_line_id", display_name="原 line ID"),
                io.String.Output("manifest_path", display_name="候选 Manifest 路径"),
                io.String.Output("candidate_report", display_name="Seed 与候选证据"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        model: RuntimeHandle,
        audio_batch: AudioBatch,
        script_plan: ScriptPlan,
        voice_bank: VoiceBank,
        target_line_id: str,
        candidate_count: int,
        seed_start: int,
        seed_step: int,
        include_original: bool,
        run_asr_qa: bool,
        project_name: str,
        subfolder: str,
        settings: GenerationSettings | None = None,
        minimum_acoustic_difference: float = 0.005,
        **kwargs,
    ) -> str:
        return stable_digest(
            {
                "model": model.to_dict(),
                "audio_batch": _audio_batch_state(audio_batch),
                "script_plan": script_plan.to_dict() if isinstance(script_plan, ScriptPlan) else None,
                "voice_bank": voice_bank.to_dict() if isinstance(voice_bank, VoiceBank) else None,
                "target_line_id": parse_line_ids(target_line_id),
                "candidate_count": int(candidate_count),
                "seed_start": int(seed_start),
                "seed_step": int(seed_step),
                "include_original": bool(include_original),
                "run_asr_qa": bool(run_asr_qa),
                "project_name": project_name,
                "subfolder": subfolder,
                "settings": _settings(settings).to_dict(),
                "minimum_acoustic_difference": float(minimum_acoustic_difference),
            }
        )

    @classmethod
    def execute(
        cls,
        model: RuntimeHandle,
        audio_batch: AudioBatch,
        script_plan: ScriptPlan,
        voice_bank: VoiceBank,
        target_line_id: str,
        candidate_count: int,
        seed_start: int,
        seed_step: int,
        include_original: bool,
        run_asr_qa: bool,
        project_name: str,
        subfolder: str,
        settings: GenerationSettings | None = None,
        minimum_acoustic_difference: float = 0.005,
    ) -> io.NodeOutput:
        if not isinstance(audio_batch, AudioBatch):
            raise TypeError("创意候选池必须连接原 AudioBatch")
        if not isinstance(script_plan, ScriptPlan) or not isinstance(voice_bank, VoiceBank):
            raise TypeError("创意候选池必须连接原脚本计划和音色库")
        if not 0.0 <= float(minimum_acoustic_difference) <= 1.0:
            raise ValueError("候选声学差异预筛阈值必须在 0–1")
        requested = parse_line_ids(target_line_id)
        if len(requested) != 1:
            raise ValueError("创意候选池一次只处理一个 line ID；多条失败项请逐句试听决定")
        source_line_id = requested[0]
        source_items = {
            str(item.get("line_id") or ""): dict(item) for item in audio_batch.items
        }
        lines = {line.line_id: line for line in script_plan.lines}
        if source_line_id not in source_items or source_line_id not in lines:
            raise ValueError(f"line ID 不属于当前脚本/AudioBatch：{source_line_id}")
        line = lines[source_line_id]
        profile = voice_bank.resolve(line.speaker)
        if profile is None:
            raise ValueError(f"角色没有音色档案：{line.speaker}")

        project = _safe_name(project_name, "creative-line-candidates")
        clean_subfolder = subfolder.rstrip("/\\") or "fireredaudio/candidates"
        project_dir = _safe_output_dir(f"{clean_subfolder}/{project}-{uuid.uuid4().hex[:8]}")
        config = _settings(settings)
        effective_count = max(2, min(7, int(candidate_count)))
        effective_step = max(1, int(seed_step))
        requests: list[dict[str, Any]] = []
        request_seeds: list[int] = []
        for offset in range(effective_count):
            seed = int(seed_start) + offset * effective_step
            request_seeds.append(seed)
            request = _base_request(model, "tts", config)
            request.update(
                {
                    "task_id": f"creative-{project}-{source_line_id}-{offset + 1:03d}",
                    "seed": seed,
                    "prompt_audio": profile.prompt_audio,
                    "prompt_text": profile.prompt_text,
                    "target_text": line.text,
                    "language": line.language,
                    "output_path": str(
                        project_dir / f"blind-{uuid.uuid4().hex[:12]}.wav"
                    ),
                    "release_after": False,
                }
            )
            requests.append(request)

        batch_result = _infer_tts_batch(model, requests)
        items: list[dict[str, Any]] = []
        original = source_items[source_line_id]
        if include_original:
            original_path = Path(str(original.get("output_path") or ""))
            if original.get("status") == "complete" and original_path.is_file():
                blind_original = project_dir / f"blind-{uuid.uuid4().hex[:12]}.wav"
                shutil.copy2(original_path, blind_original)
                items.append(
                    {
                        "line_id": "candidate-000",
                        "index": 0,
                        "source_line_id": source_line_id,
                        "speaker": line.speaker,
                        "text": line.text,
                        "language": line.language,
                        "status": "complete",
                        "output_path": str(blind_original),
                        "candidate_origin": "original_take",
                        "seed": original.get("seed"),
                        "output_sha256": file_digest(blind_original),
                        "metrics": wav_metrics(blind_original),
                    }
                )

        for outcome in batch_result.get("outcomes", []):
            index = int(outcome.get("index", 0))
            candidate_id = f"candidate-{index + 1:03d}"
            if not outcome.get("ok"):
                items.append(
                    {
                        "line_id": candidate_id,
                        "index": index + 1,
                        "source_line_id": source_line_id,
                        "status": "failed",
                        "requested_seed": requests[index]["seed"],
                        "error": outcome.get("error"),
                    }
                )
                continue
            worker_report = dict(outcome.get("result") or {})
            path = Path(str(worker_report.get("output_path") or requests[index]["output_path"]))
            if not path.is_file():
                raise RuntimeError(f"Worker 未产生创意候选：{candidate_id}")
            metrics = wav_metrics(path)
            item: dict[str, Any] = {
                "line_id": candidate_id,
                "index": index + 1,
                "source_line_id": source_line_id,
                "speaker": line.speaker,
                "text": line.text,
                "language": line.language,
                "status": "complete",
                "output_path": str(path),
                "candidate_origin": "creative_seed",
                "seed": requests[index]["seed"],
                "requested_seed": requests[index]["seed"],
                "output_sha256": file_digest(path),
                "metrics": metrics,
                "worker_report": worker_report,
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
                metric_name, error_rate = text_error_rate(line.text, hypothesis, line.language)
                item["asr_qa"] = {
                    "hypothesis": hypothesis,
                    "metric": metric_name,
                    "error_rate": error_rate,
                }
            items.append(item)

        playable = [item for item in items if item.get("status") == "complete"]
        if not playable:
            raise RuntimeError("单句创意候选池没有生成任何可试听候选")
        by_hash: dict[str, list[str]] = {}
        signatures: dict[str, dict[str, Any]] = {}
        for item in playable:
            by_hash.setdefault(str(item.get("output_sha256") or ""), []).append(str(item["line_id"]))
            signatures[str(item["line_id"])] = wav_acoustic_signature(
                str(item["output_path"])
            )
        duplicate_groups = [group for digest, group in by_hash.items() if digest and len(group) > 1]
        pairwise_acoustic_evidence: list[dict[str, Any]] = []
        acoustic_near_duplicate_pairs: list[list[str]] = []
        for left_index, left in enumerate(playable):
            left_id = str(left["line_id"])
            for right in playable[left_index + 1 :]:
                right_id = str(right["line_id"])
                evidence = acoustic_signature_distance(
                    signatures[left_id], signatures[right_id]
                )
                pairwise_acoustic_evidence.append(
                    {"left": left_id, "right": right_id, **evidence}
                )
                if evidence["score"] < float(minimum_acoustic_difference):
                    acoustic_near_duplicate_pairs.append([left_id, right_id])
        manifest_path = project_dir / "candidate-manifest.json"
        report = {
            "manifest_version": MANIFEST_VERSION,
            "kind": "creative_line_candidate_pool",
            "node_version": NODE_VERSION,
            "source_manifest_path": audio_batch.manifest_path,
            "source_line_id": source_line_id,
            "requested_seeds": request_seeds,
            "seed_step": effective_step,
            "include_original": bool(include_original),
            "generated_count": len([item for item in playable if item.get("candidate_origin") == "creative_seed"]),
            "playable_count": len(playable),
            "distinct_audio_hashes": len(by_hash),
            "duplicate_candidate_groups": duplicate_groups,
            "minimum_acoustic_difference": float(minimum_acoustic_difference),
            "pairwise_acoustic_evidence": pairwise_acoustic_evidence,
            "acoustic_near_duplicate_pairs": acoustic_near_duplicate_pairs,
            "diversity_prefilter_passed": not duplicate_groups
            and not acoustic_near_duplicate_pairs,
            "diversity_evidence_scope": (
                "自动指标仅用于发现明显重复；是否存在人耳可辨的表演差异必须匿名盲听"
            ),
            "human_listening_required": True,
            "blind_filenames": True,
            "automatic_adoption": False,
            "performance": batch_result.get("performance"),
            "items": items,
        }
        write_manifest(manifest_path, report)
        return io.NodeOutput(
            AudioBatch(str(manifest_path), tuple(items)),
            source_line_id,
            str(manifest_path),
            _json(report),
        )


class T8FireRedAudioCandidateApply(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_CandidateApply",
            display_name="FireRedAudio 采用创意候选 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description=(
                "把多 Take 试听评审板明确选中的一条创意候选非破坏地回填到原 AudioBatch，"
                "记录旧文件、Seed、人工评分和候选 Manifest，未选台词保持不变。"
            ),
            inputs=[
                AudioBatchType.Input("source_audio_batch", display_name="原批量音频"),
                AudioBatchType.Input("reviewed_candidates", display_name="已评审候选"),
                io.String.Input("selected_candidate_id", display_name="已采用候选 ID", force_input=True),
                io.String.Input("project_name", display_name="采用记录名", default="creative-candidate-adoption"),
                io.String.Input("subfolder", display_name="采用记录目录", default="fireredaudio/candidate-adoptions"),
            ],
            outputs=[
                AudioBatchType.Output("audio_batch", display_name="回填后 AudioBatch"),
                io.Audio.Output("selected_audio", display_name="已采用音频"),
                io.String.Output("manifest_path", display_name="采用 Manifest 路径"),
                io.String.Output("adoption_report", display_name="采用报告"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        source_audio_batch: AudioBatch,
        reviewed_candidates: AudioBatch,
        selected_candidate_id: str,
        project_name: str,
        subfolder: str,
        **kwargs,
    ) -> str:
        return stable_digest(
            {
                "source_audio_batch": _audio_batch_state(source_audio_batch),
                "reviewed_candidates": _audio_batch_state(reviewed_candidates),
                "selected_candidate_id": selected_candidate_id,
                "project_name": project_name,
                "subfolder": subfolder,
            }
        )

    @classmethod
    def execute(
        cls,
        source_audio_batch: AudioBatch,
        reviewed_candidates: AudioBatch,
        selected_candidate_id: str,
        project_name: str,
        subfolder: str,
    ) -> io.NodeOutput:
        if not isinstance(source_audio_batch, AudioBatch) or not isinstance(reviewed_candidates, AudioBatch):
            raise TypeError("候选采用必须连接原 AudioBatch 和已评审候选")
        selected_id = str(selected_candidate_id or "").strip()
        if not selected_id:
            raise ValueError("请从多 Take 试听评审板连接已采用候选 ID")
        selected = next(
            (dict(item) for item in reviewed_candidates.items if str(item.get("line_id") or "") == selected_id),
            None,
        )
        if selected is None or selected.get("status") != "complete":
            raise ValueError(f"候选不存在或不可播放：{selected_id}")
        selected_path = Path(str(selected.get("output_path") or ""))
        if not selected_path.is_file():
            raise ValueError(f"候选音频文件不存在：{selected_path}")
        source_line_id = str(selected.get("source_line_id") or "").strip()
        originals = {str(item.get("line_id") or ""): dict(item) for item in source_audio_batch.items}
        if not source_line_id or source_line_id not in originals:
            raise ValueError("候选没有可回填的 source_line_id，或它不属于原 AudioBatch")
        original = originals[source_line_id]
        replacement = dict(original)
        replacement.update(
            {
                "status": "complete",
                "output_path": str(selected_path),
                "creative_candidate_adopted": True,
                "candidate_line_id": selected_id,
                "candidate_seed": selected.get("seed"),
                "candidate_origin": selected.get("candidate_origin"),
                "candidate_manifest_path": reviewed_candidates.manifest_path,
                "previous_output_path": str(original.get("output_path") or ""),
                "human_review": dict(selected.get("human_review") or {}),
            }
        )
        project = _safe_name(project_name, "creative-candidate-adoption")
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
        clean_subfolder = subfolder.rstrip("/\\") or "fireredaudio/candidate-adoptions"
        output_dir = _safe_output_dir(f"{clean_subfolder}/{project}-{stamp}")
        manifest_path = output_dir / "adoption-manifest.json"
        merged = merge_audio_batch_items(source_audio_batch, [replacement], manifest_path)
        report = {
            "manifest_version": MANIFEST_VERSION,
            "kind": "creative_candidate_adoption",
            "node_version": NODE_VERSION,
            "source_manifest_path": source_audio_batch.manifest_path,
            "candidate_manifest_path": reviewed_candidates.manifest_path,
            "source_line_id": source_line_id,
            "selected_candidate_id": selected_id,
            "selected_seed": selected.get("seed"),
            "previous_output_path": str(original.get("output_path") or ""),
            "selected_output_path": str(selected_path),
            "source_files_overwritten": False,
            "items": list(merged.items),
        }
        write_manifest(manifest_path, report)
        return io.NodeOutput(merged, wav_to_audio(selected_path), str(manifest_path), _json(report))


class T8FireRedAudioAudioBatchSelect(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_AudioBatchSelect",
            display_name="FireRedAudio AudioBatch 试听选择 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="按成功序号、line ID 或角色选出一条音频，返回原生 AUDIO、条目详情和批次摘要。",
            inputs=[
                AudioBatchType.Input("audio_batch", display_name="批量音频"),
                io.Combo.Input("selection_mode", display_name="选择方式", options=["position", "line_id", "speaker"], default="position"),
                io.Int.Input("position", display_name="成功音频序号（从 1 开始）", default=1, min=1, max=100000),
                io.String.Input("line_id", display_name="line ID", default="", optional=True),
                io.String.Input("speaker", display_name="角色", default="", optional=True),
            ],
            outputs=[
                io.Audio.Output("audio", display_name="选中音频"),
                io.String.Output("item_json", display_name="选中条目 JSON"),
                io.String.Output("selected_line_id", display_name="选中 line ID"),
                io.String.Output("batch_summary", display_name="批次摘要"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        audio_batch: AudioBatch,
        selection_mode: str,
        position: int,
        line_id: str,
        speaker: str,
        **kwargs,
    ) -> str:
        return stable_digest(
            {
                "audio_batch": _audio_batch_state(audio_batch),
                "selection_mode": selection_mode,
                "position": int(position),
                "line_id": line_id,
                "speaker": speaker,
            }
        )

    @classmethod
    def execute(
        cls,
        audio_batch: AudioBatch,
        selection_mode: str,
        position: int,
        line_id: str,
        speaker: str,
    ) -> io.NodeOutput:
        item = select_audio_batch_item(
            audio_batch,
            mode=selection_mode,
            position=position,
            line_id=line_id,
            speaker=speaker,
        )
        path = Path(str(item["output_path"]))
        summary = {
            "manifest_path": audio_batch.manifest_path,
            "total": len(audio_batch.items),
            "playable": len(audio_batch.successful_items()),
            "selected_line_id": item.get("line_id"),
            "selected_speaker": item.get("speaker"),
            "selected_path": str(path),
        }
        ui: dict[str, Any] = {}
        try:
            ui = saved_audio_ui(path)
        except ValueError:
            pass
        return io.NodeOutput(
            wav_to_audio(path),
            _json(item),
            str(item.get("line_id") or ""),
            _json(summary),
            ui=ui,
        )


class T8FireRedAudioTakeReviewBoard(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_TakeReviewBoard",
            display_name="FireRedAudio 多 Take 试听评审板 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description=(
                "在结果区显示最多 8 条原生播放器/下载入口，记录 1–5 分和备注，"
                "显式采用一条 Take；评审只写新 Manifest，不改源音频。"
            ),
            inputs=[
                AudioBatchType.Input("audio_batch", display_name="候选 AudioBatch"),
                io.Int.Input(
                    "selected_position",
                    display_name="采用序号（0 = 仅盲听）",
                    default=0,
                    min=0,
                    max=8,
                ),
                io.String.Input("selected_line_id", display_name="采用 line ID（优先）", default="", optional=True),
                io.String.Input(
                    "ratings_json",
                    display_name='评分 JSON，例如 {"take-001": 5}',
                    default="{}",
                    multiline=True,
                ),
                io.String.Input(
                    "notes_json",
                    display_name='备注 JSON，例如 {"take-001": "自然"}',
                    default="{}",
                    multiline=True,
                ),
                io.String.Input("review_name", display_name="评审名称", default="take-review"),
                io.String.Input("subfolder", display_name="评审记录目录", default="fireredaudio/reviews"),
                io.Int.Input("preview_limit", display_name="试听数量", default=8, min=2, max=8),
            ],
            outputs=[
                io.Audio.Output("selected_audio", display_name="已采用 Take"),
                AudioBatchType.Output("reviewed_batch", display_name="带评审记录 AudioBatch"),
                io.String.Output("selected_line_id", display_name="已采用 line ID"),
                io.String.Output("review_manifest_path", display_name="评审 Manifest 路径"),
                io.String.Output("review_report", display_name="评审报告"),
            ],
            is_output_node=True,
        )

    @staticmethod
    def _mapping(value: str, label: str) -> dict[str, Any]:
        try:
            parsed = json.loads(str(value or "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label}不是有效 JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{label}必须是 JSON 对象")
        return {str(key): item for key, item in parsed.items()}

    @classmethod
    def fingerprint_inputs(
        cls,
        audio_batch: AudioBatch,
        selected_position: int,
        selected_line_id: str,
        ratings_json: str,
        notes_json: str,
        review_name: str,
        subfolder: str,
        preview_limit: int,
        **kwargs,
    ) -> str:
        return stable_digest(
            {
                "audio_batch": _audio_batch_state(audio_batch),
                "selected_position": int(selected_position),
                "selected_line_id": selected_line_id,
                "ratings_json": ratings_json,
                "notes_json": notes_json,
                "review_name": review_name,
                "subfolder": subfolder,
                "preview_limit": int(preview_limit),
            }
        )

    @classmethod
    def execute(
        cls,
        audio_batch: AudioBatch,
        selected_position: int,
        selected_line_id: str,
        ratings_json: str,
        notes_json: str,
        review_name: str,
        subfolder: str,
        preview_limit: int,
    ) -> io.NodeOutput:
        if not isinstance(audio_batch, AudioBatch):
            raise TypeError("试听评审板必须连接 AudioBatch")
        playable = audio_batch.successful_items()
        if not playable:
            raise ValueError("AudioBatch 中没有可试听候选")
        preview_count = max(2, min(8, int(preview_limit), len(playable)))
        blind_order = sorted(
            playable,
            key=lambda item: stable_digest(
                {
                    "manifest": audio_batch.manifest_path,
                    "sha256": item.get("output_sha256")
                    or file_digest(str(item["output_path"])),
                    "line_id": item.get("line_id"),
                }
            ),
        )
        previewed = blind_order[:preview_count]
        ratings = cls._mapping(ratings_json, "评分 JSON")
        notes = cls._mapping(notes_json, "备注 JSON")
        known_ids = {str(item.get("line_id") or "") for item in playable}
        unknown = sorted((set(ratings) | set(notes)) - known_ids)
        if unknown:
            raise ValueError("评分或备注包含未知 line ID：" + ", ".join(unknown))
        normalized_ratings: dict[str, float] = {}
        for line_id, value in ratings.items():
            try:
                score = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{line_id} 的评分必须是 1–5") from exc
            if not 1.0 <= score <= 5.0:
                raise ValueError(f"{line_id} 的评分必须是 1–5")
            normalized_ratings[line_id] = round(score, 2)
        normalized_notes = {line_id: str(value).strip() for line_id, value in notes.items()}
        requested_line_id = str(selected_line_id).strip()
        selected: dict[str, Any] | None = None
        if requested_line_id:
            selected = next(
                (
                    dict(item)
                    for item in playable
                    if str(item.get("line_id") or "") == requested_line_id
                ),
                None,
            )
            if selected is None:
                raise ValueError(f"没有找到要采用的候选：{requested_line_id}")
        elif int(selected_position) > 0:
            position = int(selected_position)
            if position > len(previewed):
                raise ValueError(f"盲听采用序号超出范围：1–{len(previewed)}")
            selected = dict(previewed[position - 1])
        selected_id = str(selected.get("line_id") or "") if selected else ""
        preview_audio = selected or dict(previewed[0])
        reviewed_items: list[dict[str, Any]] = []
        for item in audio_batch.items:
            copy = dict(item)
            line_id = str(copy.get("line_id") or "")
            copy["human_review"] = {
                "rating": normalized_ratings.get(line_id),
                "note": normalized_notes.get(line_id, ""),
                "adopted": line_id == selected_id,
            }
            reviewed_items.append(copy)
        project = _safe_name(review_name, "take-review")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        clean_subfolder = subfolder.rstrip("/\\") or "fireredaudio/reviews"
        review_dir = _safe_output_dir(f"{clean_subfolder}/{project}-{stamp}")
        manifest_path = review_dir / "review-manifest.json"
        report = {
            "manifest_version": MANIFEST_VERSION,
            "kind": "take_human_review",
            "node_version": NODE_VERSION,
            "source_manifest_path": audio_batch.manifest_path,
            "selected_line_id": selected_id,
            "selection_required": not bool(selected_id),
            "ratings": normalized_ratings,
            "notes": normalized_notes,
            "playable_count": len(playable),
            "previewed_count": len(previewed),
            "blind_order": [
                {
                    "blind_label": chr(65 + index),
                    "line_id": str(item.get("line_id") or ""),
                }
                for index, item in enumerate(previewed)
            ],
            "blind_order_applied": True,
            "source_files_overwritten": False,
            "items": reviewed_items,
        }
        write_manifest(manifest_path, report)
        reviewed_batch = AudioBatch(str(manifest_path), tuple(reviewed_items))
        ui: dict[str, Any] = {}
        try:
            descriptors = saved_audio_files_ui(
                [item["output_path"] for item in previewed]
            )["audio"]
            ui = {
                "audio": descriptors,
                "fireredaudio_take_review": [
                    {
                        "selection_required": not bool(selected_id),
                        "selected_line_id": selected_id,
                        "rows": [
                            {
                                "blind_label": chr(65 + index),
                                "line_id": str(item.get("line_id") or ""),
                                "audio": descriptors[index],
                                "rating": normalized_ratings.get(
                                    str(item.get("line_id") or "")
                                ),
                                "note": normalized_notes.get(
                                    str(item.get("line_id") or ""), ""
                                ),
                            }
                            for index, item in enumerate(previewed)
                        ],
                    }
                ],
            }
        except ValueError:
            pass
        return io.NodeOutput(
            wav_to_audio(preview_audio["output_path"]),
            reviewed_batch,
            selected_id,
            str(manifest_path),
            _json(report),
            ui=ui,
        )


class T8FireRedAudioSaveAudioBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_SaveAudioBatch",
            display_name="FireRedAudio 批量保存/下载 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description="把成功 Take 批量复制或转码到 ComfyUI output，生成便携 Manifest 和可选 ZIP，并注册原生试听/下载列表。",
            inputs=[
                AudioBatchType.Input("audio_batch", display_name="批量音频"),
                io.Combo.Input("audio_format", display_name="格式", options=["wav", "flac", "mp3", "ogg"], default="wav"),
                io.String.Input("project_name", display_name="导出项目名", default="fireredaudio-batch"),
                io.String.Input("subfolder", display_name="输出子目录", default="fireredaudio/exports"),
                io.Boolean.Input("create_zip", display_name="创建 ZIP 制作包", default=True),
                io.Boolean.Input("continue_on_error", display_name="单条导出失败后继续", default=True),
                io.Int.Input("preview_limit", display_name="结果区最多试听条数", default=16, min=1, max=100),
            ],
            outputs=[
                AudioBatchType.Output("audio_batch", display_name="原 AudioBatch（透传）"),
                io.String.Output("manifest_path", display_name="导出 Manifest 路径"),
                io.String.Output("zip_path", display_name="ZIP 路径"),
                io.String.Output("export_report", display_name="导出报告"),
            ],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        audio_batch: AudioBatch,
        audio_format: str,
        project_name: str,
        subfolder: str,
        create_zip: bool,
        continue_on_error: bool,
        preview_limit: int,
        **kwargs,
    ) -> str:
        return stable_digest(
            {
                "audio_batch": _audio_batch_state(audio_batch),
                "audio_format": audio_format,
                "project_name": project_name,
                "subfolder": subfolder,
                "create_zip": create_zip,
                "continue_on_error": continue_on_error,
                "preview_limit": int(preview_limit),
            }
        )

    @classmethod
    def execute(
        cls,
        audio_batch: AudioBatch,
        audio_format: str,
        project_name: str,
        subfolder: str,
        create_zip: bool,
        continue_on_error: bool,
        preview_limit: int,
    ) -> io.NodeOutput:
        if not isinstance(audio_batch, AudioBatch):
            raise TypeError("批量保存必须连接 AudioBatch")
        source_items = audio_batch.successful_items()
        if not source_items:
            raise ValueError("AudioBatch 中没有可以保存的成功音频")
        extension = str(audio_format).lower()
        if extension not in {"wav", "flac", "mp3", "ogg"}:
            raise ValueError("audio_format 必须是 wav/flac/mp3/ogg")
        project = _safe_name(project_name, "fireredaudio-batch")
        now = datetime.now().astimezone()
        stamp = now.strftime("%Y%m%d-%H%M%S-%f")
        clean_subfolder = subfolder.rstrip("/\\") or "fireredaudio/exports"
        export_dir = _safe_output_dir(f"{clean_subfolder}/{project}-{stamp}")
        saved_items: list[dict[str, Any]] = []
        saved_paths: list[Path] = []
        failures: list[dict[str, str]] = []
        used_names: set[str] = set()
        for position, source_item in enumerate(source_items, 1):
            source = Path(str(source_item.get("output_path") or ""))
            base = _safe_name(
                f"{int(source_item.get('index') or position):04d}-{source_item.get('speaker') or 'take'}-{source_item.get('line_id') or position}",
                f"take-{position:04d}",
            )
            candidate = base
            suffix = 2
            while candidate.casefold() in used_names:
                candidate = f"{base}-{suffix}"
                suffix += 1
            used_names.add(candidate.casefold())
            target = export_dir / f"{candidate}.{extension}"
            item = dict(source_item)
            try:
                export_audio_path(source, target, audio_format=extension)
                item.update(
                    output_path=str(target),
                    source_output_path=str(source),
                    export_sha256=file_digest(target),
                    export_format=extension,
                    status="complete",
                )
                saved_paths.append(target)
            except Exception as exc:
                failure = {
                    "line_id": str(source_item.get("line_id") or position),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(failure)
                item.update(status="failed", error=failure["error"], source_output_path=str(source))
                if not continue_on_error:
                    raise
            saved_items.append(item)
            _set_official_progress(position / max(1, len(source_items)))
        manifest_path = export_dir / "export-manifest.json"
        manifest_payload = {
            "manifest_version": MANIFEST_VERSION,
            "project_name": project,
            "source_manifest_path": audio_batch.manifest_path,
            "audio_format": extension,
            "created_at": now.isoformat(timespec="seconds"),
            "items": saved_items,
        }
        write_manifest(manifest_path, manifest_payload)
        zip_path: Path | None = None
        if create_zip:
            zip_path = export_dir.parent / f"{export_dir.name}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(export_dir.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(export_dir).as_posix())
        preview_count = max(1, min(100, int(preview_limit)))
        report = {
            "manifest_path": str(manifest_path),
            "zip_path": str(zip_path) if zip_path else "",
            "source_count": len(source_items),
            "saved": len(saved_paths),
            "failed": len(failures),
            "failures": failures,
            "audio_format": extension,
            "previewed": min(len(saved_paths), preview_count),
            "output_directory": str(export_dir),
            "source_files_overwritten": False,
        }
        return io.NodeOutput(
            audio_batch,
            str(manifest_path),
            str(zip_path) if zip_path else "",
            _json(report),
            ui=saved_audio_files_ui(saved_paths[:preview_count]),
        )


class T8FireRedAudioProductionPackage(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_ProductionPackage",
            display_name="FireRedAudio 分轨制作交付包 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description=(
                "按角色和场景导出等长对白分轨，并打包 Master、dialogue、room tone、BGM、"
                "实际 SRT/VTT、版本、参数、素材哈希和便携 Manifest。"
            ),
            inputs=[
                AudioBatchType.Input("audio_batch", display_name="批量音频"),
                io.String.Input("project_name", display_name="交付项目名", default="fireredaudio-production"),
                io.String.Input("subfolder", display_name="输出子目录", default="fireredaudio/deliveries"),
                io.Int.Input("sample_rate", display_name="制作采样率", default=48000, min=8000, max=192000),
                io.Int.Input("crossfade_ms", display_name="Master 对白交叉淡化（毫秒）", default=40, min=0, max=2000),
                io.Boolean.Input("include_role_stems", display_name="导出角色分轨", default=True),
                io.Boolean.Input("include_scene_stems", display_name="导出场景分轨", default=True),
                io.Boolean.Input("create_zip", display_name="创建一键交付 ZIP", default=True),
                io.Audio.Input("master_audio", display_name="可选外部 Master", optional=True),
                io.Audio.Input("bgm_audio", display_name="可选 BGM", optional=True),
                io.Audio.Input("room_tone_audio", display_name="可选 room tone", optional=True),
                io.String.Input("source_subtitles", display_name="可选原始字幕", multiline=True, force_input=True, optional=True),
                DeliveryPresetType.Input("delivery_preset", display_name="可选交付预设", optional=True),
            ],
            outputs=[
                AudioBatchType.Output("audio_batch", display_name="原 AudioBatch（透传）"),
                io.Audio.Output("master_audio", display_name="交付 Master"),
                io.String.Output("manifest_path", display_name="制作 Manifest 路径"),
                io.String.Output("zip_path", display_name="交付 ZIP 路径"),
                io.String.Output("package_report", display_name="制作包报告"),
            ],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        audio_batch: AudioBatch,
        project_name: str,
        subfolder: str,
        sample_rate: int,
        crossfade_ms: int,
        include_role_stems: bool,
        include_scene_stems: bool,
        create_zip: bool,
        master_audio: dict | None = None,
        bgm_audio: dict | None = None,
        room_tone_audio: dict | None = None,
        source_subtitles: str = "",
        delivery_preset: DeliveryPreset | None = None,
        **kwargs,
    ) -> str:
        def audio_digest(value: dict | None, label: str) -> str | None:
            return file_digest(audio_to_wav(value, label)) if value is not None else None

        return stable_digest(
            {
                "audio_batch": _audio_batch_state(audio_batch),
                "project_name": project_name,
                "subfolder": subfolder,
                "sample_rate": int(sample_rate),
                "crossfade_ms": int(crossfade_ms),
                "include_role_stems": bool(include_role_stems),
                "include_scene_stems": bool(include_scene_stems),
                "create_zip": bool(create_zip),
                "master": audio_digest(master_audio, "production-master"),
                "bgm": audio_digest(bgm_audio, "production-bgm"),
                "room_tone": audio_digest(room_tone_audio, "production-room-tone"),
                "source_subtitles": str(source_subtitles or ""),
                "delivery_preset": delivery_preset.to_dict() if delivery_preset else None,
            }
        )

    @classmethod
    def execute(
        cls,
        audio_batch: AudioBatch,
        project_name: str,
        subfolder: str,
        sample_rate: int,
        crossfade_ms: int,
        include_role_stems: bool,
        include_scene_stems: bool,
        create_zip: bool,
        master_audio: dict | None = None,
        bgm_audio: dict | None = None,
        room_tone_audio: dict | None = None,
        source_subtitles: str = "",
        delivery_preset: DeliveryPreset | None = None,
    ) -> io.NodeOutput:
        if not isinstance(audio_batch, AudioBatch):
            raise TypeError("分轨制作包必须连接 AudioBatch")
        if delivery_preset is not None and not isinstance(delivery_preset, DeliveryPreset):
            raise TypeError("交付预设输入类型无效")
        source_items = audio_batch.successful_items()
        if not source_items:
            raise ValueError("AudioBatch 中没有可以制作交付包的成功音频")
        source_hashes_before = {
            str(item.get("line_id") or index): file_digest(item["output_path"])
            for index, item in enumerate(source_items, 1)
        }
        project = _safe_name(project_name, "fireredaudio-production")
        now = datetime.now().astimezone()
        stamp = now.strftime("%Y%m%d-%H%M%S-%f")
        clean_subfolder = subfolder.rstrip("/\\") or "fireredaudio/deliveries"
        export_dir = _safe_output_dir(f"{clean_subfolder}/{project}-{stamp}")
        mix_dir = export_dir / "mix"
        stems_dir = export_dir / "stems"
        assets_dir = export_dir / "assets"
        subtitles_dir = export_dir / "subtitles"
        for directory in (mix_dir, stems_dir, assets_dir, subtitles_dir):
            directory.mkdir(parents=True, exist_ok=True)

        effective_rate = delivery_preset.sample_rate if delivery_preset else int(sample_rate)
        effective_crossfade = delivery_preset.crossfade_ms if delivery_preset else int(crossfade_ms)
        has_timing = any(item.get("start_seconds") is not None for item in source_items)
        effective_mode = delivery_preset.mode if delivery_preset else ("timeline" if has_timing else "sequence")
        dialogue_path = mix_dir / "dialogue-master.wav"
        dialogue_report = render_timeline_to_wav(
            source_items,
            dialogue_path,
            mode=effective_mode,
            gap_ms=(delivery_preset.gap_ms if delivery_preset else 0),
            crossfade_ms=effective_crossfade,
            peak_policy="limit",
            sample_rate=effective_rate,
            target_lufs=(delivery_preset.target_lufs if delivery_preset else None),
            loudness_range_lu=(delivery_preset.loudness_range_lu if delivery_preset else 7.0),
            true_peak_dbfs=(delivery_preset.true_peak_dbfs if delivery_preset else -1.0),
        )
        total_frames = round(float(dialogue_report["duration_seconds"]) * effective_rate)
        stem_reports: list[dict[str, Any]] = []
        if include_role_stems:
            stem_reports.extend(
                render_grouped_stems(
                    source_items,
                    dialogue_report["placements"],
                    stems_dir / "roles",
                    group_key="speaker",
                    filename_prefix="role",
                    sample_rate=effective_rate,
                    total_frames=total_frames,
                )
            )
        if include_scene_stems:
            stem_reports.extend(
                render_grouped_stems(
                    source_items,
                    dialogue_report["placements"],
                    stems_dir / "scenes",
                    group_key="scene",
                    filename_prefix="scene",
                    sample_rate=effective_rate,
                    total_frames=total_frames,
                )
            )

        master_target = mix_dir / "master.wav"
        if master_audio is not None:
            master_source = audio_to_wav(master_audio, "production-master")
            export_audio_path(master_source, master_target, audio_format="wav")
            master_origin = "connected_master_audio"
        else:
            shutil.copy2(dialogue_path, master_target)
            master_origin = "rendered_dialogue_master"

        production_assets: list[dict[str, Any]] = []
        for kind, value, filename in (
            ("bgm", bgm_audio, "bgm.wav"),
            ("room_tone", room_tone_audio, "room-tone.wav"),
        ):
            if value is None:
                continue
            source = audio_to_wav(value, f"production-{kind}")
            target = assets_dir / filename
            export_audio_path(source, target, audio_format="wav")
            production_assets.append(
                {
                    "kind": kind,
                    "path": target.relative_to(export_dir).as_posix(),
                    "sha256": file_digest(target),
                    "mixed_into_master_by_this_node": False,
                }
            )

        srt, vtt, cues = build_batch_subtitles(source_items, dialogue_report["placements"])
        srt_path = subtitles_dir / f"{project}.srt"
        vtt_path = subtitles_dir / f"{project}.vtt"
        srt_path.write_text(srt, encoding="utf-8")
        vtt_path.write_text(vtt, encoding="utf-8")
        original_subtitle_path: Path | None = None
        if str(source_subtitles or "").strip():
            extension = "vtt" if str(source_subtitles).lstrip().upper().startswith("WEBVTT") else "srt"
            original_subtitle_path = subtitles_dir / f"source-original.{extension}"
            original_subtitle_path.write_text(str(source_subtitles), encoding="utf-8")

        source_manifest = None
        source_manifest_path = Path(str(audio_batch.manifest_path or ""))
        if source_manifest_path.is_file():
            try:
                source_manifest = load_manifest(source_manifest_path)
            except ValueError:
                # ProjectExchange and evidence batches legitimately use another JSON contract.
                source_manifest = None
        definition = manifest()
        worker_code_revisions = sorted(
            {
                str((item.get("worker_report") or {}).get("code_revision"))
                for item in source_items
                if (item.get("worker_report") or {}).get("code_revision")
            }
        )
        worker_model_revisions = sorted(
            {
                str((item.get("worker_report") or {}).get("model_revision"))
                for item in source_items
                if (item.get("worker_report") or {}).get("model_revision")
            }
        )
        generated_files = sorted(path for path in export_dir.rglob("*") if path.is_file())
        file_inventory = [
            {
                "path": path.relative_to(export_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_digest(path),
            }
            for path in generated_files
        ]
        manifest_path = export_dir / "production-manifest.json"
        manifest_payload = {
            "manifest_version": MANIFEST_VERSION,
            "package_schema": "t8.firered.production-package.v1",
            "project_name": project,
            "created_at": now.isoformat(timespec="seconds"),
            "source_manifest_path": audio_batch.manifest_path,
            "source_manifest_sha256": (
                file_digest(source_manifest_path) if source_manifest_path.is_file() else None
            ),
            "versions": {
                "comfyui_node": NODE_VERSION,
                "upstream_code_revision": definition["codeRevision"],
                "upstream_model_revision": definition["modelRevision"],
                "worker_code_revisions": worker_code_revisions,
                "worker_model_revisions": worker_model_revisions,
            },
            "generation": {
                "settings": (source_manifest or {}).get("settings"),
                "model_identity": (source_manifest or {}).get("model_identity"),
                "delivery_preset": delivery_preset.to_dict() if delivery_preset else None,
                "sample_rate": effective_rate,
                "timeline_mode": effective_mode,
                "crossfade_ms": effective_crossfade,
            },
            "master": {
                "path": master_target.relative_to(export_dir).as_posix(),
                "sha256": file_digest(master_target),
                "origin": master_origin,
            },
            "dialogue_master": {
                "path": dialogue_path.relative_to(export_dir).as_posix(),
                "sha256": file_digest(dialogue_path),
                "render_report": dialogue_report,
            },
            "stems": [
                {
                    **report,
                    "path": Path(report["path"]).relative_to(export_dir).as_posix(),
                }
                for report in stem_reports
            ],
            "production_assets": production_assets,
            "subtitles": {
                "srt": srt_path.relative_to(export_dir).as_posix(),
                "vtt": vtt_path.relative_to(export_dir).as_posix(),
                "source_original": (
                    original_subtitle_path.relative_to(export_dir).as_posix()
                    if original_subtitle_path is not None
                    else None
                ),
                "cue_count": len(cues),
                "timing": "actual_render_placements",
            },
            "source_items": [
                {
                    "line_id": item.get("line_id"),
                    "speaker": item.get("speaker"),
                    "scene": item.get("scene") or "",
                    "text": item.get("text"),
                    "sha256": source_hashes_before[str(item.get("line_id") or index)],
                }
                for index, item in enumerate(source_items, 1)
            ],
            "files": file_inventory,
        }
        write_manifest(manifest_path, manifest_payload)
        source_hashes_after = {
            str(item.get("line_id") or index): file_digest(item["output_path"])
            for index, item in enumerate(source_items, 1)
        }
        sources_preserved = source_hashes_before == source_hashes_after
        if not sources_preserved:
            raise RuntimeError("制作包导出过程中源音频发生变化，已停止交付")

        zip_path: Path | None = None
        if create_zip:
            zip_path = export_dir.parent / f"{export_dir.name}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(export_dir.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(export_dir).as_posix())
        preview_paths = [master_target, dialogue_path] + [Path(item["path"]) for item in stem_reports]
        preview_paths.extend(assets_dir / Path(item["path"]).name for item in production_assets)
        report = {
            "manifest_path": str(manifest_path),
            "zip_path": str(zip_path) if zip_path else "",
            "output_directory": str(export_dir),
            "master_origin": master_origin,
            "role_stems": sum(1 for item in stem_reports if item["group_key"] == "speaker"),
            "scene_stems": sum(1 for item in stem_reports if item["group_key"] == "scene"),
            "production_assets": len(production_assets),
            "subtitle_cues": len(cues),
            "source_files_overwritten": False,
            "source_hashes_preserved": sources_preserved,
        }
        return io.NodeOutput(
            audio_batch,
            wav_to_audio(master_target),
            str(manifest_path),
            str(zip_path) if zip_path else "",
            _json(report),
            ui=saved_audio_files_ui(preview_paths[:32]),
        )


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


class T8FireRedAudioAccelerationBenchmark(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="T8_FireRedAudio_AccelerationBenchmark",
            display_name="FireRedAudio 加速实测向导 · T8star-Aix",
            category=CATEGORY,
            essentials_category="Audio",
            description=(
                "固定参考、文本、Seed 和参数，对 off/FlashAttention/DeepSpeed 等模式"
                "先暖机再测量中位耗时、RTF、峰值显存和输出哈希；只给建议，不修改模型设置。"
            ),
            inputs=[
                ModelType.Input("model"),
                io.Audio.Input("prompt_audio", display_name="固定参考音频"),
                io.String.Input("prompt_text", display_name="固定参考逐字稿", multiline=True, dynamic_prompts=True),
                io.String.Input("target_text", display_name="固定目标文本", multiline=True, dynamic_prompts=True),
                io.Combo.Input("language", display_name="语言", options=["zh", "en"], default="zh"),
                io.String.Input("modes", display_name="实测模式（逗号分隔）", default="off,flash_attention,deepspeed"),
                io.Int.Input("warmup_runs", display_name="每模式暖机次数", default=1, min=1, max=3),
                io.Int.Input("measure_runs", display_name="每模式正式次数", default=3, min=3, max=20),
                io.Float.Input("minimum_improvement_percent", display_name="最低有效提速百分比", default=10.0, min=0.0, max=100.0, step=1.0),
                io.Boolean.Input("require_reproducible_hash", display_name="推荐模式必须同 Seed 哈希一致", default=True),
                io.String.Input("project_name", display_name="基准项目名", default="acceleration-benchmark"),
                io.String.Input("subfolder", display_name="输出子目录", default="fireredaudio/benchmarks"),
                SettingsType.Input("settings", display_name="固定生成参数", optional=True),
            ],
            outputs=[
                AudioBatchType.Output("benchmark_outputs", display_name="全部正式测量音频"),
                io.String.Output("recommendation", display_name="加速建议"),
                io.String.Output("benchmark_report", display_name="完整基准报告"),
                io.String.Output("manifest_path", display_name="基准 Manifest 路径"),
            ],
            is_output_node=True,
        )

    @staticmethod
    def _modes(value: str) -> list[str]:
        allowed = {"off", "auto_safe", "flash_attention", "deepspeed", "fla_liger", "torch_compile"}
        result: list[str] = []
        for raw in str(value or "").replace("，", ",").split(","):
            mode = raw.strip().lower()
            if not mode:
                continue
            if mode not in allowed:
                raise ValueError(f"未知加速模式：{mode}")
            if mode not in result:
                result.append(mode)
        if "off" not in result:
            result.insert(0, "off")
        if len(result) < 2:
            raise ValueError("加速实测至少需要 off 和一个候选模式")
        return result

    @classmethod
    def validate_inputs(cls, prompt_text: str, target_text: str, modes: str, **kwargs) -> bool | str:
        if not str(prompt_text).strip():
            return "加速实测必须填写固定参考逐字稿，避免把 ASR 时间混入 TTS 基准。"
        if not str(target_text).strip():
            return "加速实测必须填写固定目标文本。"
        try:
            cls._modes(modes)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def fingerprint_inputs(
        cls,
        model: RuntimeHandle,
        prompt_audio: dict,
        prompt_text: str,
        target_text: str,
        language: str,
        modes: str,
        warmup_runs: int,
        measure_runs: int,
        minimum_improvement_percent: float,
        require_reproducible_hash: bool,
        project_name: str,
        subfolder: str,
        settings: GenerationSettings | None = None,
        **kwargs,
    ) -> str:
        reference = audio_to_wav(prompt_audio, "benchmark-reference")
        return stable_digest(
            {
                "model": model.to_dict(),
                "reference_sha256": file_digest(reference),
                "prompt_text": prompt_text,
                "target_text": target_text,
                "language": language,
                "modes": cls._modes(modes),
                "warmup_runs": int(warmup_runs),
                "measure_runs": int(measure_runs),
                "minimum_improvement_percent": float(minimum_improvement_percent),
                "require_reproducible_hash": bool(require_reproducible_hash),
                "project_name": project_name,
                "subfolder": subfolder,
                "settings": _settings(settings).to_dict(),
            }
        )

    @classmethod
    def execute(
        cls,
        model: RuntimeHandle,
        prompt_audio: dict,
        prompt_text: str,
        target_text: str,
        language: str,
        modes: str,
        warmup_runs: int,
        measure_runs: int,
        minimum_improvement_percent: float,
        require_reproducible_hash: bool,
        project_name: str,
        subfolder: str,
        settings: GenerationSettings | None = None,
    ) -> io.NodeOutput:
        requested_modes = cls._modes(modes)
        reference = audio_to_wav(prompt_audio, "benchmark-reference")
        project = _safe_name(project_name, "acceleration-benchmark")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        clean_subfolder = subfolder.rstrip("/\\") or "fireredaudio/benchmarks"
        benchmark_dir = _safe_output_dir(f"{clean_subfolder}/{project}-{stamp}")
        config = _settings(settings)
        mode_reports: list[dict[str, Any]] = []
        output_items: list[dict[str, Any]] = []
        preview_paths: list[str] = []
        total_iterations = len(requested_modes) * (int(warmup_runs) + int(measure_runs))
        completed_iterations = 0
        for mode in requested_modes:
            handle = replace(model, acceleration_mode=mode, release_after=False)
            mode_report: dict[str, Any] = {
                "requested_mode": mode,
                "warmup_runs": int(warmup_runs),
                "measure_runs_requested": int(measure_runs),
                "runs": [],
                "status": "pending",
            }
            try:
                try:
                    _client(handle).unload()
                except Exception:
                    pass
                for iteration in range(int(warmup_runs) + int(measure_runs)):
                    warmup = iteration < int(warmup_runs)
                    formal_index = iteration - int(warmup_runs) + 1
                    suffix = f"warmup-{iteration + 1}" if warmup else f"run-{formal_index}"
                    output = benchmark_dir / f"{mode}-{suffix}.wav"
                    request = _base_request(handle, "tts", config)
                    request.update(
                        {
                            "task_id": f"benchmark-{mode}-{suffix}-{uuid.uuid4().hex[:8]}",
                            "prompt_audio": str(reference),
                            "prompt_text": str(prompt_text).strip(),
                            "target_text": str(target_text).strip(),
                            "language": language,
                            "output_path": str(output),
                            "release_after": False,
                        }
                    )
                    started = time.perf_counter()
                    result = _infer(handle, request)
                    wall_seconds = time.perf_counter() - started
                    completed_iterations += 1
                    _set_official_progress(completed_iterations / max(1, total_iterations))
                    if warmup:
                        output.unlink(missing_ok=True)
                        Path(f"{output}.json").unlink(missing_ok=True)
                        continue
                    performance = dict(result.get("performance") or {})
                    health = _client(handle).health()
                    selection = (
                        ((health.get("acceleration") or {}).get("selection") or {})
                        if isinstance(health, dict)
                        else {}
                    )
                    metrics = wav_metrics(output)
                    digest = file_digest(output)
                    run = {
                        "run": formal_index,
                        "wall_seconds": round(wall_seconds, 6),
                        "total_seconds": round(float(performance.get("total_seconds") or wall_seconds), 6),
                        "rtf": float(performance.get("rtf") or 0.0),
                        "peak_vram_bytes": int(performance.get("gpu_peak_allocated_bytes") or 0),
                        "output_sha256": digest,
                        "output_path": str(output),
                        "output_duration_seconds": metrics["duration_seconds"],
                        "performance": performance,
                    }
                    mode_report["runs"].append(run)
                    item = {
                        "line_id": f"{mode}-run-{formal_index}",
                        "index": len(output_items) + 1,
                        "speaker": mode,
                        "text": target_text,
                        "language": language,
                        "status": "complete",
                        "output_path": str(output),
                        "output_sha256": digest,
                        "metrics": metrics,
                        "benchmark": run,
                    }
                    output_items.append(item)
                    if not preview_paths or formal_index == 1:
                        preview_paths.append(str(output))
                    mode_report["acceleration_selection"] = selection
                runs = mode_report["runs"]
                totals = [float(run["total_seconds"]) for run in runs]
                rtfs = [float(run["rtf"]) for run in runs if float(run["rtf"]) > 0]
                hashes = [str(run["output_sha256"]) for run in runs]
                effective = str((mode_report.get("acceleration_selection") or {}).get("effective") or mode)
                available = bool((mode_report.get("acceleration_selection") or {}).get("available", True))
                mode_report.update(
                    status="complete",
                    effective_mode=effective,
                    available=available,
                    median_total_seconds=round(statistics.median(totals), 6),
                    median_rtf=round(statistics.median(rtfs), 6) if rtfs else 0.0,
                    peak_vram_bytes=max(int(run["peak_vram_bytes"]) for run in runs),
                    output_hashes=hashes,
                    reproducible_hash=len(set(hashes)) == 1,
                    fallback_detected=effective != mode,
                )
            except BaseException as exc:
                if _is_processing_interrupt(exc):
                    raise
                mode_report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            mode_reports.append(mode_report)
        try:
            _client(model).unload()
        except Exception:
            pass
        completed = {item["requested_mode"]: item for item in mode_reports if item.get("status") == "complete"}
        baseline = completed.get("off")
        candidates: list[dict[str, Any]] = []
        if baseline:
            baseline_seconds = float(baseline["median_total_seconds"])
            for mode in requested_modes:
                if mode == "off" or mode not in completed:
                    continue
                item = completed[mode]
                improvement = 100.0 * (
                    baseline_seconds - float(item["median_total_seconds"])
                ) / max(baseline_seconds, 0.000001)
                threshold = max(
                    float(minimum_improvement_percent),
                    20.0 if mode in {"deepspeed", "fla_liger", "torch_compile"} else 0.0,
                )
                eligible = bool(
                    item.get("available")
                    and not item.get("fallback_detected")
                    and improvement >= threshold
                    and (not require_reproducible_hash or item.get("reproducible_hash"))
                )
                item.update(
                    improvement_percent=round(improvement, 3),
                    recommendation_threshold_percent=round(threshold, 3),
                    eligible_for_recommendation=eligible,
                )
                if eligible:
                    candidates.append(item)
        recommended_mode = "off"
        recommendation_reason = "没有候选模式达到实测提速、可用性与复现门槛，建议继续使用 off。"
        if candidates:
            winner = min(candidates, key=lambda item: float(item["median_total_seconds"]))
            recommended_mode = str(winner["requested_mode"])
            recommendation_reason = (
                f"{recommended_mode} 相比 off 的中位总耗时实测改善 "
                f"{float(winner['improvement_percent']):.3f}%，满足 "
                f"{float(winner['recommendation_threshold_percent']):.3f}% 门槛。"
            )
        elif baseline is None:
            recommended_mode = "none"
            recommendation_reason = "off 基线失败，无法给出可信加速建议。"
        recommendation = (
            f"建议模式：{recommended_mode}\n{recommendation_reason}\n"
            "本节点没有修改模型加载器或工作流设置；请人工把建议填回模型加载器。"
        )
        manifest_path = benchmark_dir / "benchmark-manifest.json"
        report = {
            "manifest_version": MANIFEST_VERSION,
            "kind": "acceleration_benchmark",
            "node_version": NODE_VERSION,
            "model": model.to_dict(),
            "reference_sha256": file_digest(reference),
            "prompt_text": str(prompt_text).strip(),
            "target_text": str(target_text).strip(),
            "language": language,
            "settings": config.to_dict(),
            "requested_modes": requested_modes,
            "warmup_runs": int(warmup_runs),
            "measure_runs": int(measure_runs),
            "minimum_improvement_percent": float(minimum_improvement_percent),
            "experimental_mode_minimum_percent": 20.0,
            "require_reproducible_hash": bool(require_reproducible_hash),
            "recommended_mode": recommended_mode,
            "recommendation_reason": recommendation_reason,
            "settings_modified": False,
            "mode_reports": mode_reports,
            "items": output_items,
        }
        write_manifest(manifest_path, report)
        batch = AudioBatch(str(manifest_path), tuple(output_items))
        ui: dict[str, Any] = {}
        try:
            ui = saved_audio_files_ui(preview_paths)
        except ValueError:
            pass
        return io.NodeOutput(batch, recommendation, _json(report), str(manifest_path), ui=ui)


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
            T8FireRedAudioLocalRepairRange,
            T8FireRedAudioLocalRepairApply,
            T8FireRedAudioReferenceCandidates,
            T8FireRedAudioReferenceQuality,
            T8FireRedAudioPrepareReference,
            T8FireRedAudioProjectExchange,
            T8FireRedAudioAudioBatchResume,
            T8FireRedAudioVoiceProfile,
            T8FireRedAudioVoiceBank,
            T8FireRedAudioScriptParser,
            T8FireRedAudioTextNormalizer,
            T8FireRedAudioBatchDubbing,
            T8FireRedAudioSynchronizedAB,
            T8FireRedAudioTimelineRender,
            T8FireRedAudioSpeechQA,
            T8FireRedAudioDurationFit,
            T8FireRedAudioLineReview,
            T8FireRedAudioBatchRetry,
            T8FireRedAudioCreativeCandidatePool,
            T8FireRedAudioCandidateApply,
            T8FireRedAudioAudioBatchSelect,
            T8FireRedAudioTakeReviewBoard,
            T8FireRedAudioSaveAudioBatch,
            T8FireRedAudioProductionPackage,
            T8FireRedAudioSaveAudio,
            T8FireRedAudioSaveSubtitle,
            T8FireRedAudioRuntimeControl,
            T8FireRedAudioPerformanceReport,
            T8FireRedAudioAccelerationBenchmark,
            T8FireRedAudioEnvironment,
        ]


async def comfy_entrypoint() -> T8FireRedAudioExtension:
    return T8FireRedAudioExtension()


__all__ = [name for name in globals() if name.startswith("T8FireRedAudio") or name == "comfy_entrypoint"]
