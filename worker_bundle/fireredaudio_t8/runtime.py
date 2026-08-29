from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import logging
import threading
import time
import tempfile
import uuid
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .constants import ALL_TASKS, CODE_REVISION, MODEL_REVISION, RUNTIME_VERSION
from .audio_inputs import prepare_audio_path
from .cache_manager import cache_status, cleanup_cache
from .errors import TaskCancelledError, WorkerProtocolError
from .long_audio import (
    deduplicate_segment_texts,
    render_jsonl,
    render_srt,
    render_vtt,
    split_pcm16_wav,
)
from .model_manager import (
    model_package_info,
    model_paths,
    normalize_model_root,
    profile_for_task,
    validate_model_dir,
)
from .presets import apply_quality_preset
from .system_info import gpu_inventory
from fireredaudio.acceleration import probe_acceleration, resolve_acceleration

logger = logging.getLogger(__name__)


@dataclass
class RuntimeState:
    loaded: bool = False
    model_root: str | None = None
    device: str | None = None
    decoder_loaded: bool = False
    memory_mode: str | None = None
    acceleration_mode: str | None = None
    loading: bool = False
    active_task: str | None = None
    active_task_id: str | None = None
    phase: str = "idle"
    progress: float = 0.0
    progress_message: str | None = None
    cancel_requested: bool = False
    completed_tasks: int = 0
    last_error: str | None = None
    task_started_at: str | None = None
    phase_started_at: str | None = None
    phase_elapsed_seconds: float = 0.0
    phase_timings: dict[str, float] = field(default_factory=dict)
    cold_start: bool | None = None
    quantization_profile: str | None = None
    quantization_format: str | None = None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class FireRedAudioRuntime:
    """Owns exactly one upstream inference engine and serializes access to it."""

    def __init__(self) -> None:
        self._engine: Any | None = None
        self._lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._cancel_event = threading.Event()
        self._state = RuntimeState()
        self._trace_started: float | None = None
        self._trace_phase: str | None = None
        self._trace_phase_started: float | None = None
        self._trace_phase_started_at: str | None = None
        self._trace_timings: dict[str, float] = {}

    def status(self) -> dict[str, Any]:
        # Status must remain readable while the model lock is held by a long inference.
        with self._state_lock:
            data = asdict(self._state)
        packages = {
            name: _package_version(name)
            for name in (
                "torch",
                "torchaudio",
                "torchcodec",
                "transformers",
                "numpy",
                "einops",
                "flash-attn",
                "flash-linear-attention",
                "liger-kernel",
                "triton-windows",
                "deepspeed",
                "torchao",
                "comfy-kitchen",
            )
        }
        requested_acceleration = data.get("acceleration_mode") or "auto_safe"
        capabilities = probe_acceleration(data.get("device") or None)
        selection = (
            dict(getattr(self._engine, "acceleration", {}) or {})
            if self._engine is not None
            else resolve_acceleration(
                requested_acceleration, data.get("device") or None, capabilities
            ).to_dict()
        )
        reference_cache = (
            self._engine.reference_cache_status()
            if self._engine is not None
            and hasattr(self._engine, "reference_cache_status")
            else {
                "enabled": True,
                "capacity": 4,
                "audio_entries": 0,
                "condition_entries": 0,
                "cpu_bytes": 0,
                "audio_hits": 0,
                "audio_misses": 0,
                "condition_hits": 0,
                "condition_misses": 0,
                "invalidations": 0,
                "evictions": 0,
                "condition_encode_seconds": 0.0,
            }
        )
        data.update(
            {
                "runtime_version": RUNTIME_VERSION,
                "code_revision": CODE_REVISION,
                "model_revision": MODEL_REVISION,
                "reference_cache": reference_cache,
                "packages": packages,
                "acceleration": {
                    "selection": selection,
                    "attention_backend": selection.get("attention_backend", "sdpa"),
                    "flash_linear_attention": bool(selection.get("use_fla")),
                    "fla_full_fast_path": bool(
                        capabilities["modules"].get("fla")
                        and capabilities["modules"].get("causal_conv1d")
                    ),
                    "liger_kernel": bool(selection.get("use_liger")),
                    "torch_compile": bool(selection.get("use_torch_compile")),
                    "latent_first_batch": True,
                    "deepspeed": {
                        "installed": bool(packages["deepspeed"]),
                        "supported": bool(packages["deepspeed"]),
                        "enabled": bool(selection.get("use_deepspeed")),
                        "single_gpu_only": True,
                        "reason": selection.get("reason", ""),
                    },
                    "capabilities": capabilities,
                    "transformers_isolation": {
                        "worker_version": packages["transformers"],
                        "host_independent": True,
                    },
                },
                "gpus": gpu_inventory(),
                "model_quantization": (
                    dict(
                        getattr(
                            getattr(self._engine, "model", None),
                            "_fireredaudio_quantization",
                            {},
                        )
                        or {}
                    )
                    if self._engine is not None
                    else {}
                ),
            }
        )
        return data

    def _update_state(self, **changes: Any) -> None:
        with self._state_lock:
            for key, value in changes.items():
                setattr(self._state, key, value)

    def _progress(self, phase: str, progress: float, message: str) -> None:
        now = time.perf_counter()
        if self._trace_started is not None:
            if self._trace_phase is not None and self._trace_phase != phase:
                assert self._trace_phase_started is not None
                elapsed = max(0.0, now - self._trace_phase_started)
                self._trace_timings[self._trace_phase] = (
                    self._trace_timings.get(self._trace_phase, 0.0) + elapsed
                )
                self._trace_phase_started = now
                self._trace_phase_started_at = datetime.now(timezone.utc).isoformat()
            elif self._trace_phase is None:
                self._trace_phase_started = now
                self._trace_phase_started_at = datetime.now(timezone.utc).isoformat()
            self._trace_phase = phase
        snapshot = self._trace_snapshot(now)
        self._update_state(
            phase=phase,
            progress=max(0.0, min(1.0, float(progress))),
            progress_message=message,
            phase_started_at=self._trace_phase_started_at,
            phase_elapsed_seconds=snapshot["phase_elapsed_seconds"],
            phase_timings=snapshot["phase_timings"],
        )

    def _begin_trace(self, *, cold_start: bool) -> None:
        now = time.perf_counter()
        self._trace_started = now
        self._trace_phase = None
        self._trace_phase_started = None
        self._trace_phase_started_at = None
        self._trace_timings = {}
        self._update_state(
            task_started_at=datetime.now(timezone.utc).isoformat(),
            phase_started_at=None,
            phase_elapsed_seconds=0.0,
            phase_timings={},
            cold_start=cold_start,
        )

    def _trace_snapshot(self, now: float | None = None) -> dict[str, Any]:
        current = time.perf_counter() if now is None else now
        timings = dict(self._trace_timings)
        phase_elapsed = 0.0
        if self._trace_phase is not None and self._trace_phase_started is not None:
            phase_elapsed = max(0.0, current - self._trace_phase_started)
            timings[self._trace_phase] = timings.get(self._trace_phase, 0.0) + phase_elapsed
        return {
            "phase_elapsed_seconds": round(phase_elapsed, 3),
            "phase_timings": {
                key: round(value, 3) for key, value in sorted(timings.items())
            },
        }

    def _reset_trace(self) -> None:
        self._trace_started = None
        self._trace_phase = None
        self._trace_phase_started = None
        self._trace_phase_started_at = None
        self._trace_timings = {}

    def _check_cancel(self) -> None:
        if self._cancel_event.is_set():
            raise TaskCancelledError("任务已由用户取消")

    def cancel(self, task_id: str | None = None) -> dict[str, Any]:
        """Signal cancellation without waiting for the long-running model lock."""
        with self._state_lock:
            active_id = self._state.active_task_id
            active_task = self._state.active_task
            if active_task is None:
                return {"cancel_requested": False, "reason": "当前没有运行中的任务"}
            if task_id and active_id and str(task_id) != active_id:
                return {
                    "cancel_requested": False,
                    "reason": "task_id 与当前任务不匹配",
                    "active_task_id": active_id,
                }
            self._state.cancel_requested = True
            self._state.phase = "cancelling"
            self._state.progress_message = "正在取消任务"
        self._cancel_event.set()
        return {
            "cancel_requested": True,
            "active_task": active_task,
            "active_task_id": active_id,
        }

    def _load(
        self,
        model_root: str | Path,
        device: str,
        require_decoder: bool,
        memory_mode: str = "auto",
        acceleration_mode: str = "auto_safe",
    ) -> None:
        root = normalize_model_root(model_root)
        profile = "full" if require_decoder else "lite"
        validate_model_dir(root, profile=profile).require_valid()
        # Only enter the visible loading phase after model files have passed
        # validation. Missing models therefore fail during "validating".
        self._progress("loading", 0.02, "正在加载模型")
        requested_root = str(root)
        requested_memory_mode = str(memory_mode or "auto").strip().lower()
        if (
            requested_memory_mode == "auto"
            and self._engine is not None
            and self._state.model_root == requested_root
            and self._state.device == device
            and self._state.acceleration_mode == acceleration_mode
            and (self._state.decoder_loaded or not require_decoder)
        ):
            # AUTO is a load-time decision. Recomputing it while the already-loaded
            # model occupies VRAM makes the next task falsely see low free memory,
            # unload a healthy full-GPU engine, and reload it as sequential.
            return
        package = model_package_info(root)
        quantization = dict(package.get("quantization") or {})
        resolved_memory_mode = _resolve_memory_mode(
            requested_memory_mode,
            device,
            require_decoder,
            recommended_min_vram_bytes=package.get("recommended_min_vram_bytes"),
        )
        if (
            self._engine is not None
            and self._state.model_root == requested_root
            and self._state.device == device
            and self._state.acceleration_mode == acceleration_mode
            and (
                self._state.memory_mode == resolved_memory_mode
                or (not require_decoder and self._state.decoder_loaded)
            )
            and (self._state.decoder_loaded or not require_decoder)
        ):
            return

        self._unload_locked()
        self._update_state(loading=True, last_error=None)
        try:
            from inference import FireRedAudioInference

            main_model, decoder = model_paths(root)
            self._engine = FireRedAudioInference(
                model_path=str(main_model),
                vae_decoder_path=str(decoder) if require_decoder else None,
                device=device,
                memory_mode=resolved_memory_mode,
                acceleration_mode=acceleration_mode,
            )
            self._update_state(
                loaded=True,
                model_root=requested_root,
                device=device,
                decoder_loaded=require_decoder,
                memory_mode=resolved_memory_mode,
                acceleration_mode=acceleration_mode,
                quantization_profile=str(quantization.get("profile") or "unknown"),
                quantization_format=str(quantization.get("format") or "unknown"),
            )
        except Exception as exc:
            self._update_state(last_error=str(exc))
            self._unload_locked(preserve_error=True)
            raise
        finally:
            self._update_state(loading=False)

    def load(
        self,
        model_root: str | Path,
        *,
        device: str = "auto",
        profile: str = "full",
        memory_mode: str = "auto",
        acceleration_mode: str = "auto_safe",
    ) -> dict[str, Any]:
        with self._lock:
            self._load(
                model_root,
                _resolve_device(device),
                profile == "full",
                memory_mode,
                acceleration_mode,
            )
            return self.status()

    def unload(self) -> dict[str, Any]:
        with self._lock:
            self._unload_locked()
            return self.status()

    def cache_status(self) -> dict[str, Any]:
        return cache_status()

    def cleanup_cache(
        self, *, max_age_hours: float = 72.0, max_size_mib: float = 2048.0, clear_all: bool = False
    ) -> dict[str, Any]:
        # The model lock prevents deletion while a task is reading a decoded input.
        with self._lock:
            return cleanup_cache(
                max_age_hours=max_age_hours,
                max_size_mib=max_size_mib,
                clear_all=clear_all,
            )

    def _unload_locked(self, preserve_error: bool = False) -> None:
        with self._state_lock:
            previous_error = self._state.last_error if preserve_error else None
            completed_tasks = self._state.completed_tasks
            active = {
                "active_task": self._state.active_task,
                "active_task_id": self._state.active_task_id,
                "phase": self._state.phase,
                "progress": self._state.progress,
                "progress_message": self._state.progress_message,
                "cancel_requested": self._state.cancel_requested,
                "task_started_at": self._state.task_started_at,
                "phase_started_at": self._state.phase_started_at,
                "phase_elapsed_seconds": self._state.phase_elapsed_seconds,
                "phase_timings": dict(self._state.phase_timings),
                "cold_start": self._state.cold_start,
            } if self._state.active_task is not None else {}
        self._engine = None
        with self._state_lock:
            self._state = RuntimeState(
                last_error=previous_error,
                completed_tasks=completed_tasks,
                **active,
            )
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            logger.debug("CUDA cache cleanup was unavailable", exc_info=True)

    def _restore_generation_residency(
        self,
        *,
        device: str,
        release_after: bool,
    ) -> bool | None:
        """Restore the main model after sequential decoding without failing output.

        ``None`` means the residency check does not apply.  A failed best-effort
        transfer is reported in performance metadata but must not discard audio
        that has already been generated and saved successfully.
        """
        if (
            release_after
            or not device.startswith("cuda")
            or self._state.memory_mode != "sequential"
            or self._engine is None
        ):
            return None
        restore = getattr(self._engine, "restore_generation_model_device", None)
        if not callable(restore):
            return None
        resident = getattr(self._engine, "generation_model_is_resident", None)
        try:
            if callable(resident) and bool(resident()):
                return True
            self._progress(
                "model_residency",
                0.985,
                "正在恢复主模型到 GPU，方便下一次任务",
            )
            self._check_cancel()
            restore()
            self._check_cancel()
            return bool(resident()) if callable(resident) else True
        except TaskCancelledError:
            raise
        except Exception:
            logger.warning("Failed to restore the generation model residency", exc_info=True)
            return False

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        request = apply_quality_preset(request)
        task = str(request.get("task", "")).strip()
        if task not in ALL_TASKS:
            raise WorkerProtocolError(f"不支持的任务：{task!r}")
        model_root = request.get("model_root")
        if not model_root:
            raise WorkerProtocolError("缺少 model_root")
        device = _resolve_device(str(request.get("device") or "auto"))
        memory_mode = str(request.get("memory_mode") or "auto")
        acceleration_mode = str(request.get("acceleration_mode") or "auto_safe")
        release_after = bool(request.get("release_after", False))
        task_id = str(request.get("task_id") or uuid.uuid4())
        started = time.perf_counter()

        with self._lock:
            self._cancel_event.clear()
            engine_before = self._engine
            self._begin_trace(cold_start=engine_before is None)
            self._update_state(
                active_task=task,
                active_task_id=task_id,
                cancel_requested=False,
                last_error=None,
            )
            self._progress("validating", 0.01, "正在校验模型与任务参数")
            _reset_cuda_peak_stats(device)
            try:
                self._check_cancel()
                self._load(
                    model_root,
                    device,
                    profile_for_task(task) == "full",
                    memory_mode,
                    acceleration_mode,
                )
                cold_start = engine_before is None or engine_before is not self._engine
                self._update_state(cold_start=cold_start)
                self._check_cancel()
                if hasattr(self._engine, "set_task_callbacks"):
                    self._engine.set_task_callbacks(self._check_cancel, self._progress)
                self._progress("running", 0.04, "模型就绪，开始推理")
                result = self._run_task(task, request)
                self._check_cancel()
                gpu_resident = self._restore_generation_residency(
                    device=device,
                    release_after=release_after,
                )
                with self._state_lock:
                    self._state.completed_tasks += 1
                self._progress("complete", 1.0, "任务完成")
                result["task"] = task
                result["task_id"] = task_id
                elapsed = round(time.perf_counter() - started, 3)
                result["elapsed_seconds"] = elapsed
                result["device"] = device
                result["code_revision"] = CODE_REVISION
                result["model_revision"] = MODEL_REVISION
                result["quality_preset"] = request["quality_preset"]
                result["performance"] = _performance_report(
                    elapsed=elapsed,
                    trace=self._trace_snapshot(),
                    cold_start=cold_start,
                    device=device,
                    memory_mode=self._state.memory_mode,
                    acceleration_mode=self._state.acceleration_mode,
                    output_duration_seconds=result.get("duration_seconds"),
                )
                if gpu_resident is not None:
                    result["performance"]["gpu_resident_after_task"] = gpu_resident
                if result.get("metadata_path"):
                    _augment_output_metadata(
                        Path(str(result["metadata_path"])),
                        {"performance": result["performance"]},
                    )
                return result
            except Exception as exc:
                if isinstance(exc, TaskCancelledError):
                    self._progress("cancelled", self._state.progress, "任务已取消")
                else:
                    self._progress("failed", self._state.progress, "任务失败")
                self._update_state(last_error=str(exc))
                raise
            finally:
                if self._engine is not None and hasattr(self._engine, "set_task_callbacks"):
                    self._engine.set_task_callbacks()
                self._update_state(
                    active_task=None,
                    active_task_id=None,
                    cancel_requested=False,
                )
                self._cancel_event.clear()
                if release_after:
                    self._unload_locked(preserve_error=True)
                self._reset_trace()

    def infer_tts_batch(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        """Run latent-first TTS and decode the successful batch after one model switch."""
        if not requests:
            return {"outcomes": [], "completed": 0, "failed": 0}
        normalized = [apply_quality_preset({**request, "task": "tts"}) for request in requests]
        first = normalized[0]
        model_root = first.get("model_root")
        if not model_root:
            raise WorkerProtocolError("批量 TTS 缺少 model_root")
        device = _resolve_device(str(first.get("device") or "auto"))
        memory_mode = str(first.get("memory_mode") or "auto")
        acceleration_mode = str(first.get("acceleration_mode") or "auto_safe")
        release_after = any(bool(request.get("release_after", False)) for request in normalized)
        for request in normalized[1:]:
            if str(request.get("model_root") or "") != str(model_root):
                raise WorkerProtocolError("同一批次必须使用相同模型目录")
            if _resolve_device(str(request.get("device") or "auto")) != device:
                raise WorkerProtocolError("同一批次必须使用相同设备")
            if str(request.get("memory_mode") or "auto") != memory_mode:
                raise WorkerProtocolError("同一批次必须使用相同显存模式")
            if str(request.get("acceleration_mode") or "auto_safe") != acceleration_mode:
                raise WorkerProtocolError("同一批次必须使用相同加速模式")
        task_id = f"tts-batch-{uuid.uuid4()}"
        started = time.perf_counter()
        with self._lock:
            self._cancel_event.clear()
            engine_before = self._engine
            self._begin_trace(cold_start=engine_before is None)
            self._update_state(
                active_task="tts_batch",
                active_task_id=task_id,
                cancel_requested=False,
                last_error=None,
            )
            self._progress("validating", 0.01, f"正在校验 {len(normalized)} 条批量 TTS")
            _reset_cuda_peak_stats(device)
            try:
                self._load(model_root, device, True, memory_mode, acceleration_mode)
                cold_start = engine_before is None or engine_before is not self._engine
                self._update_state(cold_start=cold_start)
                assert self._engine is not None
                if not hasattr(self._engine, "tts_batch"):
                    raise WorkerProtocolError("当前推理核心不支持 latent-first 批量解码")
                if hasattr(self._engine, "set_task_callbacks"):
                    self._engine.set_task_callbacks(self._check_cancel, self._progress)
                engine_requests = []
                prepared_outputs: list[Path] = []
                for request in normalized:
                    self._check_cancel()
                    prepared_outputs.append(_validated_output_path(request.get("output_path")))
                    engine_requests.append(
                        {
                            "prompt_text": _required_text(request, "prompt_text"),
                            "prompt_audio": str(
                                prepare_audio_path(_required_text(request, "prompt_audio"))
                            ),
                            "target_text": _required_text(request, "target_text"),
                            "language": str(request.get("language") or "zh"),
                            "seed": _optional_int(request.get("seed")),
                            "max_new_audio_steps": int(request.get("max_new_audio_steps", 750)),
                            "min_new_audio_steps": int(request.get("min_new_audio_steps", 6)),
                            "max_new_text_tokens": int(request.get("max_new_text_tokens", 512)),
                            "n_timesteps": int(request.get("n_timesteps", 10)),
                            "inference_cfg": float(request.get("inference_cfg", 2.0)),
                        }
                    )
                generated = self._engine.tts_batch(engine_requests)
                outcomes: list[dict[str, Any]] = []
                total_duration = 0.0
                completed = 0
                for index, (request, output_path, value) in enumerate(
                    zip(normalized, prepared_outputs, generated, strict=True)
                ):
                    if isinstance(value, Exception):
                        outcomes.append(
                            {
                                "ok": False,
                                "index": index,
                                "task_id": request.get("task_id"),
                                "error": f"{type(value).__name__}: {value}",
                            }
                        )
                        continue
                    try:
                        _save_audio(output_path, value.audio)
                        duration = round(int(value.audio.shape[-1]) / 24_000, 3)
                        total_duration += duration
                        metadata_path = _write_output_metadata(output_path, "tts", request)
                        response = {
                            "output_path": str(output_path),
                            "metadata_path": str(metadata_path),
                            "sample_rate": 24_000,
                            "duration_seconds": duration,
                            "text": value.text,
                            "task": "tts",
                            "task_id": str(request.get("task_id") or ""),
                            "device": device,
                            "code_revision": CODE_REVISION,
                            "model_revision": MODEL_REVISION,
                            "quality_preset": request["quality_preset"],
                        }
                        outcomes.append({"ok": True, "index": index, "result": response})
                        completed += 1
                    except Exception as exc:
                        outcomes.append(
                            {
                                "ok": False,
                                "index": index,
                                "task_id": request.get("task_id"),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                gpu_resident = self._restore_generation_residency(
                    device=device,
                    release_after=release_after,
                )
                elapsed = round(time.perf_counter() - started, 3)
                self._progress("complete", 1.0, "批量 latent 生成与统一解码完成")
                performance = _performance_report(
                    elapsed=elapsed,
                    trace=self._trace_snapshot(),
                    cold_start=cold_start,
                    device=device,
                    memory_mode=self._state.memory_mode,
                    acceleration_mode=self._state.acceleration_mode,
                    output_duration_seconds=total_duration,
                )
                performance.update(
                    batch_size=len(normalized),
                    successful_items=completed,
                    execution_model=(
                        "latent_first_decode_later"
                        if self._state.memory_mode == "sequential" and device.startswith("cuda")
                        else "resident_sequential_batch"
                    ),
                )
                if gpu_resident is not None:
                    performance["gpu_resident_after_task"] = gpu_resident
                for outcome in outcomes:
                    if not outcome.get("ok"):
                        continue
                    result = outcome["result"]
                    result["elapsed_seconds"] = elapsed
                    result["performance"] = performance
                    _augment_output_metadata(
                        Path(result["metadata_path"]), {"performance": performance}
                    )
                with self._state_lock:
                    self._state.completed_tasks += completed
                return {
                    "task_id": task_id,
                    "completed": completed,
                    "failed": len(normalized) - completed,
                    "outcomes": outcomes,
                    "performance": performance,
                }
            except Exception as exc:
                if isinstance(exc, TaskCancelledError):
                    self._progress("cancelled", self._state.progress, "批量任务已取消")
                else:
                    self._progress("failed", self._state.progress, "批量任务失败")
                self._update_state(last_error=str(exc))
                raise
            finally:
                if self._engine is not None and hasattr(self._engine, "set_task_callbacks"):
                    self._engine.set_task_callbacks()
                self._update_state(
                    active_task=None,
                    active_task_id=None,
                    cancel_requested=False,
                )
                self._cancel_event.clear()
                if release_after:
                    self._unload_locked(preserve_error=True)
                self._reset_trace()

    def _run_task(self, task: str, request: dict[str, Any]) -> dict[str, Any]:
        assert self._engine is not None
        self._check_cancel()
        seed = request.get("seed")
        if seed is not None and seed != "":
            from inference import set_seed

            set_seed(int(seed))

        if task == "long_asr":
            audio_path = request.get("audio_path")
            if not audio_path:
                raise WorkerProtocolError("长音频 ASR 缺少 audio_path")
            prepared = prepare_audio_path(audio_path)
            prompt = str(request.get("prompt") or "Transcribe speech to text.")
            chunk_seconds = float(request.get("chunk_seconds", 30.0))
            overlap_seconds = float(request.get("overlap_seconds", 1.0))
            silence_search_seconds = float(request.get("silence_search_seconds", 1.5))
            with tempfile.TemporaryDirectory(prefix="fireredaudio-t8-long-asr-") as temp_dir:
                chunks, duration = split_pcm16_wav(
                    prepared,
                    temp_dir,
                    chunk_seconds=chunk_seconds,
                    overlap_seconds=overlap_seconds,
                    silence_search_seconds=silence_search_seconds,
                )
                public_segments: list[dict[str, object]] = []
                total = max(1, len(chunks))
                for index, chunk in enumerate(chunks):
                    self._check_cancel()
                    if hasattr(self._engine, "set_task_callbacks"):
                        self._engine.set_task_callbacks(
                            self._check_cancel,
                            lambda phase, local, message, i=index: self._progress(
                                f"segment_{i + 1}_{phase}",
                                0.05 + 0.9 * (i + local) / total,
                                f"分段 {i + 1}/{total}：{message}",
                            ),
                        )
                    result = self._engine.understand(
                        chunk.path,
                        prompt,
                        task="asr",
                        enable_thinking=False,
                        max_new_tokens=_optional_int(request.get("max_new_tokens")),
                    )
                    public_segments.append(chunk.public_dict(result.answer.strip()))
                    self._progress(
                        "long_asr",
                        0.05 + 0.9 * (index + 1) / total,
                        f"已完成 {index + 1}/{total} 个分段",
                    )
            transcript, public_segments = deduplicate_segment_texts(public_segments)
            return {
                "answer": transcript,
                "segments": public_segments,
                "srt": render_srt(public_segments),
                "vtt": render_vtt(public_segments),
                "jsonl": render_jsonl(public_segments),
                "duration_seconds": round(duration, 3),
                "chunk_seconds": chunk_seconds,
                "overlap_seconds": overlap_seconds,
                "silence_search_seconds": silence_search_seconds,
                "timing_accuracy": "segment-level approximate timestamps; not word alignment",
            }

        if task in {"asr", "understand", "long_locate"}:
            audio_paths = request.get("audio_paths") or request.get("audio_path")
            if not audio_paths:
                raise WorkerProtocolError("ASR/音频理解缺少 audio_paths")
            if isinstance(audio_paths, (str, Path)):
                audio_paths = prepare_audio_path(audio_paths)
            else:
                audio_paths = [prepare_audio_path(item) for item in audio_paths]
            self._check_cancel()
            prompt = str(request.get("prompt") or "Transcribe speech to text.")
            result = self._engine.understand(
                audio_paths,
                prompt,
                task="asr" if task == "asr" else "understand",
                enable_thinking=bool(request.get("enable_thinking", False)),
                max_new_tokens=_optional_int(request.get("max_new_tokens")),
            )
            response = {"answer": result.answer, "reasoning": result.reasoning}
            if task == "long_locate":
                response["mode"] = str(request.get("mode") or "timeline_summary")
                response["structured"] = _parse_json_answer(result.answer)
            return response

        output_path = _validated_output_path(request.get("output_path"))
        common = {
            "max_new_audio_steps": int(request.get("max_new_audio_steps", 750)),
            "min_new_audio_steps": int(request.get("min_new_audio_steps", 6)),
            "max_new_text_tokens": int(request.get("max_new_text_tokens", 512)),
            "n_timesteps": int(request.get("n_timesteps", 10)),
            "inference_cfg": float(request.get("inference_cfg", 2.0)),
        }
        if task == "tts":
            result = self._engine.tts(
                prompt_text=_required_text(request, "prompt_text"),
                prompt_audio=prepare_audio_path(_required_text(request, "prompt_audio")),
                target_text=_required_text(request, "target_text"),
                language=str(request.get("language") or "zh"),
                **common,
            )
        elif task == "edit":
            result = self._engine.edit(
                audio_path=prepare_audio_path(_required_text(request, "audio_path")),
                instruction=_required_text(request, "instruction"),
                edit_type=str(request.get("edit_type") or "semantic"),
                **common,
            )
        else:
            result = self._engine.voice_design(
                instruction=_required_text(request, "instruction"),
                text=_required_text(request, "text"),
                **common,
            )
        _save_audio(output_path, result.audio)
        self._check_cancel()
        metadata_path = _write_output_metadata(output_path, task, request)
        response: dict[str, Any] = {
            "output_path": str(output_path),
            "metadata_path": str(metadata_path),
            "sample_rate": 24_000,
            "duration_seconds": round(int(result.audio.shape[-1]) / 24_000, 3),
        }
        if getattr(result, "text", None) is not None:
            response["text"] = result.text
        if task == "voice_design":
            response.update(
                requested_seed=getattr(result, "requested_seed", None),
                actual_seed=getattr(result, "actual_seed", None),
                quality_retry_count=int(getattr(result, "quality_retry_count", 0) or 0),
                quality_gate_reason=getattr(result, "quality_gate_reason", None),
            )
            _augment_output_metadata(
                metadata_path,
                {
                    "requested_seed": response["requested_seed"],
                    "actual_seed": response["actual_seed"],
                    "quality_retry_count": response["quality_retry_count"],
                    "quality_gate_reason": response["quality_gate_reason"],
                },
            )
        return response


def _save_audio(path: Path, audio: Any) -> None:
    from fireredaudio.utils.audio import write_pcm16_wav

    write_pcm16_wav(str(path), audio, 24_000)


def _write_output_metadata(path: Path, task: str, request: dict[str, Any]) -> Path:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    safe_keys = (
        "task_id",
        "quality_preset",
        "device",
        "memory_mode",
        "acceleration_mode",
        "seed",
        "language",
        "edit_type",
        "max_new_audio_steps",
        "min_new_audio_steps",
        "max_new_text_tokens",
        "n_timesteps",
        "inference_cfg",
    )
    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "output_file": path.name,
        "output_sha256": digest,
        "sample_rate": 24_000,
        "runtime_version": RUNTIME_VERSION,
        "code_revision": CODE_REVISION,
        "model_revision": MODEL_REVISION,
        "settings": {key: request[key] for key in safe_keys if key in request},
        "privacy": "Prompt text and input paths are intentionally omitted.",
    }
    target = path.with_suffix(path.suffix + ".json")
    target.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _augment_output_metadata(path: Path, values: dict[str, Any]) -> None:
    """Atomically append non-sensitive runtime evidence to an output sidecar."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return
        payload.update(values)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    except Exception:
        logger.warning("无法补充生成性能记录：%s", path, exc_info=True)


def _reset_cuda_peak_stats(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.reset_peak_memory_stats(torch.device(device))
    except Exception:
        logger.debug("CUDA peak memory reset unavailable", exc_info=True)


def _performance_report(
    *,
    elapsed: float,
    trace: dict[str, Any],
    cold_start: bool,
    device: str,
    memory_mode: str | None,
    output_duration_seconds: Any,
    acceleration_mode: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "cold_start": bool(cold_start),
        "total_seconds": elapsed,
        "phase_seconds": trace["phase_timings"],
        "device": device,
        "memory_mode": memory_mode,
        "acceleration_mode": acceleration_mode,
    }
    try:
        duration = float(output_duration_seconds)
    except (TypeError, ValueError):
        duration = 0.0
    if duration > 0:
        report["output_duration_seconds"] = round(duration, 3)
        report["rtf"] = round(elapsed / duration, 3)
    if str(device).startswith("cuda"):
        try:
            import torch

            target = torch.device(device)
            report["gpu_peak_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(target)
            )
            report["gpu_peak_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(target)
            )
        except Exception:
            logger.debug("CUDA peak memory report unavailable", exc_info=True)
    return report


def _required_text(request: dict[str, Any], key: str) -> str:
    value = str(request.get(key) or "").strip()
    if not value:
        raise WorkerProtocolError(f"缺少 {key}")
    return value


def _validated_output_path(value: Any) -> Path:
    if not value:
        raise WorkerProtocolError("生成任务缺少 output_path")
    target = Path(str(value)).expanduser().resolve()
    if target.suffix.lower() != ".wav":
        raise WorkerProtocolError("output_path 必须使用 .wav 后缀")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _parse_json_answer(value: str) -> Any | None:
    text = str(value or "").strip()
    candidates = [text]
    if "```" in text:
        for block in text.split("```"):
            candidate = block.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].lstrip()
            if candidate.startswith(("{", "[")):
                candidates.append(candidate)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _resolve_memory_mode(
    requested: str,
    device: str,
    require_decoder: bool,
    *,
    recommended_min_vram_bytes: int | None = None,
) -> str:
    if not require_decoder:
        return "full_gpu"
    if requested in {"full_gpu", "sequential", "decoder_cpu"}:
        return requested
    if requested != "auto":
        raise WorkerProtocolError("memory_mode 必须是 auto/full_gpu/sequential/decoder_cpu")
    if not str(device).startswith("cuda"):
        return "decoder_cpu"
    try:
        import torch

        index = torch.device(device).index or 0
        free, _total = torch.cuda.mem_get_info(index)
        threshold = int(recommended_min_vram_bytes or 36 * 1024**3)
        return "sequential" if free < threshold else "full_gpu"
    except Exception:
        return "sequential"


def _resolve_device(requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value != "auto":
        return value
    devices = gpu_inventory()
    return str(devices[0]["id"]) if devices else "cpu"
