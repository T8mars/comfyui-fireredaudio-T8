from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .errors import WorkerProtocolError
from .audio_quality import analyze_audio
from .audio_post import master_audio
from .delivery_presets import public_export_presets, resolve_export_config
from .errors import TaskCancelledError
from .production_quality import analyze_production_audio
from .project_store import ProjectStore, project_root_from_parent, safe_project_path
from .script_parser import parse_script, parse_script_file
from .timeline import render_timeline, render_track_stems
from .long_audio import render_srt, render_vtt


def handle_project_request(
    route: str, payload: dict[str, Any], *, runtime: Any | None = None
) -> dict[str, Any]:
    path = route.rstrip("/")
    if path == "/v1/project/create":
        project_root = _required(payload, "project_root")
        project_name = str(payload.get("name") or "") or None
        if bool(payload.get("create_under_parent", False)):
            project_root = project_root_from_parent(project_root, project_name or "")
        store = ProjectStore.create(
            project_root, project_name
        )
        return store.snapshot()
    store = ProjectStore(_required(payload, "project_root"))
    if path == "/v1/project/open":
        if bool(payload.get("recover_interrupted", False)):
            store.recover_interrupted_jobs()
        return store.snapshot()
    if path == "/v1/project/import-asset":
        return {
            "asset": store.import_asset(
                _required(payload, "source"),
                kind=str(payload.get("kind") or "audio"),
                copy_into_project=bool(payload.get("copy_into_project", True)),
                metadata=_mapping(payload.get("metadata")),
            ),
            "project": store.snapshot(),
        }
    if path == "/v1/project/import-script":
        import_mode = str(payload.get("mode") or "merge").strip().lower()
        if import_mode not in {"merge", "replace"}:
            raise WorkerProtocolError("台词导入模式必须是 merge/replace")
        known = payload.get("known_speakers")
        known_speakers = (
            {str(value) for value in known}
            if isinstance(known, list)
            else {str(value["name"]) for value in store.list_voice_profiles()}
        )
        if payload.get("script_path"):
            parsed = parse_script_file(
                _required(payload, "script_path"),
                known_speakers=known_speakers,
                default_speaker=str(payload.get("default_speaker") or "旁白"),
            )
        else:
            parsed = parse_script(
                str(payload.get("text") or ""),
                format_hint=str(payload.get("format") or "auto"),
                known_speakers=known_speakers,
                default_speaker=str(payload.get("default_speaker") or "旁白"),
            )
        result = parsed.to_dict()
        for issue in result["issues"]:
            if issue.get("code") == "unknown_speaker":
                issue["severity"] = "warning"
        blocking_issues = [
            issue
            for issue in result["issues"]
            if issue.get("severity") == "error" and issue.get("code") != "unknown_speaker"
        ]
        result["can_commit"] = not blocking_issues
        result["preview"] = (
            store.preview_script_replace(parsed.lines)
            if import_mode == "replace"
            else store.preview_script_merge(parsed.lines)
        )
        if bool(payload.get("commit", False)):
            if blocking_issues:
                result["committed"] = False
            else:
                if import_mode == "replace":
                    merged = store.replace_script_lines_safe(
                        parsed.lines,
                        confirmation=str(payload.get("confirmation") or ""),
                        source_format=parsed.format,
                        source_name=str(payload.get("script_path") or "pasted-script"),
                    )
                    result["backup"] = merged["backup"]
                else:
                    merged = store.merge_script_lines(
                        parsed.lines,
                        source_format=parsed.format,
                        source_name=str(payload.get("script_path") or "pasted-script"),
                    )
                result["lines"] = merged["lines"]
                result["revision_id"] = merged["revision_id"]
                result["preview"] = merged["diff"]
                result["committed"] = True
        else:
            result["committed"] = False
        result["project"] = store.snapshot()
        return result
    if path == "/v1/project/backups/list":
        return {"backups": store.list_script_backups(), "project": store.snapshot()}
    if path == "/v1/project/voices/list":
        voices = store.list_voice_profiles()
        for voice in voices:
            reference_asset_id = voice.get("reference_asset_id")
            if not reference_asset_id:
                voice["reference_path"] = None
                voice["reference_exists"] = False
                continue
            try:
                voice["reference_path"] = str(
                    store.resolve_asset_path(str(reference_asset_id))
                )
                voice["reference_exists"] = True
            except WorkerProtocolError:
                voice["reference_path"] = None
                voice["reference_exists"] = False
        return {"voices": voices, "project": store.snapshot()}
    if path == "/v1/project/voices/upsert":
        return {
            "voice": store.upsert_voice_profile(_mapping(payload.get("voice"))),
            "project": store.snapshot(),
        }
    if path == "/v1/project/casting/list":
        return {"casting": store.casting_readiness(), "project": store.snapshot()}
    if path == "/v1/project/casting/map":
        return {
            "role": store.map_casting_role(
                _required(payload, "speaker"), _required(payload, "voice_profile_id")
            ),
            "casting": store.casting_readiness(),
            "project": store.snapshot(),
        }
    if path == "/v1/project/casting/confirm":
        return {
            "role": store.confirm_casting_role(
                _required(payload, "speaker"),
                representative_take_id=_optional(payload, "representative_take_id"),
                confirmed=bool(payload.get("confirmed", True)),
                notes=str(payload.get("notes") or ""),
            ),
            "casting": store.casting_readiness(),
            "project": store.snapshot(),
        }
    if path == "/v1/project/lines/list":
        return {
            "lines": store.list_script_lines(
                include_archived=bool(payload.get("include_archived", False))
            ),
            "project": store.snapshot(),
        }
    if path == "/v1/project/lines/restore":
        line_ids = payload.get("line_ids")
        if not isinstance(line_ids, list):
            raise WorkerProtocolError("归档恢复需要 line_ids 列表")
        restored = store.restore_archived_lines(line_ids)
        return {**restored, "project": store.snapshot()}
    if path == "/v1/project/pronunciation/list":
        return {"entries": store.list_pronunciation_entries(), "project": store.snapshot()}
    if path == "/v1/project/pronunciation/upsert":
        return {
            "entry": store.upsert_pronunciation_entry(_mapping(payload.get("entry"))),
            "project": store.snapshot(),
        }
    if path == "/v1/project/pronunciation/delete":
        return {
            "deleted": store.delete_pronunciation_entry(_required(payload, "entry_id")),
            "project": store.snapshot(),
        }
    if path == "/v1/project/pronunciation/preview":
        return {
            "preview": store.pronunciation_preview(
                str(payload.get("text") or ""), language=str(payload.get("language") or "zh")
            ),
            "project": store.snapshot(),
        }
    if path == "/v1/project/lines/patch":
        return {
            "line": store.patch_script_line(
                _required(payload, "line_id"), _mapping(payload.get("values"))
            ),
            "project": store.snapshot(),
        }
    if path == "/v1/project/queue/enqueue":
        line_ids = payload.get("line_ids")
        return {
            "jobs": store.enqueue_lines(
                line_ids if isinstance(line_ids, list) else None,
                force=bool(payload.get("force", False)),
                require_casting=bool(payload.get("require_casting", False)),
            ),
            "project": store.snapshot(),
        }
    if path == "/v1/project/queue/list":
        statuses = payload.get("statuses")
        return {
            "jobs": store.list_jobs(statuses if isinstance(statuses, list) else None),
            "project": store.snapshot(),
        }
    if path == "/v1/project/queue/pause":
        return {"control": store.set_queue_paused(True), "project": store.snapshot()}
    if path == "/v1/project/queue/cancel-pending":
        return {"cancelled": store.cancel_pending_jobs(), "project": store.snapshot()}
    if path == "/v1/project/queue/cancel":
        cancellation = store.request_job_cancellation(_required(payload, "job_id"))
        runtime_result: dict[str, Any] | None = None
        if cancellation["needs_runtime_cancel"]:
            if runtime is None:
                store.clear_job_cancellations([cancellation["job_id"]])
                raise WorkerProtocolError("当前任务缺少可取消的推理运行时")
            status = runtime.status()
            active_id = status.get("active_task_id")
            if status.get("active_task") == "tts_batch":
                runtime_result = runtime.cancel(active_id)
            elif str(active_id or "") == str(cancellation["job_id"]):
                runtime_result = runtime.cancel(str(cancellation["job_id"]))
            else:
                store.clear_job_cancellations([cancellation["job_id"]])
                raise WorkerProtocolError("该句已离开推理阶段，请刷新队列状态")
        return {
            **cancellation,
            "runtime": runtime_result,
            "project": store.snapshot(),
        }
    if path == "/v1/project/queue/priority":
        return {
            "job": store.set_job_priority(
                _required(payload, "job_id"), int(payload.get("priority", 0))
            ),
            "project": store.snapshot(),
        }
    if path == "/v1/project/queue/update":
        return {
            "job": store.update_job(
                _required(payload, "job_id"),
                status=_optional(payload, "status"),
                stage=_optional(payload, "stage"),
                checkpoint_rel_path=_optional(payload, "checkpoint_rel_path"),
                error=_optional(payload, "error"),
                take_id=_optional(payload, "take_id"),
            ),
            "project": store.snapshot(),
        }
    if path == "/v1/project/queue/resume":
        store.set_queue_paused(False)
        return {"resumed": store.resume_interrupted_jobs(), "project": store.snapshot()}
    if path == "/v1/project/queue/recover":
        return {"recovered": store.recover_interrupted_jobs(), "project": store.snapshot()}
    if path == "/v1/project/queue/run":
        if runtime is None:
            raise WorkerProtocolError("项目队列缺少推理运行时")
        return _run_project_queue(store, runtime, payload)
    if path == "/v1/project/takes/list":
        return {
            "takes": store.list_takes(_required(payload, "line_id")),
            "project": store.snapshot(),
        }
    if path == "/v1/project/takes/add":
        return {
            "take": store.add_take(
                _required(payload, "line_id"),
                _required(payload, "audio_path"),
                parent_take_id=_optional(payload, "parent_take_id"),
                seed=None if payload.get("seed") in (None, "") else int(payload["seed"]),
                preset=str(payload.get("preset") or "balanced"),
                model_revision=_optional(payload, "model_revision"),
                quality=_mapping(payload.get("quality")),
                adopt=bool(payload.get("adopt", False)),
            ),
            "project": store.snapshot(),
        }
    if path == "/v1/project/takes/adopt":
        return {
            "take": store.adopt_take(
                _required(payload, "take_id"),
                timeline_clip=(
                    _mapping(payload.get("timeline_clip"))
                    if payload.get("timeline_clip") is not None
                    else None
                ),
                force=bool(payload.get("force", False)),
            ),
            "project": store.snapshot(),
        }
    if path == "/v1/project/takes/review":
        return {
            "take": store.review_take(
                _required(payload, "take_id"),
                action=str(payload.get("action") or "note"),
                note=None if "note" not in payload else str(payload.get("note") or ""),
                locked=None if "locked" not in payload else bool(payload.get("locked")),
            ),
            "project": store.snapshot(),
        }
    if path == "/v1/project/takes/prepare-ab":
        return {
            "comparison": store.prepare_take_comparison(
                _required(payload, "take_a_id"),
                _required(payload, "take_b_id"),
                target_lufs=float(payload.get("target_lufs", -20.0)),
                sync_onset=bool(payload.get("sync_onset", True)),
                match_loudness=bool(payload.get("match_loudness", True)),
            ),
            "project": store.snapshot(),
        }
    if path == "/v1/project/artifacts/list":
        return {"artifacts": store.list_project_artifacts(), "project": store.snapshot()}
    if path == "/v1/project/transcripts/asr":
        if runtime is None:
            raise WorkerProtocolError("ASR 校对缺少推理运行时")
        source = _required(payload, "audio_path")
        asset = store.import_asset(source, kind="asr-source", copy_into_project=True)
        audio_path = store.resolve_asset_path(asset["id"])
        result = runtime.infer(
            {
                "task": "long_asr",
                "task_id": str(payload.get("task_id") or f"project-asr-{asset['id']}"),
                "model_root": _required(payload, "model_root"),
                "device": str(payload.get("device") or "auto"),
                "memory_mode": str(payload.get("memory_mode") or "auto"),
                "release_after": bool(payload.get("release_after", False)),
                "audio_path": str(audio_path),
                "prompt": str(payload.get("prompt") or "Transcribe speech to text."),
                "chunk_seconds": float(payload.get("chunk_seconds", 30.0)),
                "overlap_seconds": float(payload.get("overlap_seconds", 1.0)),
                "silence_search_seconds": float(payload.get("silence_search_seconds", 1.5)),
                "max_new_tokens": int(payload.get("max_new_tokens", 300)),
            }
        )
        segments = store.replace_transcript_segments(asset["id"], result.get("segments") or [])
        return {
            "asset": asset,
            "audio_path": str(audio_path),
            "segments": segments,
            "transcript": result.get("answer", ""),
            "performance": result.get("performance"),
            "project": store.snapshot(),
        }
    if path == "/v1/project/transcripts/import":
        asset_id = _required(payload, "asset_id")
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise WorkerProtocolError("segments 必须是数组")
        return {
            "segments": store.replace_transcript_segments(asset_id, segments),
            "project": store.snapshot(),
        }
    if path == "/v1/project/transcripts/list":
        asset_id = str(payload.get("asset_id") or "") or None
        segments = store.list_transcript_segments(asset_id)
        sources = []
        for current_asset_id in sorted({segment["asset_id"] for segment in segments}):
            asset = store.get_asset(current_asset_id)
            try:
                audio_path = str(store.resolve_asset_path(current_asset_id))
                exists = True
            except WorkerProtocolError:
                audio_path = asset["rel_path"]
                exists = False
            sources.append(
                {
                    "asset_id": current_asset_id,
                    "source_name": asset["source_name"],
                    "audio_path": audio_path,
                    "exists": exists,
                }
            )
        return {
            "segments": segments,
            "sources": sources,
            "project": store.snapshot(),
        }
    if path == "/v1/project/transcripts/patch":
        return {
            "segment": store.patch_transcript_segment(
                _required(payload, "segment_id"), _mapping(payload.get("values"))
            ),
            "project": store.snapshot(),
        }
    if path == "/v1/project/transcripts/search-replace":
        return {
            "changed": store.replace_transcript_text(
                _required(payload, "asset_id"),
                _required(payload, "search"),
                str(payload.get("replacement") or ""),
                case_sensitive=bool(payload.get("case_sensitive", True)),
            ),
            "project": store.snapshot(),
        }
    if path == "/v1/project/transcripts/export":
        asset_id = _required(payload, "asset_id")
        segments = store.list_transcript_segments(asset_id)
        if bool(payload.get("confirmed_only", False)):
            segments = [segment for segment in segments if segment["status"] == "confirmed"]
        if not segments:
            raise WorkerProtocolError("没有可导出的校对分段")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = safe_project_path(store.root, Path("scripts") / f"transcript-{stamp}")
        srt_path = base.with_suffix(".srt")
        vtt_path = base.with_suffix(".vtt")
        text_path = base.with_suffix(".txt")
        public_segments = [
            {
                "start_seconds": segment["start_seconds"],
                "end_seconds": segment["end_seconds"],
                "text": segment["text"],
            }
            for segment in segments
        ]
        srt_path.write_text(render_srt(public_segments), encoding="utf-8")
        vtt_path.write_text(render_vtt(public_segments), encoding="utf-8")
        text_path.write_text("\n".join(segment["text"] for segment in segments), encoding="utf-8")
        return {
            "paths": {"srt": str(srt_path), "vtt": str(vtt_path), "text": str(text_path)},
            "segment_count": len(segments),
            "project": store.snapshot(),
        }
    if path == "/v1/project/timeline/list":
        return {"clips": store.list_timeline_clips(), "project": store.snapshot()}
    if path == "/v1/project/production/list":
        return {"clips": store.list_production_clips(), "project": store.snapshot()}
    if path == "/v1/project/production/upsert":
        clip = _mapping(payload.get("clip"))
        if bool(clip.get("fill_gaps")) and float(clip.get("duration") or 0.0) <= 0:
            inputs = store.timeline_render_inputs()
            sequence_duration = sum(float(value.get("duration") or 0.0) for value in inputs)
            timeline_duration = max(
                (
                    float(value.get("position") or 0.0)
                    + float(value.get("duration") or 0.0)
                    for value in inputs
                ),
                default=0.0,
            )
            clip["duration"] = max(sequence_duration, timeline_duration)
            if float(clip["duration"]) <= 0:
                raise WorkerProtocolError("自动补空隙前需要至少一个有时长的对白片段")
        return {
            "clip": store.upsert_production_clip(clip),
            "project": store.snapshot(),
        }
    if path == "/v1/project/production/delete":
        return {
            "deleted": store.delete_production_clip(_required(payload, "clip_id")),
            "project": store.snapshot(),
        }
    if path == "/v1/project/delivery/preflight":
        return {"preflight": store.delivery_preflight(), "project": store.snapshot()}
    if path == "/v1/project/timeline/upsert":
        return {
            "clip": store.upsert_timeline_clip(_mapping(payload.get("clip"))),
            "project": store.snapshot(),
        }
    if path == "/v1/project/timeline/upsert-batch":
        clips = payload.get("clips")
        if not isinstance(clips, list):
            raise WorkerProtocolError("clips 必须是数组")
        return {
            "clips": store.upsert_timeline_clips(
                [_mapping(value) for value in clips]
            ),
            "project": store.snapshot(),
        }
    if path == "/v1/project/timeline/delete":
        clip_ids = payload.get("clip_ids")
        if not isinstance(clip_ids, list):
            raise WorkerProtocolError("clip_ids 必须是数组")
        deleted = store.delete_timeline_clips([str(value) for value in clip_ids])
        return {
            "deleted": deleted,
            "count": len(deleted),
            "project": store.snapshot(),
        }
    if path == "/v1/project/markers/list":
        return {"markers": store.list_markers(), "project": store.snapshot()}
    if path == "/v1/project/markers/upsert":
        return {
            "marker": store.upsert_marker(_mapping(payload.get("marker"))),
            "project": store.snapshot(),
        }
    if path == "/v1/project/markers/delete":
        return {
            "deleted": store.delete_marker(_required(payload, "marker_id")),
            "project": store.snapshot(),
        }
    if path == "/v1/project/markers/import-locator":
        imported = store.import_locator_markers(
            payload.get("structured"),
            replace_source=bool(payload.get("replace_source", False)),
            source_metadata=_mapping(payload.get("source_metadata")),
        )
        return {"markers": imported, "count": len(imported), "project": store.snapshot()}
    if path == "/v1/project/exchange/export":
        return _export_project_exchange(store)
    if path == "/v1/project/export-presets":
        return {"presets": public_export_presets(), "project": store.snapshot()}
    if path == "/v1/project/timeline/render":
        clips = payload.get("clips")
        dialogue_inputs = clips if isinstance(clips, list) else store.timeline_render_inputs()
        production_inputs = store.production_render_inputs()
        render_inputs = [*dialogue_inputs, *production_inputs]
        if not render_inputs:
            raise WorkerProtocolError("项目时间线没有可渲染片段")
        delivery_mode = str(payload.get("delivery_mode") or "draft").lower()
        if delivery_mode not in {"draft", "final"}:
            raise WorkerProtocolError("交付模式必须是 draft/final")
        preflight = store.delivery_preflight()
        if delivery_mode == "final" and not preflight["ok"]:
            messages = "；".join(issue["message"] for issue in preflight["issues"][:8])
            raise WorkerProtocolError(f"最终交付预检未通过：{messages or '项目不完整'}")
        if delivery_mode == "final":
            required_line_ids = {str(value["line_id"]) for value in preflight["lines"]}
            rendered_line_ids = {str(value.get("line_id") or "") for value in render_inputs}
            missing_line_ids = required_line_ids - rendered_line_ids
            if missing_line_ids:
                raise WorkerProtocolError(
                    f"最终交付缺少 {len(missing_line_ids)} 条已采用台词的时间线片段"
                )
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        prefix = "final" if delivery_mode == "final" else "draft"
        relative_output = str(
            payload.get("output_rel_path") or f"renders/{prefix}-render-{stamp}.wav"
        )
        output = safe_project_path(store.root, relative_output)
        export_config = resolve_export_config(payload)
        strategy = str(export_config["strategy"])
        result = render_timeline(
            render_inputs,
            output,
            strategy=strategy,
            allow_overlap=bool(payload.get("allow_overlap", False)),
            crossfade_seconds=float(export_config["crossfade_seconds"]),
        )
        mastering = None
        if bool(export_config["normalize_loudness"]):
            mastering = master_audio(
                output,
                output,
                target_lufs=float(export_config["target_lufs"]),
                loudness_range_lu=float(export_config["loudness_range_lu"]),
                true_peak_dbfs=float(export_config["true_peak_ceiling_dbfs"]),
                highpass_hz=export_config["highpass_hz"],
            )
        quality_report = analyze_production_audio(
            output,
            target_lufs=float(export_config["target_lufs"]),
            tolerance_lu=float(payload.get("tolerance_lu", 2.0)),
            true_peak_ceiling_dbfs=float(export_config["true_peak_ceiling_dbfs"]),
        )
        stems = {}
        if bool(export_config["render_stems"]):
            stem_results = render_track_stems(
                render_inputs,
                safe_project_path(store.root, f"renders/stems-{stamp}"),
                strategy=strategy,
                allow_overlap=bool(payload.get("allow_overlap", False)),
                crossfade_seconds=float(export_config["crossfade_seconds"]),
            )
            stems = {key: value.to_dict() for key, value in stem_results.items()}
        subtitles = _render_project_subtitles(
            store, render_inputs, result.to_dict(), output.with_suffix("")
        )
        manifest_path = output.with_suffix(".manifest.json")
        manifest_payload = {
            "schema_version": 1,
            "project": store.summary().id,
            "delivery_mode": delivery_mode,
            "delivery_preflight": preflight,
            "render": result.to_dict(),
            "stems": stems,
            "subtitles": subtitles,
            "quality_report": quality_report,
            "mastering": mastering,
            "production_clips": production_inputs,
            "export_config": export_config,
        }
        temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(manifest_path)
        recorded = store.record_render(
            output,
            manifest_path=manifest_path,
            quality_report=quality_report,
            delivery_mode=delivery_mode,
            rendered_line_ids=[str(value.get("line_id") or "") for value in render_inputs],
        )
        return {
            "delivery_mode": delivery_mode,
            "delivery_preflight": preflight,
            "render": result.to_dict(),
            "stems": stems,
            "subtitles": subtitles,
            "quality_report": quality_report,
            "mastering": mastering,
            "production_clips": production_inputs,
            "export_config": export_config,
            "manifest_path": str(manifest_path),
            "record": recorded,
            "project": store.snapshot(),
        }
    raise WorkerProtocolError(f"未知项目接口：{route}")


def _render_project_subtitles(
    store: ProjectStore,
    render_inputs: list[dict[str, Any]],
    render_result: dict[str, Any],
    base_path: Path,
) -> dict[str, str]:
    """Render subtitles from the exact post-placement timeline used for audio."""
    inputs_by_id = {str(value.get("id") or ""): value for value in render_inputs}
    lines_by_id = {str(value["id"]): value for value in store.list_script_lines()}
    segments: list[dict[str, object]] = []
    for rendered in render_result.get("clips") or []:
        source = inputs_by_id.get(str(rendered.get("id") or ""), {})
        line = lines_by_id.get(str(source.get("line_id") or ""), {})
        text = str(source.get("text") or line.get("text") or "").strip()
        if not text:
            continue
        speaker = str(source.get("speaker") or line.get("speaker") or "").strip()
        if speaker and speaker != "旁白":
            text = f"{speaker}：{text}"
        start = max(0.0, float(rendered.get("actual_start") or 0.0))
        duration = max(0.001, float(rendered.get("duration") or 0.0))
        segments.append(
            {"start_seconds": start, "end_seconds": start + duration, "text": text}
        )
    if not segments:
        return {}
    srt_path = base_path.with_suffix(".srt")
    vtt_path = base_path.with_suffix(".vtt")
    for target, content in (
        (srt_path, render_srt(segments)),
        (vtt_path, render_vtt(segments)),
    ):
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
    return {"srt": str(srt_path), "vtt": str(vtt_path)}


def _export_project_exchange(store: ProjectStore) -> dict[str, Any]:
    summary = store.summary()
    voices = []
    for voice in store.list_voice_profiles():
        asset_id = str(voice.get("reference_asset_id") or "")
        if not asset_id:
            continue
        asset = store.get_asset(asset_id)
        audio = store.resolve_asset_path(asset_id)
        try:
            audio_relative = audio.relative_to(store.root).as_posix()
        except ValueError:
            audio_relative = None
        voices.append(
            {
                "profile_id": voice["id"],
                "name": voice["name"],
                "prompt_audio": audio_relative,
                "prompt_audio_absolute": str(audio),
                "prompt_audio_sha256": asset["sha256"],
                "prompt_text": voice["transcript"],
                "language": voice["language"],
                "tags": voice["tags"],
                "authorization_notes": voice["authorization_notes"],
            }
        )
    script_lines = [
        {
            "line_id": line["id"],
            "index": int(line["order_index"]) + 1,
            "speaker": line["speaker"],
            "text": line["text"],
            "language": line["language"],
            "start_seconds": line["target_start"],
            "end_seconds": line["target_end"],
        }
        for line in store.list_script_lines()
    ]
    take_items = []
    for artifact in store.list_project_artifacts():
        if artifact["kind"] != "take" or artifact["status"] != "adopted" or not artifact["exists"]:
            continue
        audio_path = Path(artifact["path"])
        take_items.append(
            {
                "line_id": artifact["line_id"],
                "speaker": artifact["speaker"],
                "text": artifact["text"],
                "index": int(artifact["order_index"]) + 1,
                "status": "complete",
                "output_path": audio_path.relative_to(store.root).as_posix(),
                "output_path_absolute": str(audio_path),
                "output_sha256": artifact["sha256"],
            }
        )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = safe_project_path(store.root, Path("scripts") / f"comfyui-exchange-{stamp}.json")
    payload = {
        "format": "t8.firered.project.exchange",
        "version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "project_root": "..",
        "project": {"id": summary.id, "name": summary.name, "schema_version": summary.schema_version},
        "voice_bank": {"profiles": voices, "count": len(voices)},
        "script_plan": {"source_format": "desktop-project", "lines": script_lines, "issues": [], "valid": True},
        "audio_batch": {"items": take_items, "complete": len(take_items), "total": len(script_lines)},
        "timeline": {"clips": store.list_timeline_clips(), "markers": store.list_markers()},
        "privacy": "Prompt paths are project-relative when possible; authorization notes remain local project data.",
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return {"path": str(target), "manifest": payload, "project": store.snapshot()}


def _run_project_queue(store: ProjectStore, runtime: Any, payload: dict[str, Any]) -> dict[str, Any]:
    model_root = _required(payload, "model_root")
    maximum = max(1, min(1000, int(payload.get("max_items", 1000))))
    stop_on_error = bool(payload.get("stop_on_error", False))
    adopt = bool(payload.get("adopt", False))
    queued = store.list_jobs(["queued"])[:maximum]
    latent_batch_size = max(1, min(32, int(payload.get("latent_batch_size", 8))))
    if (
        len(queued) > 1
        and latent_batch_size > 1
        and not bool(payload.get("disable_latent_batch", False))
        and hasattr(runtime, "infer_tts_batch")
    ):
        return _run_project_queue_latent_batch(
            store,
            runtime,
            payload,
            queued,
            batch_size=latent_batch_size,
            stop_on_error=stop_on_error,
            adopt=adopt,
        )
    outcomes: list[dict[str, Any]] = []
    completed = failed = cancelled = 0
    for job in queued:
        if store.queue_control()["paused"]:
            break
        job_id = str(job["id"])
        if store.get_job(job_id)["status"] != "queued":
            continue
        scratch = safe_project_path(store.root, Path("cache") / "jobs" / f"{job_id}.wav")
        scratch.parent.mkdir(parents=True, exist_ok=True)
        try:
            store.update_job(job_id, status="validating", stage="generate", error="")
            request = _queue_inference_request(store, job, payload, model_root, scratch)
            store.update_job(job_id, status="running", stage="generate")
            generated = runtime.infer(request)
            take = _commit_project_generation(
                store, job, request, generated, scratch, adopt=adopt
            )
            completed += 1
            outcomes.append({"job_id": job_id, "status": "completed", "take": take})
        except TaskCancelledError as exc:
            checkpoint = scratch.relative_to(store.root).as_posix() if scratch.exists() else None
            store.update_job(
                job_id,
                status="cancelled",
                checkpoint_rel_path=checkpoint,
                error=str(exc),
            )
            cancelled += 1
            outcomes.append({"job_id": job_id, "status": "cancelled", "error": str(exc)})
            store.clear_job_cancellations([job_id])
            break
        except Exception as exc:
            checkpoint = scratch.relative_to(store.root).as_posix() if scratch.exists() else None
            store.update_job(
                job_id,
                status="failed",
                checkpoint_rel_path=checkpoint,
                error=str(exc)[:4000],
            )
            failed += 1
            outcomes.append({"job_id": job_id, "status": "failed", "error": str(exc)})
            if stop_on_error:
                break
    return {
        "selected": len(queued),
        "completed": completed,
        "failed": failed,
        "cancelled": cancelled,
        "paused": store.queue_control()["paused"],
        "outcomes": outcomes,
        "jobs": store.list_jobs(),
        "execution_model": "sequential_full_pipeline",
        "project": store.snapshot(),
    }


def _run_project_queue_latent_batch(
    store: ProjectStore,
    runtime: Any,
    payload: dict[str, Any],
    queued: list[dict[str, Any]],
    *,
    batch_size: int,
    stop_on_error: bool,
    adopt: bool,
) -> dict[str, Any]:
    model_root = _required(payload, "model_root")
    outcomes: list[dict[str, Any]] = []
    completed = failed = cancelled = 0
    execution_models: set[str] = set()
    stop = False
    for offset in range(0, len(queued), batch_size):
        if stop or store.queue_control()["paused"]:
            break
        prepared: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
        for job in queued[offset : offset + batch_size]:
            if store.queue_control()["paused"]:
                break
            job_id = str(job["id"])
            if store.get_job(job_id)["status"] != "queued":
                continue
            scratch = safe_project_path(store.root, Path("cache") / "jobs" / f"{job_id}.wav")
            scratch.parent.mkdir(parents=True, exist_ok=True)
            try:
                store.update_job(job_id, status="validating", stage="generate", error="")
                request = _queue_inference_request(store, job, payload, model_root, scratch)
                store.update_job(job_id, status="running", stage="generate")
                prepared.append((job, request, scratch))
            except Exception as exc:
                store.update_job(job_id, status="failed", stage="generate", error=str(exc)[:4000])
                failed += 1
                outcomes.append({"job_id": job_id, "status": "failed", "error": str(exc)})
                if stop_on_error:
                    stop = True
                    break
        if not prepared or stop:
            continue
        try:
            batch_result = runtime.infer_tts_batch([request for _job, request, _path in prepared])
            execution_models.add(
                str(
                    (batch_result.get("performance") or {}).get("execution_model")
                    or batch_result.get("execution_model")
                    or "latent_first_decode_later"
                )
            )
        except TaskCancelledError as exc:
            requested_ids = store.requested_job_cancellations()
            for job, _request, scratch in prepared:
                job_id = str(job["id"])
                checkpoint = scratch.relative_to(store.root).as_posix() if scratch.exists() else None
                should_cancel = not requested_ids or job_id in requested_ids
                if should_cancel:
                    store.update_job(
                        job_id,
                        status="cancelled",
                        checkpoint_rel_path=checkpoint,
                        error=str(exc),
                    )
                    cancelled += 1
                    outcomes.append({"job_id": job_id, "status": "cancelled", "error": str(exc)})
                else:
                    store.update_job(
                        job_id,
                        status="queued",
                        checkpoint_rel_path=checkpoint,
                        error="同批另一句被取消，本句已自动重新排队",
                    )
                    outcomes.append({"job_id": job_id, "status": "queued", "error": "批次取消后重新排队"})
            if requested_ids:
                store.clear_job_cancellations(requested_ids)
            break
        except Exception as exc:
            for job, _request, scratch in prepared:
                checkpoint = scratch.relative_to(store.root).as_posix() if scratch.exists() else None
                store.update_job(
                    str(job["id"]),
                    status="failed",
                    checkpoint_rel_path=checkpoint,
                    error=str(exc)[:4000],
                )
                failed += 1
                outcomes.append({"job_id": job["id"], "status": "failed", "error": str(exc)})
            if stop_on_error:
                break
            continue
        by_index = {int(item["index"]): item for item in batch_result.get("outcomes", [])}
        for index, (job, request, scratch) in enumerate(prepared):
            item = by_index.get(index) or {"ok": False, "error": "批量 Worker 未返回该条结果"}
            if not item.get("ok"):
                error = str(item.get("error") or "批量生成失败")
                store.update_job(
                    str(job["id"]), status="failed", stage="generate", error=error[:4000]
                )
                failed += 1
                outcomes.append({"job_id": job["id"], "status": "failed", "error": error})
                if stop_on_error:
                    stop = True
                    break
                continue
            try:
                take = _commit_project_generation(
                    store, job, request, item["result"], scratch, adopt=adopt
                )
                completed += 1
                outcomes.append({"job_id": job["id"], "status": "completed", "take": take})
            except Exception as exc:
                store.update_job(
                    str(job["id"]), status="failed", stage="qa", error=str(exc)[:4000]
                )
                failed += 1
                outcomes.append({"job_id": job["id"], "status": "failed", "error": str(exc)})
                if stop_on_error:
                    stop = True
                    break
    return {
        "selected": len(queued),
        "completed": completed,
        "failed": failed,
        "cancelled": cancelled,
        "paused": store.queue_control()["paused"],
        "outcomes": outcomes,
        "jobs": store.list_jobs(),
        "execution_model": (
            next(iter(execution_models))
            if len(execution_models) == 1
            else "+".join(sorted(execution_models))
            if execution_models
            else "latent_first_decode_later"
        ),
        "latent_batch_size": batch_size,
        "project": store.snapshot(),
    }


def _commit_project_generation(
    store: ProjectStore,
    job: dict[str, Any],
    request: dict[str, Any],
    generated: dict[str, Any],
    scratch: Path,
    *,
    adopt: bool,
) -> dict[str, Any]:
    store.update_job(str(job["id"]), stage="qa")
    signal_report = analyze_audio(generated["output_path"])
    lineage = {
        "signal": signal_report,
        "generation": {
            key: generated[key]
            for key in (
                "task_id",
                "elapsed_seconds",
                "device",
                "code_revision",
                "model_revision",
                "quality_preset",
                "performance",
                "requested_seed",
                "actual_seed",
                "quality_retry_count",
                "quality_gate_reason",
            )
            if key in generated
        },
    }
    take = store.add_take(
        str(job["line_id"]),
        generated["output_path"],
        seed=generated.get("actual_seed", request.get("seed")),
        preset=str(request.get("quality_preset") or "balanced"),
        model_revision=str(generated.get("model_revision") or "") or None,
        quality=lineage,
        adopt=adopt,
    )
    _write_take_sidecar(store, take, job, request, generated, lineage)
    store.update_job(
        str(job["id"]),
        status="completed",
        stage="qa",
        take_id=take["id"],
        checkpoint_rel_path=take["rel_path"],
        error="",
    )
    scratch.unlink(missing_ok=True)
    scratch.with_suffix(scratch.suffix + ".json").unlink(missing_ok=True)
    return take


def _queue_inference_request(
    store: ProjectStore,
    job: dict[str, Any],
    payload: dict[str, Any],
    model_root: str,
    output_path: Path,
) -> dict[str, Any]:
    settings = _mapping(job.get("payload"))
    profile_id = str(settings.get("voice_profile_id") or "")
    if profile_id:
        voice = store.get_voice_profile(profile_id)
        asset_id = str(voice.get("reference_asset_id") or "")
        if not asset_id:
            raise WorkerProtocolError(f"音色 {voice['name']} 没有参考音频")
        prompt_audio = store.resolve_asset_path(asset_id)
        prompt_text = str(voice.get("transcript") or "").strip()
        if not prompt_text:
            raise WorkerProtocolError(f"音色 {voice['name']} 没有参考文本")
    else:
        prompt_audio_value = str(payload.get("default_prompt_audio") or "").strip()
        prompt_text = str(payload.get("default_prompt_text") or "").strip()
        if not prompt_audio_value or not prompt_text:
            raise WorkerProtocolError("台词未映射音色，且未提供默认参考音频/文本")
        prompt_audio = Path(prompt_audio_value).expanduser().resolve()
        if not prompt_audio.is_file():
            raise WorkerProtocolError(f"默认参考音频不存在：{prompt_audio}")
    return {
        "task": "tts",
        "task_id": str(job["id"]),
        "model_root": model_root,
        "device": str(payload.get("device") or "auto"),
        "memory_mode": str(payload.get("memory_mode") or "auto"),
        "release_after": bool(payload.get("release_after", False)),
        "prompt_audio": str(prompt_audio),
        "prompt_text": prompt_text,
        "target_text": str(settings.get("spoken_text") or settings.get("text") or ""),
        "language": str(settings.get("language") or "zh"),
        "seed": settings.get("seed"),
        "quality_preset": str(settings.get("quality_preset") or "balanced"),
        "output_path": str(output_path),
    }


def _write_take_sidecar(
    store: ProjectStore,
    take: dict[str, Any],
    job: dict[str, Any],
    request: dict[str, Any],
    generated: dict[str, Any],
    quality: dict[str, Any],
) -> Path:
    audio = safe_project_path(store.root, take["rel_path"])
    sidecar = audio.with_suffix(audio.suffix + ".json")
    data = {
        "schema_version": 1,
        "project_id": store.summary().id,
        "job_id": job["id"],
        "line_id": job["line_id"],
        "take_id": take["id"],
        "target_text": request["target_text"],
        "voice_profile_id": job.get("payload", {}).get("voice_profile_id"),
        "output_file": audio.name,
        "output_sha256": take["sha256"],
        "generation": {
            key: generated[key]
            for key in (
                "elapsed_seconds",
                "device",
                "code_revision",
                "model_revision",
                "quality_preset",
                "performance",
                "requested_seed",
                "actual_seed",
                "quality_retry_count",
                "quality_gate_reason",
            )
            if key in generated
        },
        "quality": quality,
    }
    temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(sidecar)
    return sidecar


def _required(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise WorkerProtocolError(f"缺少 {key}")
    return value


def _optional(payload: dict[str, Any], key: str) -> str | None:
    if key not in payload or payload[key] is None:
        return None
    return str(payload[key])


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise WorkerProtocolError("字段必须是 JSON 对象")
    return value
