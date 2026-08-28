from __future__ import annotations

from typing import Any

from .errors import WorkerProtocolError


EXPORT_PRESETS: dict[str, dict[str, Any]] = {
    "audiobook": {
        "label": "有声书",
        "strategy": "sequence",
        "normalize_loudness": True,
        "target_lufs": -20.0,
        "loudness_range_lu": 7.0,
        "true_peak_ceiling_dbfs": -3.0,
        "highpass_hz": 70.0,
        "render_stems": False,
        "crossfade_seconds": 0.04,
    },
    "podcast": {
        "label": "播客 / 访谈",
        "strategy": "sequence",
        "normalize_loudness": True,
        "target_lufs": -16.0,
        "loudness_range_lu": 7.0,
        "true_peak_ceiling_dbfs": -1.0,
        "highpass_hz": 70.0,
        "render_stems": True,
        "crossfade_seconds": 0.08,
    },
    "video_dialogue": {
        "label": "视频对白",
        "strategy": "timeline",
        "normalize_loudness": True,
        "target_lufs": -23.0,
        "loudness_range_lu": 11.0,
        "true_peak_ceiling_dbfs": -1.0,
        "highpass_hz": 80.0,
        "render_stems": True,
        "crossfade_seconds": 0.03,
    },
}


def public_export_presets() -> dict[str, dict[str, Any]]:
    return {name: dict(value) for name, value in EXPORT_PRESETS.items()}


def resolve_export_config(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("export_preset") or "custom").strip().lower()
    if name != "custom" and name not in EXPORT_PRESETS:
        raise WorkerProtocolError(f"未知导出预设：{name}")
    if name in EXPORT_PRESETS:
        value = {"export_preset": name, **EXPORT_PRESETS[name]}
    else:
        value = {
            "export_preset": "custom",
            "label": "自定义",
            "strategy": str(payload.get("strategy") or "timeline").lower(),
            "normalize_loudness": bool(payload.get("normalize_loudness", False)),
            "target_lufs": float(payload.get("target_lufs", -16.0)),
            "loudness_range_lu": float(payload.get("loudness_range_lu", 11.0)),
            "true_peak_ceiling_dbfs": float(
                payload.get("true_peak_ceiling_dbfs", -1.0)
            ),
            "highpass_hz": (
                None
                if payload.get("highpass_hz") in (None, "", 0, 0.0)
                else float(payload["highpass_hz"])
            ),
            "render_stems": bool(payload.get("render_stems", False)),
            "crossfade_seconds": float(payload.get("crossfade_seconds", 0.0)),
        }
    if value["strategy"] not in {"sequence", "timeline", "overlay"}:
        raise WorkerProtocolError("时间线策略必须是 sequence/timeline/overlay")
    if not -35.0 <= float(value["target_lufs"]) <= -8.0:
        raise WorkerProtocolError("导出目标响度必须在 -35…-8 LUFS")
    if not -9.0 <= float(value["true_peak_ceiling_dbfs"]) <= 0.0:
        raise WorkerProtocolError("导出 True Peak 上限必须在 -9…0 dBFS")
    if value["highpass_hz"] is not None and not 20.0 <= float(value["highpass_hz"]) <= 300.0:
        raise WorkerProtocolError("导出高通必须在 20…300 Hz")
    if not 0.0 <= float(value["crossfade_seconds"]) <= 2.0:
        raise WorkerProtocolError("交叉淡化必须在 0…2 秒")
    return value
