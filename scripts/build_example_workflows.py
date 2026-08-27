from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "example_workflows" / "ui"
API = ROOT / "example_workflows" / "api"


def node(
    node_id: int,
    node_type: str,
    pos: list[int],
    widgets: list[Any],
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    size: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "pos": pos,
        "size": size or [340, 210],
        "flags": {},
        "order": node_id - 1,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {"Node name for S&R": node_type},
        "widgets_values": widgets,
    }


def port(name: str, kind: str, link: int | None = None) -> dict[str, Any]:
    value = {"name": name, "type": kind}
    if link is not None:
        value["link"] = link
    return value


def out(name: str, kind: str, links: list[int] | None = None, slot_index: int = 0) -> dict[str, Any]:
    return {"name": name, "type": kind, "links": links or [], "slot_index": slot_index}


def workflow(nodes: list[dict[str, Any]], links: list[list[Any]]) -> dict[str, Any]:
    return {
        "last_node_id": max(item["id"] for item in nodes),
        "last_link_id": max([item[0] for item in links], default=0),
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.9, "offset": [80, 80]}},
        "version": 0.4,
    }


def model_node(output_links: list[int]) -> dict[str, Any]:
    return node(
        1,
        "T8_FireRedAudio_ModelLoader",
        [60, 80],
        ["FireRedAudio", "", "auto", "auto", "auto_safe", "full", "managed", "", "", "", False, False],
        [],
        [out("model", "T8_FIREREDAUDIO_MODEL", output_links), out("model_info", "STRING")],
        [370, 330],
    )


def save(name: str, data: dict[str, Any]) -> None:
    (UI if name.startswith("ui_") else API).mkdir(parents=True, exist_ok=True)
    target = (UI if name.startswith("ui_") else API) / (name.removeprefix("ui_").removeprefix("api_") + ".json")
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build() -> None:
    load_audio = node(2, "LoadAudio", [60, 450], ["voice_reference.wav", None, None], [], [out("AUDIO", "AUDIO", [2])], [300, 170])
    settings = node(3, "T8_FireRedAudio_GenerationSettings", [470, 80], ["balanced", 42, 750, 6, 512, 10, 2.0], [], [out("settings", "T8_FIREREDAUDIO_SETTINGS", [3])], [320, 280])
    tts = node(4, "T8_FireRedAudio_TTS", [850, 180], ["同时，他强调微调要科学有序。", "欢迎使用 FireRedAudio，来自 T8star-Aix。", "zh", True], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("prompt_audio", "AUDIO", 2), port("settings", "T8_FIREREDAUDIO_SETTINGS", 3)], [out("audio", "AUDIO", [4]), out("report", "STRING"), out("reference_transcript", "STRING", slot_index=2)], [390, 330])
    save_audio = node(5, "SaveAudio", [1300, 180], ["fireredaudio/tts"], [port("audio", "AUDIO", 4)], [out("audio", "AUDIO")], [280, 120])
    save("ui_01_zero_shot_tts", workflow([model_node([1]), load_audio, settings, tts, save_audio], [[1, 1, 0, 4, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 4, 1, "AUDIO"], [3, 3, 0, 4, 2, "T8_FIREREDAUDIO_SETTINGS"], [4, 4, 0, 5, 0, "AUDIO"]]))
    save("api_01_zero_shot_tts", {
        "1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "cuda:0", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}},
        "3": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}},
        "4": {"class_type": "T8_FireRedAudio_TTS", "inputs": {"model": ["1", 0], "prompt_audio": ["2", 0], "prompt_text": "同时，他强调微调要科学有序。", "target_text": "欢迎使用 FireRedAudio，来自 T8star-Aix。", "language": "zh", "auto_transcribe_reference": True, "settings": ["3", 0]}},
        "5": {"class_type": "SaveAudio", "inputs": {"audio": ["4", 0], "filename_prefix": "fireredaudio/tts"}},
    })

    asr = node(3, "T8_FireRedAudio_ASR", [500, 220], ["Transcribe speech to text.", 300], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2)], [out("transcript", "STRING"), out("report", "STRING")])
    save("ui_02_asr", workflow([model_node([1]), load_audio, asr], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "AUDIO"]]))
    save("api_02_asr", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "cuda:0", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}}, "3": {"class_type": "T8_FireRedAudio_ASR", "inputs": {"model": ["1", 0], "audio": ["2", 0], "prompt": "Transcribe speech to text.", "max_new_tokens": 300}}})

    long_asr = node(3, "T8_FireRedAudio_LongASR", [500, 220], ["Transcribe speech to text.", 30.0, 1.0, 300, 1.5], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2)], [out("transcript", "STRING"), out("srt", "STRING"), out("segments_json", "STRING"), out("report", "STRING"), out("vtt", "STRING"), out("jsonl", "STRING")], [390, 340])
    save("ui_06_long_audio_asr", workflow([model_node([1]), load_audio, long_asr], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "AUDIO"]]))
    save("api_06_long_audio_asr", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "long_recording.wav"}}, "3": {"class_type": "T8_FireRedAudio_LongASR", "inputs": {"model": ["1", 0], "audio": ["2", 0], "prompt": "Transcribe speech to text.", "chunk_seconds": 30.0, "overlap_seconds": 1.0, "max_new_tokens": 300, "silence_search_seconds": 1.5}}})

    understand = node(3, "T8_FireRedAudio_Understand", [500, 220], ["请总结音频内容。", True, 1024], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2), port("audio_2", "AUDIO")], [out("answer", "STRING"), out("reasoning", "STRING"), out("report", "STRING")])
    save("ui_03_audio_understanding", workflow([model_node([1]), load_audio, understand], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "AUDIO"]]))
    save("api_03_audio_understanding", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "cuda:0", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}}, "3": {"class_type": "T8_FireRedAudio_Understand", "inputs": {"model": ["1", 0], "audio": ["2", 0], "prompt": "请总结音频内容。", "enable_thinking": True, "max_new_tokens": 1024}}})

    design_settings = node(2, "T8_FireRedAudio_GenerationSettings", [470, 80], ["balanced", 42, 750, 6, 512, 10, 2.0], [], [out("settings", "T8_FIREREDAUDIO_SETTINGS", [2])], [320, 280])
    design = node(3, "T8_FireRedAudio_VoiceDesign", [850, 180], ["青年女性，音色清亮，语速适中，语调自然亲切。", "欢迎使用 FireRedAudio。"], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("settings", "T8_FIREREDAUDIO_SETTINGS", 2)], [out("audio", "AUDIO", [3]), out("report", "STRING")])
    save_design = node(4, "SaveAudio", [1280, 180], ["fireredaudio/voice_design"], [port("audio", "AUDIO", 3)], [out("audio", "AUDIO")], [280, 120])
    save("ui_04_voice_design", workflow([model_node([1]), design_settings, design, save_design], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "T8_FIREREDAUDIO_SETTINGS"], [3, 3, 0, 4, 0, "AUDIO"]]))
    save("api_04_voice_design", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "cuda:0", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}}, "3": {"class_type": "T8_FireRedAudio_VoiceDesign", "inputs": {"model": ["1", 0], "instruction": "青年女性，音色清亮，语速适中，语调自然亲切。", "text": "欢迎使用 FireRedAudio。", "settings": ["2", 0]}}, "4": {"class_type": "SaveAudio", "inputs": {"audio": ["3", 0], "filename_prefix": "fireredaudio/voice_design"}}})

    edit = node(4, "T8_FireRedAudio_SpeechEdit", [850, 220], ["adjust the speed to 1.2", "acoustic"], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2), port("settings", "T8_FIREREDAUDIO_SETTINGS", 3)], [out("audio", "AUDIO", [4]), out("edited_text", "STRING"), out("report", "STRING")])
    save_edit = node(5, "SaveAudio", [1280, 220], ["fireredaudio/edit"], [port("audio", "AUDIO", 4)], [out("audio", "AUDIO")], [280, 120])
    save("ui_05_speech_edit", workflow([model_node([1]), load_audio, settings, edit, save_edit], [[1, 1, 0, 4, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 4, 1, "AUDIO"], [3, 3, 0, 4, 2, "T8_FIREREDAUDIO_SETTINGS"], [4, 4, 0, 5, 0, "AUDIO"]]))
    save("api_05_speech_edit", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "cuda:0", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}}, "3": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}}, "4": {"class_type": "T8_FireRedAudio_SpeechEdit", "inputs": {"model": ["1", 0], "audio": ["2", 0], "instruction": "adjust the speed to 1.2", "edit_type": "acoustic", "settings": ["3", 0]}}, "5": {"class_type": "SaveAudio", "inputs": {"audio": ["4", 0], "filename_prefix": "fireredaudio/edit"}}})

    locator = node(3, "T8_FireRedAudio_LongLocator", [500, 210], ["timeline_summary", "", True, 2048], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2)], [out("answer", "STRING"), out("structured_json", "STRING"), out("reasoning", "STRING"), out("report", "STRING")], [410, 300])
    save("ui_07_long_audio_locator", workflow([model_node([1]), load_audio, locator], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "AUDIO"]]))
    save("api_07_long_audio_locator", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "long_recording.wav"}}, "3": {"class_type": "T8_FireRedAudio_LongLocator", "inputs": {"model": ["1", 0], "audio": ["2", 0], "mode": "timeline_summary", "query": "", "enable_thinking": True, "max_new_tokens": 2048}}})

    acoustic_edit = node(4, "T8_FireRedAudio_AcousticEdit", [850, 220], ["speed", 3, 1.2, 1.0], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2), port("settings", "T8_FIREREDAUDIO_SETTINGS", 3)], [out("audio", "AUDIO"), out("instruction", "STRING"), out("report", "STRING")], [410, 300])
    save("ui_08_parametric_acoustic_edit", workflow([model_node([1]), load_audio, settings, acoustic_edit], [[1, 1, 0, 4, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 4, 1, "AUDIO"], [3, 3, 0, 4, 2, "T8_FIREREDAUDIO_SETTINGS"]]))
    save("api_08_parametric_acoustic_edit", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}}, "3": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}}, "4": {"class_type": "T8_FireRedAudio_AcousticEdit", "inputs": {"model": ["1", 0], "audio": ["2", 0], "operation": "speed", "pitch_steps": 3, "speed": 1.2, "volume": 1.0, "settings": ["3", 0]}}})

    quality = node(3, "T8_FireRedAudio_ReferenceQuality", [500, 220], [], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2)], [out("audio", "AUDIO"), out("quality_report", "STRING")], [390, 180])
    save("ui_09_reference_audio_quality", workflow([model_node([1]), load_audio, quality], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "AUDIO"]]))
    save("api_09_reference_audio_quality", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}}, "3": {"class_type": "T8_FireRedAudio_ReferenceQuality", "inputs": {"model": ["1", 0], "audio": ["2", 0]}}})

    cleanup_audio = node(2, "LoadAudio", [60, 450], ["voice_reference.wav", None, None], [], [out("AUDIO", "AUDIO", [3])], [300, 170])
    cleanup_quality = node(3, "T8_FireRedAudio_ReferenceQuality", [430, 180], [], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 3)], [out("audio", "AUDIO", [4]), out("quality_report", "STRING")], [390, 180])
    cleanup = node(4, "T8_FireRedAudio_PrepareReference", [850, 210], [True, False, True], [port("model", "T8_FIREREDAUDIO_MODEL", 2), port("audio", "AUDIO", 4)], [out("clean_audio", "AUDIO", [5]), out("cleanup_report", "STRING"), out("output_path", "STRING")], [410, 250])
    cleanup_save = node(5, "SaveAudio", [1320, 230], ["fireredaudio/reference-clean"], [port("audio", "AUDIO", 5)], [out("audio", "AUDIO")], [300, 120])
    save("ui_15_reference_cleanup", workflow([model_node([1, 2]), cleanup_audio, cleanup_quality, cleanup, cleanup_save], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 1, 0, 4, 0, "T8_FIREREDAUDIO_MODEL"], [3, 2, 0, 3, 1, "AUDIO"], [4, 3, 0, 4, 1, "AUDIO"], [5, 4, 0, 5, 0, "AUDIO"]]))
    save("api_15_reference_cleanup", {
        "1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}},
        "3": {"class_type": "T8_FireRedAudio_ReferenceQuality", "inputs": {"model": ["1", 0], "audio": ["2", 0]}},
        "4": {"class_type": "T8_FireRedAudio_PrepareReference", "inputs": {"model": ["1", 0], "audio": ["3", 0], "trim_silence": True, "normalize_loudness": False, "speech_highpass": True}},
        "5": {"class_type": "SaveAudio", "inputs": {"audio": ["4", 0], "filename_prefix": "fireredaudio/reference-clean"}},
    })

    compare_audio = node(3, "LoadAudio", [60, 650], ["comparison.wav", None, None], [], [out("AUDIO", "AUDIO", [3])], [300, 170])
    multi = node(4, "T8_FireRedAudio_MultiUnderstand", [520, 260], ["Compare these recordings and explain the relevant similarities and differences.", True, 1536], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio_0", "AUDIO", 2), port("audio_1", "AUDIO", 3)], [out("answer", "STRING"), out("reasoning", "STRING"), out("report", "STRING")], [430, 320])
    save("ui_10_multi_audio_understanding", workflow([model_node([1]), load_audio, compare_audio, multi], [[1, 1, 0, 4, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 4, 1, "AUDIO"], [3, 3, 0, 4, 2, "AUDIO"]]))
    save("api_10_multi_audio_understanding", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}}, "3": {"class_type": "LoadAudio", "inputs": {"audio": "comparison.wav"}}, "4": {"class_type": "T8_FireRedAudio_MultiUnderstand", "inputs": {"model": ["1", 0], "audios.audio_0": ["2", 0], "audios.audio_1": ["3", 0], "prompt": "Compare these recordings and explain the relevant similarities and differences.", "enable_thinking": True, "max_new_tokens": 1536}}})

    export_settings = node(2, "T8_FireRedAudio_GenerationSettings", [470, 80], ["balanced", 42, 750, 6, 512, 10, 2.0], [], [out("settings", "T8_FIREREDAUDIO_SETTINGS", [2])], [320, 280])
    export_design = node(3, "T8_FireRedAudio_VoiceDesign", [850, 180], ["A warm, calm narrator with clear articulation.", "This example exports an MP3 file."], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("settings", "T8_FIREREDAUDIO_SETTINGS", 2)], [out("audio", "AUDIO", [3]), out("report", "STRING")], [410, 250])
    export_audio = node(4, "T8_FireRedAudio_SaveAudio", [1310, 210], ["mp3", "voice_design", "fireredaudio/exports"], [port("audio", "AUDIO", 3)], [out("saved_path", "STRING")], [360, 210])
    save("ui_11_multiformat_audio_export", workflow([model_node([1]), export_settings, export_design, export_audio], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "T8_FIREREDAUDIO_SETTINGS"], [3, 3, 0, 4, 0, "AUDIO"]]))
    save("api_11_multiformat_audio_export", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}}, "3": {"class_type": "T8_FireRedAudio_VoiceDesign", "inputs": {"model": ["1", 0], "instruction": "A warm, calm narrator with clear articulation.", "text": "This example exports an MP3 file.", "settings": ["2", 0]}}, "4": {"class_type": "T8_FireRedAudio_SaveAudio", "inputs": {"audio": ["3", 0], "audio_format": "mp3", "filename_prefix": "voice_design", "subfolder": "fireredaudio/exports"}}})

    subtitle_asr = node(3, "T8_FireRedAudio_LongASR", [500, 180], ["Transcribe speech to text.", 30.0, 1.0, 300, 1.5], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2)], [out("transcript", "STRING"), out("srt", "STRING", slot_index=1), out("segments_json", "STRING", slot_index=2), out("report", "STRING", slot_index=3), out("vtt", "STRING", [3], slot_index=4), out("jsonl", "STRING", slot_index=5)], [390, 340])
    subtitle_save = node(4, "T8_FireRedAudio_SaveSubtitle", [990, 250], ["vtt", "long_audio", "fireredaudio/subtitles"], [port("content", "STRING", 3)], [out("saved_path", "STRING")], [360, 210])
    save("ui_12_subtitle_export", workflow([model_node([1]), load_audio, subtitle_asr, subtitle_save], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "AUDIO"], [3, 3, 4, 4, 0, "STRING"]]))
    save("api_12_subtitle_export", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "long_recording.wav"}}, "3": {"class_type": "T8_FireRedAudio_LongASR", "inputs": {"model": ["1", 0], "audio": ["2", 0], "prompt": "Transcribe speech to text.", "chunk_seconds": 30.0, "overlap_seconds": 1.0, "max_new_tokens": 300, "silence_search_seconds": 1.5}}, "4": {"class_type": "T8_FireRedAudio_SaveSubtitle", "inputs": {"content": ["3", 4], "text_format": "vtt", "filename_prefix": "long_audio", "subfolder": "fireredaudio/subtitles"}}})

    narrator_audio = node(2, "LoadAudio", [50, 460], ["narrator_reference.wav", None, None], [], [out("AUDIO", "AUDIO", [2])], [300, 170])
    narrator_profile = node(3, "T8_FireRedAudio_VoiceProfile", [390, 430], ["旁白", "这是旁白参考音频的逐字稿。", "zh", "温暖,清晰"], [port("audio", "AUDIO", 2)], [out("profile", "T8_FIREREDAUDIO_VOICE_PROFILE", [3]), out("profile_json", "STRING")], [390, 260])
    actor_audio = node(4, "LoadAudio", [50, 720], ["actor_reference.wav", None, None], [], [out("AUDIO", "AUDIO", [4])], [300, 170])
    actor_profile = node(5, "T8_FireRedAudio_VoiceProfile", [390, 720], ["小夏", "这是小夏参考音频的逐字稿。", "zh", "年轻,自然"], [port("audio", "AUDIO", 4)], [out("profile", "T8_FIREREDAUDIO_VOICE_PROFILE", [5]), out("profile_json", "STRING")], [390, 260])
    voice_bank = node(6, "T8_FireRedAudio_VoiceBank", [830, 560], [], [port("profile_0", "T8_FIREREDAUDIO_VOICE_PROFILE", 3), port("profile_1", "T8_FIREREDAUDIO_VOICE_PROFILE", 5)], [out("voice_bank", "T8_FIREREDAUDIO_VOICE_BANK", [6, 8]), out("voice_bank_json", "STRING")], [350, 210])
    parser = node(7, "T8_FireRedAudio_ScriptParser", [1240, 460], ["旁白：欢迎收听本期节目。\n[00:00:03,000 --> 00:00:06,000] 小夏：大家好，我是小夏。", "role_script", "旁白", False], [port("voice_bank", "T8_FIREREDAUDIO_VOICE_BANK", 6)], [out("script_plan", "T8_FIREREDAUDIO_SCRIPT_PLAN", [7]), out("normalized_json", "STRING"), out("preflight_report", "STRING")], [430, 360])
    production_settings = node(8, "T8_FireRedAudio_GenerationSettings", [470, 80], ["balanced", 42, 750, 6, 512, 10, 2.0], [], [out("settings", "T8_FIREREDAUDIO_SETTINGS", [9])], [320, 280])
    batch = node(9, "T8_FireRedAudio_BatchDubbing", [1740, 330], ["role-demo", "fireredaudio/projects", True, True, 8], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("script_plan", "T8_FIREREDAUDIO_SCRIPT_PLAN", 7), port("voice_bank", "T8_FIREREDAUDIO_VOICE_BANK", 8), port("settings", "T8_FIREREDAUDIO_SETTINGS", 9)], [out("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", [11, 12]), out("manifest_path", "STRING"), out("batch_report", "STRING")], [430, 330])
    timeline = node(10, "T8_FireRedAudio_TimelineRender", [2230, 250], ["timeline", 120, 0, False, "limit", 24000], [port("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", 11)], [out("audio", "AUDIO", [13]), out("timeline_report", "STRING")], [390, 340])
    speech_qa = node(11, "T8_FireRedAudio_SpeechQA", [2230, 590], [0.2, 0.001, 0.8, 0.5, 512], [port("model", "T8_FIREREDAUDIO_MODEL", 10), port("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", 12)], [out("qa", "T8_FIREREDAUDIO_SPEECH_QA"), out("qa_report", "STRING"), out("failed_line_ids", "STRING")], [420, 330])
    production_save = node(12, "T8_FireRedAudio_SaveAudio", [2700, 270], ["wav", "role-demo-mix", "fireredaudio/renders"], [port("audio", "AUDIO", 13)], [out("saved_path", "STRING")], [360, 220])
    role_links = [
        [1, 1, 0, 9, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 0, "AUDIO"],
        [3, 3, 0, 6, 0, "T8_FIREREDAUDIO_VOICE_PROFILE"], [4, 4, 0, 5, 0, "AUDIO"],
        [5, 5, 0, 6, 1, "T8_FIREREDAUDIO_VOICE_PROFILE"], [6, 6, 0, 7, 0, "T8_FIREREDAUDIO_VOICE_BANK"],
        [7, 7, 0, 9, 1, "T8_FIREREDAUDIO_SCRIPT_PLAN"], [8, 6, 0, 9, 2, "T8_FIREREDAUDIO_VOICE_BANK"],
        [9, 8, 0, 9, 3, "T8_FIREREDAUDIO_SETTINGS"], [10, 1, 0, 11, 0, "T8_FIREREDAUDIO_MODEL"],
        [11, 9, 0, 10, 0, "T8_FIREREDAUDIO_AUDIO_BATCH"], [12, 9, 0, 11, 1, "T8_FIREREDAUDIO_AUDIO_BATCH"],
        [13, 10, 0, 12, 0, "AUDIO"],
    ]
    save("ui_13_role_dubbing_pipeline", workflow([model_node([1, 10]), narrator_audio, narrator_profile, actor_audio, actor_profile, voice_bank, parser, production_settings, batch, timeline, speech_qa, production_save], role_links))
    save("api_13_role_dubbing_pipeline", {
        "1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "narrator_reference.wav"}},
        "3": {"class_type": "T8_FireRedAudio_VoiceProfile", "inputs": {"audio": ["2", 0], "name": "旁白", "prompt_text": "这是旁白参考音频的逐字稿。", "language": "zh", "tags": "温暖,清晰"}},
        "4": {"class_type": "LoadAudio", "inputs": {"audio": "actor_reference.wav"}},
        "5": {"class_type": "T8_FireRedAudio_VoiceProfile", "inputs": {"audio": ["4", 0], "name": "小夏", "prompt_text": "这是小夏参考音频的逐字稿。", "language": "zh", "tags": "年轻,自然"}},
        "6": {"class_type": "T8_FireRedAudio_VoiceBank", "inputs": {"profiles.profile_0": ["3", 0], "profiles.profile_1": ["5", 0]}},
        "7": {"class_type": "T8_FireRedAudio_ScriptParser", "inputs": {"voice_bank": ["6", 0], "script": "旁白：欢迎收听本期节目。\n[00:00:03,000 --> 00:00:06,000] 小夏：大家好，我是小夏。", "source_format": "role_script", "default_speaker": "旁白", "strict_validation": False}},
        "8": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}},
        "9": {"class_type": "T8_FireRedAudio_BatchDubbing", "inputs": {"model": ["1", 0], "script_plan": ["7", 0], "voice_bank": ["6", 0], "project_name": "role-demo", "subfolder": "fireredaudio/projects", "resume": True, "continue_on_error": True, "batch_size": 8, "settings": ["8", 0]}},
        "10": {"class_type": "T8_FireRedAudio_TimelineRender", "inputs": {"audio_batch": ["9", 0], "mode": "timeline", "gap_ms": 120, "crossfade_ms": 0, "auto_fill_gaps": False, "peak_policy": "limit", "sample_rate": 24000}},
        "11": {"class_type": "T8_FireRedAudio_SpeechQA", "inputs": {"model": ["1", 0], "audio_batch": ["9", 0], "max_text_error_rate": 0.2, "max_clipping_ratio": 0.001, "max_silence_ratio": 0.8, "max_cue_overrun_seconds": 0.5, "max_new_tokens": 512}},
        "12": {"class_type": "T8_FireRedAudio_SaveAudio", "inputs": {"audio": ["10", 0], "audio_format": "wav", "filename_prefix": "role-demo-mix", "subfolder": "fireredaudio/renders"}},
    })

    srt_audio = node(2, "LoadAudio", [60, 430], ["narrator_reference.wav", None, None], [], [out("AUDIO", "AUDIO", [2])], [300, 170])
    srt_profile = node(3, "T8_FireRedAudio_VoiceProfile", [410, 400], ["旁白", "这是旁白参考音频的逐字稿。", "zh", ""], [port("audio", "AUDIO", 2)], [out("profile", "T8_FIREREDAUDIO_VOICE_PROFILE", [3]), out("profile_json", "STRING")], [390, 260])
    srt_bank = node(4, "T8_FireRedAudio_VoiceBank", [850, 420], [], [port("profile_0", "T8_FIREREDAUDIO_VOICE_PROFILE", 3)], [out("voice_bank", "T8_FIREREDAUDIO_VOICE_BANK", [4, 6]), out("voice_bank_json", "STRING")], [350, 190])
    srt_parser = node(5, "T8_FireRedAudio_ScriptParser", [1250, 350], ["1\n00:00:00,000 --> 00:00:02,500\n[旁白] 第一条字幕。\n\n2\n00:00:03,000 --> 00:00:05,500\n[旁白] 第二条字幕。", "srt", "旁白", True], [port("voice_bank", "T8_FIREREDAUDIO_VOICE_BANK", 4)], [out("script_plan", "T8_FIREREDAUDIO_SCRIPT_PLAN", [5]), out("normalized_json", "STRING"), out("preflight_report", "STRING")], [430, 390])
    srt_settings = node(6, "T8_FireRedAudio_GenerationSettings", [470, 70], ["balanced", 42, 750, 6, 512, 10, 2.0], [], [out("settings", "T8_FIREREDAUDIO_SETTINGS", [7])], [320, 280])
    srt_batch = node(7, "T8_FireRedAudio_BatchDubbing", [1730, 300], ["srt-demo", "fireredaudio/projects", True, False, 8], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("script_plan", "T8_FIREREDAUDIO_SCRIPT_PLAN", 5), port("voice_bank", "T8_FIREREDAUDIO_VOICE_BANK", 6), port("settings", "T8_FIREREDAUDIO_SETTINGS", 7)], [out("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", [8]), out("manifest_path", "STRING"), out("batch_report", "STRING")], [430, 330])
    srt_room_tone = node(10, "LoadAudio", [1740, 760], ["room_tone.wav", None, None], [], [out("AUDIO", "AUDIO", [10])], [300, 170])
    srt_delivery = node(11, "T8_FireRedAudio_DeliveryPreset", [2220, 760], ["video_dialogue"], [], [out("delivery_preset", "T8_FIREREDAUDIO_DELIVERY_PRESET", [11, 12]), out("preset_report", "STRING")], [390, 190])
    srt_timeline = node(8, "T8_FireRedAudio_TimelineRender", [2220, 330], ["timeline", 120, 0, True, "limit", 24000], [port("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", 8), port("room_tone_audio", "AUDIO", 10), port("delivery_preset", "T8_FIREREDAUDIO_DELIVERY_PRESET", 11)], [out("audio", "AUDIO", [9]), out("timeline_report", "STRING")], [430, 380])
    srt_save = node(9, "T8_FireRedAudio_SaveAudio", [2710, 350], ["wav", "srt-demo-mix", "fireredaudio/renders"], [port("audio", "AUDIO", 9), port("delivery_preset", "T8_FIREREDAUDIO_DELIVERY_PRESET", 12)], [out("saved_path", "STRING")], [380, 240])
    srt_links = [[1, 1, 0, 7, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 0, "AUDIO"], [3, 3, 0, 4, 0, "T8_FIREREDAUDIO_VOICE_PROFILE"], [4, 4, 0, 5, 0, "T8_FIREREDAUDIO_VOICE_BANK"], [5, 5, 0, 7, 1, "T8_FIREREDAUDIO_SCRIPT_PLAN"], [6, 4, 0, 7, 2, "T8_FIREREDAUDIO_VOICE_BANK"], [7, 6, 0, 7, 3, "T8_FIREREDAUDIO_SETTINGS"], [8, 7, 0, 8, 0, "T8_FIREREDAUDIO_AUDIO_BATCH"], [9, 8, 0, 9, 0, "AUDIO"], [10, 10, 0, 8, 1, "AUDIO"], [11, 11, 0, 8, 2, "T8_FIREREDAUDIO_DELIVERY_PRESET"], [12, 11, 0, 9, 1, "T8_FIREREDAUDIO_DELIVERY_PRESET"]]
    save("ui_14_srt_dubbing_pipeline", workflow([model_node([1]), srt_audio, srt_profile, srt_bank, srt_parser, srt_settings, srt_batch, srt_room_tone, srt_delivery, srt_timeline, srt_save], srt_links))
    save("api_14_srt_dubbing_pipeline", {
        "1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "narrator_reference.wav"}},
        "3": {"class_type": "T8_FireRedAudio_VoiceProfile", "inputs": {"audio": ["2", 0], "name": "旁白", "prompt_text": "这是旁白参考音频的逐字稿。", "language": "zh", "tags": ""}},
        "4": {"class_type": "T8_FireRedAudio_VoiceBank", "inputs": {"profiles.profile_0": ["3", 0]}},
        "5": {"class_type": "T8_FireRedAudio_ScriptParser", "inputs": {"voice_bank": ["4", 0], "script": "1\n00:00:00,000 --> 00:00:02,500\n[旁白] 第一条字幕。\n\n2\n00:00:03,000 --> 00:00:05,500\n[旁白] 第二条字幕。", "source_format": "srt", "default_speaker": "旁白", "strict_validation": True}},
        "6": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}},
        "7": {"class_type": "T8_FireRedAudio_BatchDubbing", "inputs": {"model": ["1", 0], "script_plan": ["5", 0], "voice_bank": ["4", 0], "project_name": "srt-demo", "subfolder": "fireredaudio/projects", "resume": True, "continue_on_error": False, "batch_size": 8, "settings": ["6", 0]}},
        "8": {"class_type": "T8_FireRedAudio_TimelineRender", "inputs": {"audio_batch": ["7", 0], "mode": "timeline", "gap_ms": 120, "crossfade_ms": 0, "auto_fill_gaps": True, "peak_policy": "limit", "sample_rate": 24000, "room_tone_audio": ["10", 0], "delivery_preset": ["11", 0]}},
        "9": {"class_type": "T8_FireRedAudio_SaveAudio", "inputs": {"audio": ["8", 0], "audio_format": "wav", "filename_prefix": "srt-demo-mix", "subfolder": "fireredaudio/renders", "delivery_preset": ["11", 0]}},
        "10": {"class_type": "LoadAudio", "inputs": {"audio": "room_tone.wav"}},
        "11": {"class_type": "T8_FireRedAudio_DeliveryPreset", "inputs": {"preset_name": "video_dialogue"}},
    })

    ab_audio_a = node(1, "LoadAudio", [60, 190], ["candidate_a.wav", None, None], [], [out("AUDIO", "AUDIO", [1])], [300, 170])
    ab_audio_b = node(2, "LoadAudio", [60, 470], ["candidate_b.wav", None, None], [], [out("AUDIO", "AUDIO", [2])], [300, 170])
    ab_compare = node(3, "T8_FireRedAudio_SynchronizedAB", [500, 260], [True, True, -20.0, -42.0, 20], [port("audio_a", "AUDIO", 1), port("audio_b", "AUDIO", 2)], [out("audio_a_synced", "AUDIO", [3]), out("audio_b_synced", "AUDIO", [4], slot_index=1), out("comparison_report", "STRING", slot_index=2)], [430, 350])
    ab_save_a = node(4, "T8_FireRedAudio_SaveAudio", [1010, 190], ["wav", "comparison-A", "fireredaudio/comparisons"], [port("audio", "AUDIO", 3)], [out("saved_path", "STRING")], [370, 220])
    ab_save_b = node(5, "T8_FireRedAudio_SaveAudio", [1010, 500], ["wav", "comparison-B", "fireredaudio/comparisons"], [port("audio", "AUDIO", 4)], [out("saved_path", "STRING")], [370, 220])
    save("ui_16_synchronized_ab", workflow([ab_audio_a, ab_audio_b, ab_compare, ab_save_a, ab_save_b], [[1, 1, 0, 3, 0, "AUDIO"], [2, 2, 0, 3, 1, "AUDIO"], [3, 3, 0, 4, 0, "AUDIO"], [4, 3, 1, 5, 0, "AUDIO"]]))
    save("api_16_synchronized_ab", {
        "1": {"class_type": "LoadAudio", "inputs": {"audio": "candidate_a.wav"}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "candidate_b.wav"}},
        "3": {"class_type": "T8_FireRedAudio_SynchronizedAB", "inputs": {"audio_a": ["1", 0], "audio_b": ["2", 0], "synchronize_onset": True, "match_loudness": True, "target_lufs": -20.0, "onset_threshold_dbfs": -42.0, "preroll_ms": 20}},
        "4": {"class_type": "T8_FireRedAudio_SaveAudio", "inputs": {"audio": ["3", 0], "audio_format": "wav", "filename_prefix": "comparison-A", "subfolder": "fireredaudio/comparisons"}},
        "5": {"class_type": "T8_FireRedAudio_SaveAudio", "inputs": {"audio": ["3", 1], "audio_format": "wav", "filename_prefix": "comparison-B", "subfolder": "fireredaudio/comparisons"}},
    })

    reference_audio = node(2, "LoadAudio", [60, 470], ["voice_reference.wav", None, None], [], [out("AUDIO", "AUDIO", [2])], [300, 170])
    reference_asr = node(3, "T8_FireRedAudio_ReferenceTranscript", [440, 400], [512], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("reference_audio", "AUDIO", 2)], [out("reference_audio", "AUDIO", [3]), out("transcript", "STRING", [4], slot_index=1), out("report", "STRING", slot_index=2)], [420, 240])
    reference_settings = node(4, "T8_FireRedAudio_GenerationSettings", [470, 70], ["balanced", 42, 750, 6, 512, 10, 2.0], [], [out("settings", "T8_FIREREDAUDIO_SETTINGS", [6])], [320, 280])
    reference_tts = node(5, "T8_FireRedAudio_TTS", [960, 250], ["欢迎使用自动逐字稿驱动的声音克隆。", "zh", True], [port("model", "T8_FIREREDAUDIO_MODEL", 5), port("prompt_audio", "AUDIO", 3), port("prompt_text", "STRING", 4), port("settings", "T8_FIREREDAUDIO_SETTINGS", 6)], [out("audio", "AUDIO", [7]), out("report", "STRING", [8], slot_index=1), out("reference_transcript", "STRING", slot_index=2)], [440, 350])
    reference_save = node(6, "T8_FireRedAudio_SaveAudio", [1480, 190], ["wav", "asr-reference-tts", "fireredaudio/tts"], [port("audio", "AUDIO", 7)], [out("saved_path", "STRING"), out("audio", "AUDIO", slot_index=1)], [390, 240])
    reference_perf = node(7, "T8_FireRedAudio_PerformanceReport", [1480, 520], [1.0], [port("generation_report", "STRING", 8)], [out("summary", "STRING"), out("performance_json", "STRING", slot_index=1), out("rtf", "FLOAT", slot_index=2), out("total_seconds", "FLOAT", slot_index=3), out("peak_vram_gib", "FLOAT", slot_index=4)], [430, 280])
    reference_links = [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "AUDIO"], [3, 3, 0, 5, 1, "AUDIO"], [4, 3, 1, 5, 2, "STRING"], [5, 1, 0, 5, 0, "T8_FIREREDAUDIO_MODEL"], [6, 4, 0, 5, 3, "T8_FIREREDAUDIO_SETTINGS"], [7, 5, 0, 6, 0, "AUDIO"], [8, 5, 1, 7, 0, "STRING"]]
    save("ui_17_reference_asr_tts", workflow([model_node([1, 5]), reference_audio, reference_asr, reference_settings, reference_tts, reference_save, reference_perf], reference_links))
    save("api_17_reference_asr_tts", {
        "1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}},
        "3": {"class_type": "T8_FireRedAudio_ReferenceTranscript", "inputs": {"model": ["1", 0], "reference_audio": ["2", 0], "max_new_tokens": 512}},
        "4": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}},
        "5": {"class_type": "T8_FireRedAudio_TTS", "inputs": {"model": ["1", 0], "prompt_audio": ["3", 0], "prompt_text": ["3", 1], "target_text": "欢迎使用自动逐字稿驱动的声音克隆。", "language": "zh", "auto_transcribe_reference": True, "settings": ["4", 0]}},
        "6": {"class_type": "T8_FireRedAudio_SaveAudio", "inputs": {"audio": ["5", 0], "audio_format": "wav", "filename_prefix": "asr-reference-tts", "subfolder": "fireredaudio/tts"}},
        "7": {"class_type": "T8_FireRedAudio_PerformanceReport", "inputs": {"generation_report": ["5", 1], "target_rtf": 1.0}},
    })

    audition_audio = node(2, "LoadAudio", [60, 470], ["voice_reference.wav", None, None], [], [out("AUDIO", "AUDIO", [2])], [300, 170])
    audition_settings = node(3, "T8_FireRedAudio_GenerationSettings", [470, 70], ["balanced", 42, 750, 6, 512, 10, 2.0], [], [out("settings", "T8_FIREREDAUDIO_SETTINGS", [3])], [320, 280])
    audition = node(4, "T8_FireRedAudio_SeedAudition", [850, 230], ["", "请从多个候选中选择最自然的一条。", "zh", 42, 4, True, "narrator-audition", "fireredaudio/auditions"], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("prompt_audio", "AUDIO", 2), port("settings", "T8_FIREREDAUDIO_SETTINGS", 3)], [out("recommended_audio", "AUDIO", [4]), out("all_takes", "T8_FIREREDAUDIO_AUDIO_BATCH", slot_index=1), out("manifest_path", "STRING", slot_index=2), out("reference_transcript", "STRING", slot_index=3), out("audition_report", "STRING", slot_index=4)], [460, 430])
    audition_save = node(5, "T8_FireRedAudio_SaveAudio", [1400, 260], ["wav", "recommended-take", "fireredaudio/auditions"], [port("audio", "AUDIO", 4)], [out("saved_path", "STRING"), out("audio", "AUDIO", slot_index=1)], [390, 240])
    save("ui_18_seed_audition", workflow([model_node([1]), audition_audio, audition_settings, audition, audition_save], [[1, 1, 0, 4, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 4, 1, "AUDIO"], [3, 3, 0, 4, 2, "T8_FIREREDAUDIO_SETTINGS"], [4, 4, 0, 5, 0, "AUDIO"]]))
    save("api_18_seed_audition", {
        "1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}},
        "3": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}},
        "4": {"class_type": "T8_FireRedAudio_SeedAudition", "inputs": {"model": ["1", 0], "prompt_audio": ["2", 0], "prompt_text": "", "target_text": "请从多个候选中选择最自然的一条。", "language": "zh", "seed_start": 42, "take_count": 4, "run_asr_qa": True, "project_name": "narrator-audition", "subfolder": "fireredaudio/auditions", "settings": ["3", 0]}},
        "5": {"class_type": "T8_FireRedAudio_SaveAudio", "inputs": {"audio": ["4", 0], "audio_format": "wav", "filename_prefix": "recommended-take", "subfolder": "fireredaudio/auditions"}},
    })

    evidence_audio = node(2, "LoadAudio", [60, 450], ["long_recording.wav", None, None], [], [out("AUDIO", "AUDIO", [2, 3])], [300, 170])
    evidence_locator = node(3, "T8_FireRedAudio_LongLocator", [480, 210], ["content_to_time", "找出提到项目结论的所有片段。", True, 2048], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2)], [out("answer", "STRING"), out("structured_json", "STRING", [4], slot_index=1), out("reasoning", "STRING", slot_index=2), out("report", "STRING", slot_index=3)], [430, 310])
    evidence_clips = node(4, "T8_FireRedAudio_EvidenceClips", [980, 250], ["evidence-demo", "fireredaudio/evidence", 0.25, 8.0, 20], [port("source_audio", "AUDIO", 3), port("structured_json", "STRING", 4)], [out("first_clip", "AUDIO", [5]), out("evidence_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", slot_index=1), out("cut_list_json", "STRING", slot_index=2), out("manifest_path", "STRING", slot_index=3)], [460, 360])
    evidence_save = node(5, "T8_FireRedAudio_SaveAudio", [1510, 260], ["wav", "first-evidence", "fireredaudio/evidence"], [port("audio", "AUDIO", 5)], [out("saved_path", "STRING"), out("audio", "AUDIO", slot_index=1)], [390, 240])
    save("ui_19_long_audio_evidence", workflow([model_node([1]), evidence_audio, evidence_locator, evidence_clips, evidence_save], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "AUDIO"], [3, 2, 0, 4, 0, "AUDIO"], [4, 3, 1, 4, 1, "STRING"], [5, 4, 0, 5, 0, "AUDIO"]]))
    save("api_19_long_audio_evidence", {
        "1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "long_recording.wav"}},
        "3": {"class_type": "T8_FireRedAudio_LongLocator", "inputs": {"model": ["1", 0], "audio": ["2", 0], "mode": "content_to_time", "query": "找出提到项目结论的所有片段。", "enable_thinking": True, "max_new_tokens": 2048}},
        "4": {"class_type": "T8_FireRedAudio_EvidenceClips", "inputs": {"source_audio": ["2", 0], "structured_json": ["3", 1], "project_name": "evidence-demo", "subfolder": "fireredaudio/evidence", "padding_seconds": 0.25, "default_clip_seconds": 8.0, "max_clips": 20}},
        "5": {"class_type": "T8_FireRedAudio_SaveAudio", "inputs": {"audio": ["4", 0], "audio_format": "wav", "filename_prefix": "first-evidence", "subfolder": "fireredaudio/evidence"}},
    })

    loop_audio_a = node(2, "LoadAudio", [60, 430], ["narrator_reference.wav", None, None], [], [out("AUDIO", "AUDIO", [2])], [300, 170])
    loop_profile_a = node(3, "T8_FireRedAudio_VoiceProfile", [410, 390], ["旁白", "这是旁白的参考音频逐字稿。", "zh", "旁白,稳定"], [port("audio", "AUDIO", 2)], [out("profile", "T8_FIREREDAUDIO_VOICE_PROFILE", [3]), out("profile_json", "STRING")], [390, 260])
    loop_audio_b = node(4, "LoadAudio", [60, 700], ["actor_reference.wav", None, None], [], [out("AUDIO", "AUDIO", [4])], [300, 170])
    loop_profile_b = node(5, "T8_FireRedAudio_VoiceProfile", [410, 690], ["小夏", "这是小夏的参考音频逐字稿。", "zh", "青年,自然"], [port("audio", "AUDIO", 4)], [out("profile", "T8_FIREREDAUDIO_VOICE_PROFILE", [5]), out("profile_json", "STRING")], [390, 260])
    loop_bank = node(6, "T8_FireRedAudio_VoiceBank", [850, 510], [], [port("profile_0", "T8_FIREREDAUDIO_VOICE_PROFILE", 3), port("profile_1", "T8_FIREREDAUDIO_VOICE_PROFILE", 5)], [out("voice_bank", "T8_FIREREDAUDIO_VOICE_BANK", [6, 8, 21]), out("voice_bank_json", "STRING")], [350, 220])
    loop_parser = node(7, "T8_FireRedAudio_ScriptParser", [1250, 470], ["旁白：欢迎进入批量创作闭环。\n小夏：未通过质检的台词会单独返修。\n旁白：通过的文件不会重新生成。", "role_script", "旁白", True], [port("voice_bank", "T8_FIREREDAUDIO_VOICE_BANK", 6)], [out("script_plan", "T8_FIREREDAUDIO_SCRIPT_PLAN", [7, 10]), out("normalized_json", "STRING"), out("preflight_report", "STRING")], [450, 390])
    loop_settings = node(8, "T8_FireRedAudio_GenerationSettings", [470, 70], ["balanced", 42, 750, 6, 512, 10, 2.0], [], [out("settings", "T8_FIREREDAUDIO_SETTINGS", [9, 15])], [320, 280])
    loop_batch = node(9, "T8_FireRedAudio_BatchDubbing", [1780, 300], ["creator-loop", "fireredaudio/projects", True, True, 8], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("script_plan", "T8_FIREREDAUDIO_SCRIPT_PLAN", 7), port("voice_bank", "T8_FIREREDAUDIO_VOICE_BANK", 8), port("settings", "T8_FIREREDAUDIO_SETTINGS", 9)], [out("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", [11, 13]), out("manifest_path", "STRING"), out("batch_report", "STRING")], [450, 360])
    loop_qa = node(10, "T8_FireRedAudio_SpeechQA", [2280, 210], [0.2, 0.001, 0.8, 0.5, 512], [port("model", "T8_FIREREDAUDIO_MODEL", 20), port("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", 11)], [out("qa", "T8_FIREREDAUDIO_SPEECH_QA"), out("qa_report", "STRING"), out("failed_line_ids", "STRING", [14], slot_index=2)], [450, 360])
    loop_retry = node(11, "T8_FireRedAudio_BatchRetry", [2780, 300], ["creator-loop-repair", "fireredaudio/repairs", "increment", 1, 2, 8], [port("model", "T8_FIREREDAUDIO_MODEL", 12), port("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", 13), port("script_plan", "T8_FIREREDAUDIO_SCRIPT_PLAN", 10), port("voice_bank", "T8_FIREREDAUDIO_VOICE_BANK", 21), port("failed_line_ids", "STRING", 14), port("settings", "T8_FIREREDAUDIO_SETTINGS", 15)], [out("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", [16, 17, 18]), out("manifest_path", "STRING"), out("repair_report", "STRING")], [470, 430])
    loop_select = node(12, "T8_FireRedAudio_AudioBatchSelect", [3320, 120], ["position", 1, "", ""], [port("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", 16)], [out("audio", "AUDIO"), out("item_json", "STRING"), out("selected_line_id", "STRING"), out("batch_summary", "STRING")], [420, 310])
    loop_export = node(13, "T8_FireRedAudio_SaveAudioBatch", [3320, 480], ["wav", "creator-loop-delivery", "fireredaudio/exports", True, True, 16], [port("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", 17)], [out("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH"), out("manifest_path", "STRING"), out("zip_path", "STRING"), out("export_report", "STRING")], [450, 360])
    loop_timeline = node(14, "T8_FireRedAudio_TimelineRender", [3320, 870], ["sequence", 120, 40, False, "limit", 24000], [port("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", 18)], [out("audio", "AUDIO", [19]), out("timeline_report", "STRING")], [430, 340])
    loop_save_mix = node(15, "T8_FireRedAudio_SaveAudio", [3820, 900], ["wav", "creator-loop-master", "fireredaudio/renders"], [port("audio", "AUDIO", 19)], [out("saved_path", "STRING"), out("audio", "AUDIO", slot_index=1)], [390, 240])
    loop_links = [
        [1, 1, 0, 9, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 0, "AUDIO"],
        [3, 3, 0, 6, 0, "T8_FIREREDAUDIO_VOICE_PROFILE"], [4, 4, 0, 5, 0, "AUDIO"],
        [5, 5, 0, 6, 1, "T8_FIREREDAUDIO_VOICE_PROFILE"], [6, 6, 0, 7, 0, "T8_FIREREDAUDIO_VOICE_BANK"],
        [7, 7, 0, 9, 1, "T8_FIREREDAUDIO_SCRIPT_PLAN"], [8, 6, 0, 9, 2, "T8_FIREREDAUDIO_VOICE_BANK"],
        [9, 8, 0, 9, 3, "T8_FIREREDAUDIO_SETTINGS"], [10, 7, 0, 11, 2, "T8_FIREREDAUDIO_SCRIPT_PLAN"],
        [11, 9, 0, 10, 1, "T8_FIREREDAUDIO_AUDIO_BATCH"], [12, 1, 0, 11, 0, "T8_FIREREDAUDIO_MODEL"],
        [13, 9, 0, 11, 1, "T8_FIREREDAUDIO_AUDIO_BATCH"], [14, 10, 2, 11, 4, "STRING"],
        [15, 8, 0, 11, 5, "T8_FIREREDAUDIO_SETTINGS"], [16, 11, 0, 12, 0, "T8_FIREREDAUDIO_AUDIO_BATCH"],
        [17, 11, 0, 13, 0, "T8_FIREREDAUDIO_AUDIO_BATCH"], [18, 11, 0, 14, 0, "T8_FIREREDAUDIO_AUDIO_BATCH"],
        [19, 14, 0, 15, 0, "AUDIO"], [20, 1, 0, 10, 0, "T8_FIREREDAUDIO_MODEL"],
        [21, 6, 0, 11, 3, "T8_FIREREDAUDIO_VOICE_BANK"],
    ]
    save("ui_20_creator_qa_repair_delivery", workflow([model_node([1, 12, 20]), loop_audio_a, loop_profile_a, loop_audio_b, loop_profile_b, loop_bank, loop_parser, loop_settings, loop_batch, loop_qa, loop_retry, loop_select, loop_export, loop_timeline, loop_save_mix], loop_links))
    save("api_20_creator_qa_repair_delivery", {
        "1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "narrator_reference.wav"}},
        "3": {"class_type": "T8_FireRedAudio_VoiceProfile", "inputs": {"audio": ["2", 0], "name": "旁白", "prompt_text": "这是旁白的参考音频逐字稿。", "language": "zh", "tags": "旁白,稳定"}},
        "4": {"class_type": "LoadAudio", "inputs": {"audio": "actor_reference.wav"}},
        "5": {"class_type": "T8_FireRedAudio_VoiceProfile", "inputs": {"audio": ["4", 0], "name": "小夏", "prompt_text": "这是小夏的参考音频逐字稿。", "language": "zh", "tags": "青年,自然"}},
        "6": {"class_type": "T8_FireRedAudio_VoiceBank", "inputs": {"profiles.profile_0": ["3", 0], "profiles.profile_1": ["5", 0]}},
        "7": {"class_type": "T8_FireRedAudio_ScriptParser", "inputs": {"voice_bank": ["6", 0], "script": "旁白：欢迎进入批量创作闭环。\n小夏：未通过质检的台词会单独返修。\n旁白：通过的文件不会重新生成。", "source_format": "role_script", "default_speaker": "旁白", "strict_validation": True}},
        "8": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}},
        "9": {"class_type": "T8_FireRedAudio_BatchDubbing", "inputs": {"model": ["1", 0], "script_plan": ["7", 0], "voice_bank": ["6", 0], "project_name": "creator-loop", "subfolder": "fireredaudio/projects", "resume": True, "continue_on_error": True, "batch_size": 8, "settings": ["8", 0]}},
        "10": {"class_type": "T8_FireRedAudio_SpeechQA", "inputs": {"model": ["1", 0], "audio_batch": ["9", 0], "max_text_error_rate": 0.2, "max_clipping_ratio": 0.001, "max_silence_ratio": 0.8, "max_cue_overrun_seconds": 0.5, "max_new_tokens": 512}},
        "11": {"class_type": "T8_FireRedAudio_BatchRetry", "inputs": {"model": ["1", 0], "audio_batch": ["9", 0], "script_plan": ["7", 0], "voice_bank": ["6", 0], "failed_line_ids": ["10", 2], "project_name": "creator-loop-repair", "subfolder": "fireredaudio/repairs", "seed_strategy": "increment", "seed_step": 1, "max_attempts": 2, "batch_size": 8, "settings": ["8", 0]}},
        "12": {"class_type": "T8_FireRedAudio_AudioBatchSelect", "inputs": {"audio_batch": ["11", 0], "selection_mode": "position", "position": 1, "line_id": "", "speaker": ""}},
        "13": {"class_type": "T8_FireRedAudio_SaveAudioBatch", "inputs": {"audio_batch": ["11", 0], "audio_format": "wav", "project_name": "creator-loop-delivery", "subfolder": "fireredaudio/exports", "create_zip": True, "continue_on_error": True, "preview_limit": 16}},
        "14": {"class_type": "T8_FireRedAudio_TimelineRender", "inputs": {"audio_batch": ["11", 0], "mode": "sequence", "gap_ms": 120, "crossfade_ms": 40, "auto_fill_gaps": False, "peak_policy": "limit", "sample_rate": 24000}},
        "15": {"class_type": "T8_FireRedAudio_SaveAudio", "inputs": {"audio": ["14", 0], "audio_format": "wav", "filename_prefix": "creator-loop-master", "subfolder": "fireredaudio/renders"}},
    })

    repair_source = node(2, "LoadAudio", [60, 430], ["podcast_source.wav", None, None], [], [out("AUDIO", "AUDIO", [2])], [300, 170])
    repair_settings = node(3, "T8_FireRedAudio_GenerationSettings", [470, 70], ["balanced", 42, 750, 6, 512, 10, 2.0], [], [out("settings", "T8_FIREREDAUDIO_SETTINGS", [6])], [320, 280])
    repair_range = node(4, "T8_FireRedAudio_LocalRepairRange", [450, 410], ["manual", 2.0, 5.0, 1, 250], [port("audio", "AUDIO", 2), port("locator_json", "STRING")], [out("original_audio", "AUDIO"), out("repair_clip", "AUDIO", [3], slot_index=1), out("repair_plan", "T8_FIREREDAUDIO_LOCAL_REPAIR_PLAN", [4], slot_index=2), out("range_report", "STRING", slot_index=3)], [450, 360])
    repair_edit = node(5, "T8_FireRedAudio_SpeechEdit", [980, 300], ["replace the last sentence with: 这是一段已经修复的台词", "semantic"], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 3), port("settings", "T8_FIREREDAUDIO_SETTINGS", 6)], [out("audio", "AUDIO", [7]), out("edited_text", "STRING"), out("report", "STRING", slot_index=2)], [470, 320])
    repair_apply = node(6, "T8_FireRedAudio_LocalRepairApply", [1530, 350], [40], [port("repair_plan", "T8_FIREREDAUDIO_LOCAL_REPAIR_PLAN", 4), port("edited_clip", "AUDIO", 7)], [out("original_audio", "AUDIO", [8]), out("repaired_audio", "AUDIO", [9], slot_index=1), out("replacement_report", "STRING", slot_index=2)], [460, 300])
    repair_save_a = node(7, "T8_FireRedAudio_SaveAudio", [2070, 250], ["wav", "local-repair-A-original", "fireredaudio/repairs"], [port("audio", "AUDIO", 8)], [out("saved_path", "STRING"), out("audio", "AUDIO", slot_index=1)], [390, 240])
    repair_save_b = node(8, "T8_FireRedAudio_SaveAudio", [2070, 560], ["wav", "local-repair-B-repaired", "fireredaudio/repairs"], [port("audio", "AUDIO", 9)], [out("saved_path", "STRING"), out("audio", "AUDIO", slot_index=1)], [390, 240])
    repair_links = [[1, 1, 0, 5, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 4, 0, "AUDIO"], [3, 4, 1, 5, 1, "AUDIO"], [4, 4, 2, 6, 0, "T8_FIREREDAUDIO_LOCAL_REPAIR_PLAN"], [6, 3, 0, 5, 2, "T8_FIREREDAUDIO_SETTINGS"], [7, 5, 0, 6, 1, "AUDIO"], [8, 6, 0, 7, 0, "AUDIO"], [9, 6, 1, 8, 0, "AUDIO"]]
    save("ui_21_podcast_local_repair", workflow([model_node([1]), repair_source, repair_settings, repair_range, repair_edit, repair_apply, repair_save_a, repair_save_b], repair_links))
    save("api_21_podcast_local_repair", {
        "1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "acceleration_mode": "auto_safe", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "podcast_source.wav"}},
        "3": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}},
        "4": {"class_type": "T8_FireRedAudio_LocalRepairRange", "inputs": {"audio": ["2", 0], "range_mode": "manual", "start_seconds": 2.0, "end_seconds": 5.0, "range_index": 1, "context_ms": 250}},
        "5": {"class_type": "T8_FireRedAudio_SpeechEdit", "inputs": {"model": ["1", 0], "audio": ["4", 1], "instruction": "replace the last sentence with: 这是一段已经修复的台词", "edit_type": "semantic", "settings": ["3", 0]}},
        "6": {"class_type": "T8_FireRedAudio_LocalRepairApply", "inputs": {"repair_plan": ["4", 2], "edited_clip": ["5", 0], "crossfade_ms": 40}},
        "7": {"class_type": "T8_FireRedAudio_SaveAudio", "inputs": {"audio": ["6", 0], "audio_format": "wav", "filename_prefix": "local-repair-A-original", "subfolder": "fireredaudio/repairs"}},
        "8": {"class_type": "T8_FireRedAudio_SaveAudio", "inputs": {"audio": ["6", 1], "audio_format": "wav", "filename_prefix": "local-repair-B-repaired", "subfolder": "fireredaudio/repairs"}},
    })

    package_exchange = node(1, "T8_FireRedAudio_ProjectExchange", [60, 340], ["project-exchange.json"], [], [out("voice_bank", "T8_FIREREDAUDIO_VOICE_BANK"), out("script_plan", "T8_FIREREDAUDIO_SCRIPT_PLAN", slot_index=1), out("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", [1], slot_index=2), out("report", "STRING", slot_index=3)], [410, 250])
    package_master = node(2, "LoadAudio", [60, 650], ["approved_master.wav", None, None], [], [out("AUDIO", "AUDIO", [2])], [300, 170])
    package_bgm = node(3, "LoadAudio", [430, 650], ["bgm.wav", None, None], [], [out("AUDIO", "AUDIO", [3])], [300, 170])
    package_room = node(4, "LoadAudio", [800, 650], ["room_tone.wav", None, None], [], [out("AUDIO", "AUDIO", [4])], [300, 170])
    package_preset = node(5, "T8_FireRedAudio_DeliveryPreset", [520, 130], ["podcast"], [], [out("delivery_preset", "T8_FIREREDAUDIO_DELIVERY_PRESET", [5]), out("preset_report", "STRING")], [390, 190])
    package_node = node(6, "T8_FireRedAudio_ProductionPackage", [1250, 320], ["podcast-delivery", "fireredaudio/deliveries", 48000, 40, True, True, True], [port("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", 1), port("master_audio", "AUDIO", 2), port("bgm_audio", "AUDIO", 3), port("room_tone_audio", "AUDIO", 4), port("source_subtitles", "STRING"), port("delivery_preset", "T8_FIREREDAUDIO_DELIVERY_PRESET", 5)], [out("audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH"), out("master_audio", "AUDIO", slot_index=1), out("manifest_path", "STRING", slot_index=2), out("zip_path", "STRING", slot_index=3), out("package_report", "STRING", slot_index=4)], [520, 440])
    package_links = [[1, 1, 2, 6, 0, "T8_FIREREDAUDIO_AUDIO_BATCH"], [2, 2, 0, 6, 1, "AUDIO"], [3, 3, 0, 6, 2, "AUDIO"], [4, 4, 0, 6, 3, "AUDIO"], [5, 5, 0, 6, 5, "T8_FIREREDAUDIO_DELIVERY_PRESET"]]
    save("ui_22_production_package", workflow([package_exchange, package_master, package_bgm, package_room, package_preset, package_node], package_links))
    save("api_22_production_package", {
        "1": {"class_type": "T8_FireRedAudio_ProjectExchange", "inputs": {"exchange_path": "project-exchange.json"}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "approved_master.wav"}},
        "3": {"class_type": "LoadAudio", "inputs": {"audio": "bgm.wav"}},
        "4": {"class_type": "LoadAudio", "inputs": {"audio": "room_tone.wav"}},
        "5": {"class_type": "T8_FireRedAudio_DeliveryPreset", "inputs": {"preset_name": "podcast"}},
        "6": {"class_type": "T8_FireRedAudio_ProductionPackage", "inputs": {"audio_batch": ["1", 2], "project_name": "podcast-delivery", "subfolder": "fireredaudio/deliveries", "sample_rate": 48000, "crossfade_ms": 40, "include_role_stems": True, "include_scene_stems": True, "create_zip": True, "master_audio": ["2", 0], "bgm_audio": ["3", 0], "room_tone_audio": ["4", 0], "delivery_preset": ["5", 0]}},
    })


if __name__ == "__main__":
    build()
