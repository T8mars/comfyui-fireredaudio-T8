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
        ["FireRedAudio", "", "auto", "auto", "full", "managed", "", "", "", False, False],
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
    tts = node(4, "T8_FireRedAudio_TTS", [850, 180], ["同时，他强调微调要科学有序。", "欢迎使用 FireRedAudio，来自 T8star-Aix。", "zh"], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("prompt_audio", "AUDIO", 2), port("settings", "T8_FIREREDAUDIO_SETTINGS", 3)], [out("audio", "AUDIO", [4]), out("report", "STRING")], [390, 300])
    save_audio = node(5, "SaveAudio", [1300, 180], ["fireredaudio/tts"], [port("audio", "AUDIO", 4)], [out("audio", "AUDIO")], [280, 120])
    save("ui_01_zero_shot_tts", workflow([model_node([1]), load_audio, settings, tts, save_audio], [[1, 1, 0, 4, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 4, 1, "AUDIO"], [3, 3, 0, 4, 2, "T8_FIREREDAUDIO_SETTINGS"], [4, 4, 0, 5, 0, "AUDIO"]]))
    save("api_01_zero_shot_tts", {
        "1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "cuda:0", "memory_mode": "auto", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}},
        "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}},
        "3": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}},
        "4": {"class_type": "T8_FireRedAudio_TTS", "inputs": {"model": ["1", 0], "prompt_audio": ["2", 0], "prompt_text": "同时，他强调微调要科学有序。", "target_text": "欢迎使用 FireRedAudio，来自 T8star-Aix。", "language": "zh", "settings": ["3", 0]}},
        "5": {"class_type": "SaveAudio", "inputs": {"audio": ["4", 0], "filename_prefix": "fireredaudio/tts"}},
    })

    asr = node(3, "T8_FireRedAudio_ASR", [500, 220], ["Transcribe speech to text.", 300], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2)], [out("transcript", "STRING"), out("report", "STRING")])
    save("ui_02_asr", workflow([model_node([1]), load_audio, asr], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "AUDIO"]]))
    save("api_02_asr", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "cuda:0", "memory_mode": "auto", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}}, "3": {"class_type": "T8_FireRedAudio_ASR", "inputs": {"model": ["1", 0], "audio": ["2", 0], "prompt": "Transcribe speech to text.", "max_new_tokens": 300}}})

    long_asr = node(3, "T8_FireRedAudio_LongASR", [500, 220], ["Transcribe speech to text.", 30.0, 1.0, 300, 1.5], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2)], [out("transcript", "STRING"), out("srt", "STRING"), out("segments_json", "STRING"), out("report", "STRING"), out("vtt", "STRING"), out("jsonl", "STRING")], [390, 340])
    save("ui_06_long_audio_asr", workflow([model_node([1]), load_audio, long_asr], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "AUDIO"]]))
    save("api_06_long_audio_asr", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "long_recording.wav"}}, "3": {"class_type": "T8_FireRedAudio_LongASR", "inputs": {"model": ["1", 0], "audio": ["2", 0], "prompt": "Transcribe speech to text.", "chunk_seconds": 30.0, "overlap_seconds": 1.0, "max_new_tokens": 300, "silence_search_seconds": 1.5}}})

    understand = node(3, "T8_FireRedAudio_Understand", [500, 220], ["请总结音频内容。", True, 1024], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2), port("audio_2", "AUDIO")], [out("answer", "STRING"), out("reasoning", "STRING"), out("report", "STRING")])
    save("ui_03_audio_understanding", workflow([model_node([1]), load_audio, understand], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "AUDIO"]]))
    save("api_03_audio_understanding", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "cuda:0", "memory_mode": "auto", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}}, "3": {"class_type": "T8_FireRedAudio_Understand", "inputs": {"model": ["1", 0], "audio": ["2", 0], "prompt": "请总结音频内容。", "enable_thinking": True, "max_new_tokens": 1024}}})

    design_settings = node(2, "T8_FireRedAudio_GenerationSettings", [470, 80], ["balanced", 42, 750, 6, 512, 10, 2.0], [], [out("settings", "T8_FIREREDAUDIO_SETTINGS", [2])], [320, 280])
    design = node(3, "T8_FireRedAudio_VoiceDesign", [850, 180], ["青年女性，音色清亮，语速适中，语调自然亲切。", "欢迎使用 FireRedAudio。"], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("settings", "T8_FIREREDAUDIO_SETTINGS", 2)], [out("audio", "AUDIO", [3]), out("report", "STRING")])
    save_design = node(4, "SaveAudio", [1280, 180], ["fireredaudio/voice_design"], [port("audio", "AUDIO", 3)], [out("audio", "AUDIO")], [280, 120])
    save("ui_04_voice_design", workflow([model_node([1]), design_settings, design, save_design], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "T8_FIREREDAUDIO_SETTINGS"], [3, 3, 0, 4, 0, "AUDIO"]]))
    save("api_04_voice_design", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "cuda:0", "memory_mode": "auto", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}}, "3": {"class_type": "T8_FireRedAudio_VoiceDesign", "inputs": {"model": ["1", 0], "instruction": "青年女性，音色清亮，语速适中，语调自然亲切。", "text": "欢迎使用 FireRedAudio。", "settings": ["2", 0]}}, "4": {"class_type": "SaveAudio", "inputs": {"audio": ["3", 0], "filename_prefix": "fireredaudio/voice_design"}}})

    edit = node(4, "T8_FireRedAudio_SpeechEdit", [850, 220], ["adjust the speed to 1.2", "acoustic"], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2), port("settings", "T8_FIREREDAUDIO_SETTINGS", 3)], [out("audio", "AUDIO", [4]), out("edited_text", "STRING"), out("report", "STRING")])
    save_edit = node(5, "SaveAudio", [1280, 220], ["fireredaudio/edit"], [port("audio", "AUDIO", 4)], [out("audio", "AUDIO")], [280, 120])
    save("ui_05_speech_edit", workflow([model_node([1]), load_audio, settings, edit, save_edit], [[1, 1, 0, 4, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 4, 1, "AUDIO"], [3, 3, 0, 4, 2, "T8_FIREREDAUDIO_SETTINGS"], [4, 4, 0, 5, 0, "AUDIO"]]))
    save("api_05_speech_edit", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "cuda:0", "memory_mode": "auto", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}}, "3": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}}, "4": {"class_type": "T8_FireRedAudio_SpeechEdit", "inputs": {"model": ["1", 0], "audio": ["2", 0], "instruction": "adjust the speed to 1.2", "edit_type": "acoustic", "settings": ["3", 0]}}, "5": {"class_type": "SaveAudio", "inputs": {"audio": ["4", 0], "filename_prefix": "fireredaudio/edit"}}})

    locator = node(3, "T8_FireRedAudio_LongLocator", [500, 210], ["timeline_summary", "", True, 2048], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2)], [out("answer", "STRING"), out("structured_json", "STRING"), out("reasoning", "STRING"), out("report", "STRING")], [410, 300])
    save("ui_07_long_audio_locator", workflow([model_node([1]), load_audio, locator], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "AUDIO"]]))
    save("api_07_long_audio_locator", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "long_recording.wav"}}, "3": {"class_type": "T8_FireRedAudio_LongLocator", "inputs": {"model": ["1", 0], "audio": ["2", 0], "mode": "timeline_summary", "query": "", "enable_thinking": True, "max_new_tokens": 2048}}})

    acoustic_edit = node(4, "T8_FireRedAudio_AcousticEdit", [850, 220], ["speed", 3, 1.2, 1.0], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2), port("settings", "T8_FIREREDAUDIO_SETTINGS", 3)], [out("audio", "AUDIO"), out("instruction", "STRING"), out("report", "STRING")], [410, 300])
    save("ui_08_parametric_acoustic_edit", workflow([model_node([1]), load_audio, settings, acoustic_edit], [[1, 1, 0, 4, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 4, 1, "AUDIO"], [3, 3, 0, 4, 2, "T8_FIREREDAUDIO_SETTINGS"]]))
    save("api_08_parametric_acoustic_edit", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}}, "3": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}}, "4": {"class_type": "T8_FireRedAudio_AcousticEdit", "inputs": {"model": ["1", 0], "audio": ["2", 0], "operation": "speed", "pitch_steps": 3, "speed": 1.2, "volume": 1.0, "settings": ["3", 0]}}})

    quality = node(3, "T8_FireRedAudio_ReferenceQuality", [500, 220], [], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2)], [out("audio", "AUDIO"), out("quality_report", "STRING")], [390, 180])
    save("ui_09_reference_audio_quality", workflow([model_node([1]), load_audio, quality], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "AUDIO"]]))
    save("api_09_reference_audio_quality", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}}, "3": {"class_type": "T8_FireRedAudio_ReferenceQuality", "inputs": {"model": ["1", 0], "audio": ["2", 0]}}})

    compare_audio = node(3, "LoadAudio", [60, 650], ["comparison.wav", None, None], [], [out("AUDIO", "AUDIO", [3])], [300, 170])
    multi = node(4, "T8_FireRedAudio_MultiUnderstand", [520, 260], ["Compare these recordings and explain the relevant similarities and differences.", True, 1536], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio_0", "AUDIO", 2), port("audio_1", "AUDIO", 3)], [out("answer", "STRING"), out("reasoning", "STRING"), out("report", "STRING")], [430, 320])
    save("ui_10_multi_audio_understanding", workflow([model_node([1]), load_audio, compare_audio, multi], [[1, 1, 0, 4, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 4, 1, "AUDIO"], [3, 3, 0, 4, 2, "AUDIO"]]))
    save("api_10_multi_audio_understanding", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "voice_reference.wav"}}, "3": {"class_type": "LoadAudio", "inputs": {"audio": "comparison.wav"}}, "4": {"class_type": "T8_FireRedAudio_MultiUnderstand", "inputs": {"model": ["1", 0], "audio_0": ["2", 0], "audio_1": ["3", 0], "prompt": "Compare these recordings and explain the relevant similarities and differences.", "enable_thinking": True, "max_new_tokens": 1536}}})

    export_settings = node(2, "T8_FireRedAudio_GenerationSettings", [470, 80], ["balanced", 42, 750, 6, 512, 10, 2.0], [], [out("settings", "T8_FIREREDAUDIO_SETTINGS", [2])], [320, 280])
    export_design = node(3, "T8_FireRedAudio_VoiceDesign", [850, 180], ["A warm, calm narrator with clear articulation.", "This example exports an MP3 file."], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("settings", "T8_FIREREDAUDIO_SETTINGS", 2)], [out("audio", "AUDIO", [3]), out("report", "STRING")], [410, 250])
    export_audio = node(4, "T8_FireRedAudio_SaveAudio", [1310, 210], ["mp3", "voice_design", "fireredaudio/exports"], [port("audio", "AUDIO", 3)], [out("saved_path", "STRING")], [360, 210])
    save("ui_11_multiformat_audio_export", workflow([model_node([1]), export_settings, export_design, export_audio], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "T8_FIREREDAUDIO_SETTINGS"], [3, 3, 0, 4, 0, "AUDIO"]]))
    save("api_11_multiformat_audio_export", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "profile": "full", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "T8_FireRedAudio_GenerationSettings", "inputs": {"quality_preset": "balanced", "seed": 42, "max_new_audio_steps": 750, "min_new_audio_steps": 6, "max_new_text_tokens": 512, "n_timesteps": 10, "inference_cfg": 2.0}}, "3": {"class_type": "T8_FireRedAudio_VoiceDesign", "inputs": {"model": ["1", 0], "instruction": "A warm, calm narrator with clear articulation.", "text": "This example exports an MP3 file.", "settings": ["2", 0]}}, "4": {"class_type": "T8_FireRedAudio_SaveAudio", "inputs": {"audio": ["3", 0], "audio_format": "mp3", "filename_prefix": "voice_design", "subfolder": "fireredaudio/exports"}}})

    subtitle_asr = node(3, "T8_FireRedAudio_LongASR", [500, 180], ["Transcribe speech to text.", 30.0, 1.0, 300, 1.5], [port("model", "T8_FIREREDAUDIO_MODEL", 1), port("audio", "AUDIO", 2)], [out("transcript", "STRING"), out("srt", "STRING", slot_index=1), out("segments_json", "STRING", slot_index=2), out("report", "STRING", slot_index=3), out("vtt", "STRING", [3], slot_index=4), out("jsonl", "STRING", slot_index=5)], [390, 340])
    subtitle_save = node(4, "T8_FireRedAudio_SaveSubtitle", [990, 250], ["vtt", "long_audio", "fireredaudio/subtitles"], [port("content", "STRING", 3)], [out("saved_path", "STRING")], [360, 210])
    save("ui_12_subtitle_export", workflow([model_node([1]), load_audio, subtitle_asr, subtitle_save], [[1, 1, 0, 3, 0, "T8_FIREREDAUDIO_MODEL"], [2, 2, 0, 3, 1, "AUDIO"], [3, 3, 4, 4, 0, "STRING"]]))
    save("api_12_subtitle_export", {"1": {"class_type": "T8_FireRedAudio_ModelLoader", "inputs": {"model_name": "FireRedAudio", "custom_model_path": "", "device": "auto", "memory_mode": "auto", "profile": "lite", "worker_mode": "managed", "runtime_python": "", "worker_url": "", "worker_token": "", "verify_hashes": False, "release_after_run": False}}, "2": {"class_type": "LoadAudio", "inputs": {"audio": "long_recording.wav"}}, "3": {"class_type": "T8_FireRedAudio_LongASR", "inputs": {"model": ["1", 0], "audio": ["2", 0], "prompt": "Transcribe speech to text.", "chunk_seconds": 30.0, "overlap_seconds": 1.0, "max_new_tokens": 300, "silence_search_seconds": 1.5}}, "4": {"class_type": "T8_FireRedAudio_SaveSubtitle", "inputs": {"content": ["3", 4], "text_format": "vtt", "filename_prefix": "long_audio", "subfolder": "fireredaudio/subtitles"}}})


if __name__ == "__main__":
    build()
