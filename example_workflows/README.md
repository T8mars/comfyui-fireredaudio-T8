# 示例工作流

- `01_zero_shot_tts`：参考音频声音克隆
- `02_asr`：多语言语音识别
- `03_audio_understanding`：音频问答与 thinking
- `04_voice_design`：自然语言声音设计
- `05_speech_edit`：语义/声学语音编辑
- `06_long_audio_asr`：分段长音频转写、时间戳 JSON 与 SRT
- `07_long_audio_locator`：长录音时间线、时间找内容与内容找时间
- `08_parametric_acoustic_edit`：参数化调整音高、速度或音量
- `09_reference_audio_quality`：参考音频响度、削波、静音和格式质检
- `10_multi_audio_understanding`：动态多音频比较、说话人与事件对照
- `11_multiformat_audio_export`：生成后直接导出 MP3，可改为 WAV/FLAC/OGG
- `12_subtitle_export`：长音频转写后直接保存 WebVTT，可切换 SRT/TXT/JSONL
- `13_role_dubbing_pipeline`：双角色音色库、角色脚本预检、可恢复批配音、时间线混音与成品 QA
- `14_srt_dubbing_pipeline`：SRT 角色绑定、严格预检、可恢复配音与按字幕时间码渲染

`ui` 目录可直接拖入 ComfyUI；`api` 目录用于 `/prompt` API。请先把示例引用的 `voice_reference.wav`、`comparison.wav`、`long_recording.wav`、`narrator_reference.wav` 或 `actor_reference.wav` 上传到 ComfyUI input，并按实际扫描结果修改模型加载器中的模型名称。

V3 Autogrow 输入在 `/prompt` API 中必须使用点路径，例如 `profiles.profile_0` 或 `audios.audio_0`；示例 API 已按此格式提供，不能改成未分组的 `profile_0`/`audio_0`。

长音频 ASR v2 还会输出 WebVTT 与 JSONL。字幕时间为静音感知的分段级近似时间，不是词级强制对齐。
