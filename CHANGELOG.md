# Changelog

## 0.3.0

- 新增长音频静音感知切段、重叠文本去重，以及 SRT、WebVTT、JSONL 输出。
- 新增长音频时间线、时间找内容、内容找时间与结构化摘要节点。
- 新增 1–8 路动态多音频比较理解节点。
- 新增参数化声学编辑和参考音频质量分析节点。
- 新增 WAV、FLAC、MP3、OGG 音频以及 SRT、VTT、TXT、JSONL 安全保存节点。
- GPU 列表、实时空闲显存、自动设备与缓存管理现在可由节点运行时控制查看。
- 使用 ComfyUI V3 官方进度接口，并继续用隔离 Worker 兼容宿主 Transformers 4.x 与运行时 Transformers 5.8。
- 示例扩充到 12 组 UI/API 工作流。

## 0.2.0

- 新增长音频分段 ASR、时间戳 JSON 与 SRT 输出。
- 新增快速、均衡、高质量和自定义生成参数模式。
- ComfyUI 进度条现在同步隔离 Worker 状态；停止队列会联动取消推理。
- 运行时控制节点增加 `cancel_current`。
- 增加 Worker 崩溃恢复、下载磁盘预检和阶段进度。
- 在 Transformers 4.52.1、4.57.6 宿主上继续使用隔离 Transformers 5.8 运行时。

## 0.1.0

- 首个完整节点发行版：ASR、音频理解、零样本 TTS、声音设计和语音编辑。
