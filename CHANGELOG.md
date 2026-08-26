# Changelog

## 0.2.0

- 新增长音频分段 ASR、时间戳 JSON 与 SRT 输出。
- 新增快速、均衡、高质量和自定义生成参数模式。
- ComfyUI 进度条现在同步隔离 Worker 状态；停止队列会联动取消推理。
- 运行时控制节点增加 `cancel_current`。
- 增加 Worker 崩溃恢复、下载磁盘预检和阶段进度。
- 在 Transformers 4.52.1、4.57.6 宿主上继续使用隔离 Transformers 5.8 运行时。

## 0.1.0

- 首个完整节点发行版：ASR、音频理解、零样本 TTS、声音设计和语音编辑。
