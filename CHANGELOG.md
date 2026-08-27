# Changelog

## 0.9.0

- 新增“同步 A/B 对比”节点：非破坏同步有效声音起点、两遍 EBU R128 响度匹配、等长补齐，并输出可审计的处理前后报告。
- 新增“交付预设”节点：提供有声书、播客和视频对白三组排列、交叉淡化、LUFS、LRA、True Peak、采样率及保存格式设置。
- 时间线渲染新增相邻对白等功率交叉淡化，以及使用可选 room tone 仅填充真实空隙的能力；交付预设会在渲染后执行两遍母带处理。
- 保存音频可连接同一交付预设并自动采用预设格式；第 14 组 SRT 示例升级为 room tone + 视频对白交付，第 16 组示例演示同步 A/B。
- RTX 5090 Laptop、外置真模型、`sequential + auto_safe` 连续 20 轮 TTS 通过：20/20 WAV 有效，暖机后中位耗时 15.184 秒，首尾五轮任务后显存中位数均为 574 MiB、漂移 0 MiB。
- 节点总数增至 26；默认加速仍为 `auto_safe`，DeepSpeed 继续要求同工作流暖机中位数至少快 20% 才值得手动采用。

## 0.8.0

- Worker 项目接口同步 schema v7：安全脚本替换/备份、归档恢复、单任务取消、发音词表与轻量制作轨。
- 项目生成任务区分字幕显示文本和实际朗读文本，保留原脚本并记录命中的发音词条。
- 时间线渲染支持 BGM、环境声和 room tone 循环、增益、淡化及对白 ducking；制作轨不改变对白排布。
- 保持宿主零依赖、隔离 Transformers 5.8、预编译加速轮子与模型外置边界；本版未新增 ComfyUI 节点类型。

## 0.7.0

- Worker 新增按文件指纹失效的参考波形与 RedAE/Patch 条件 LRU，同一角色批量 TTS 不再重复计算参考条件；缓存随模型进程卸载，不写入项目。
- Runtime 状态新增缓存命中、未命中、失效、淘汰、CPU 占用和首次编码耗时诊断。
- 项目清单原子替换增加 Windows 短时文件锁退避，100 行、8 角色、中断恢复和 8 条 latent 批次压力夹具通过。
- 继续保持隔离 Transformers 5.8、单 GPU、预编译 FlashAttention/DeepSpeed wheel 和 SDPA 回退；模型仍外置。

## 0.6.0

- 新增“参考音频非破坏清理”节点与第 15 组 UI/API 示例工作流；清理副本不会覆盖源文件，并保留处理前后质检与完整参数记录。
- 修正所有 UI 示例的模型加载器控件序列，确保默认加速模式实际为 `auto_safe`，并增加全量工作流回归测试。
- Worker 同步桌面时间线批量保存/删除 API、参考音频清理端点和 50 句队列恢复验证。
- 环境报告统一为 Python 3.10、Torch/Torchaudio 2.8.0+cu128、Transformers 5.8，继续使用预编译 FlashAttention/DeepSpeed Windows wheel，禁止源码编译。
- 节点总数增至 24；多卡仍不启用。

## 0.5.0

- 隔离运行时对齐 Python 3.10、Torch/Torchaudio 2.8.0+cu128、Transformers 5.8。
- 内置经哈希和 ABI 校验的 FlashAttention 2.8.3、DeepSpeed 0.17.5、Triton-Windows、FLA 与 Liger；禁止从源码构建 FlashAttention/DeepSpeed。
- 模型加载器新增自动安全、FlashAttention、DeepSpeed BF16、FLA+Liger 与 SDPA 基线选择，失败原因可诊断并安全回退。
- DeepSpeed、FlashAttention、FLA/Liger 均通过真实 FireRedAudio 短 TTS；DeepSpeed 单卡尚未证明性能收益，保持手动实验。
- 同步桌面项目的音色双向同步、时间线拖动和撤销/重做交换数据。

## 0.4.0

- 新增音色档案、1–8 音色库，以及 SRT/角色脚本/JSON 解析预检节点。
- 新增逐条原子 manifest、输入指纹缓存和中断恢复的批量配音节点；保持隔离 Worker 与 Transformers 兼容边界。
- 新增 sequence、timeline、overlay 时间线渲染，以及峰值限制和时间槽溢出报告。
- 新增基于 ASR 回读、中文 CER/英文 WER、削波、静音和时长的成品语音 QA。
- 新增完整角色配音与 SRT 配音两组 UI/API 示例工作流。
- 新增桌面项目交换节点，载入角色音色库、脚本计划与 adopted take。
- Worker 同步声音设计退化门禁、分阶段性能证据与 Transformers 隔离兼容报告。

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
