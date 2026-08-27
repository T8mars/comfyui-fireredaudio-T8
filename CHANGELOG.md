# Changelog

## 0.13.0

- 新增“长录音参考候选”节点：非破坏提取 3–15 秒候选，保留源采样率/声道，按削波、静音、能量对比、语音活动和时长排序，并可使用明确标注为代理而非准确率的 ASR 可懂度复排。
- 新增“多 Take 试听评审板”输出节点：显示 2–8 条 ComfyUI 原生播放器/下载入口，记录 1–5 分、备注与唯一采用项到独立 Manifest；不修改源 AudioBatch/WAV。
- 多 Seed 试音改用不暴露 Seed 的 `take-001.wav` 文件名，Seed 继续完整保存在审计 Manifest，支持真正盲听。
- 新增“加速实测向导”：固定参考、逐字稿、目标文本、Seed 和参数，逐模式暖机并至少正式测量三次，比较中位耗时、RTF、峰值显存和 SHA-256；识别实际回退，只给建议不自动修改加载器。
- 新增第 23–25 组 UI/API 工作流，打通长录音筛参考 → 人工试听 → ASR → VoiceProfile、多 Seed 盲听评分与真实加速 A/B。
- 无真模型执行级验收覆盖源哈希保全、原生试听、唯一采用、9 次正式基准和回退排除；节点总数增至 39，宿主继续零依赖，隔离 Worker 继续固定 Transformers 5.8。
- 真模型 RTX 5090 Laptop / `sequential` 实测三模式各 1 次暖机 + 3 次正式运行：`off` 24.182 秒、FlashAttention 21.179 秒（快 12.418%）、DeepSpeed 20.659 秒（快 14.569%），均无回退且同模式输出哈希可复现；按实验模式 20% 门槛推荐 FlashAttention。
- 修复生成参数界面已有 `custom` 选项但 Worker 协议拒绝的问题；Worker 现在会校验并接受自定义步数、Token、扩散步数和 CFG 范围。

## 0.12.0

- 新增“局部修复范围”和“局部修复回填 A/B”节点：支持手工时间或长音频定位 JSON，裁出含上下文的修复片段，编辑后验证源哈希并非破坏回填。
- 回填会保留原音频采样率和声道数，安全适配编辑片段的采样率/声道，在替换区域边缘使用等功率交叉淡化；报告记录时间范围、时长变化、哈希、峰值与源文件保全状态。
- 新增“分轨制作交付包”输出节点：按角色/场景导出等长对白分轨，并收集 Master、dialogue、BGM、room tone、实际 SRT/VTT、版本、生成参数、交付预设和素材 SHA-256 到 Manifest/ZIP。
- 角色脚本新增 `# 场景：名称`、`# Scene: name` 和 `## 名称` 标题；JSON 与桌面项目交换中的 `scene` 会贯穿 ScriptPlan、AudioBatch 与场景分轨。
- 新增第 21–22 组 UI/API 工作流和无真模型执行级验收；覆盖 48 kHz 立体声源、24 kHz 单声道编辑片段、角色/场景分轨、制作素材、字幕、ZIP 及审计信息。
- 节点总数增至 36；宿主继续零依赖，隔离 Worker 继续固定 Transformers 5.8，模型仍外置。

## 0.11.0

- “可恢复批量配音”改用 Worker `/v1/infer/tts-batch` 分块执行；支持不同角色/参考音频，在顺序显存模式中先生成一批 latent 再统一解码，并继续保留逐条指纹缓存、原子 manifest、失败隔离和 ComfyUI 取消。
- 新增“QA 失败项定向返修”节点：直接消费 Speech QA 的失败 line ID，只返修目标台词；支持固定/递增 Seed、最多五次尝试、独立返修文件与非破坏合并 manifest。
- 新增“AudioBatch 试听选择”节点：可按成功序号、line ID 或角色选出原生 AUDIO，并返回条目详情和批次摘要。
- 新增“批量保存/下载”输出节点：批量导出 WAV/FLAC/MP3/OGG、注册 ComfyUI 原生试听/下载列表，并生成便携 manifest 与可选 ZIP。
- 新增第 20 组完整创作闭环 UI/API 示例，以及无真模型执行级验收脚本，覆盖 `2+1` 分块、Manifest 恢复、QA Seed 返修、源文件不覆盖和 ZIP 交付。
- 新 `BatchDubbing` 包装层真模型两句 Smoke 通过：2/2 有效 WAV、单次 `latent_first_decode_later` 批次；重新执行 2/2 Manifest 命中、0.030 秒完成且输出哈希不变。
- 节点总数增至 33；宿主继续零依赖，隔离 Worker 继续固定 Transformers 5.8，模型仍外置。

## 0.10.0

- 新增“参考音频 ASR 逐字稿”节点；逐字稿可直接连接 TTS/音色档案，TTS 留空时也可自动 ASR 后继续生成并回传实际逐字稿。
- 新增“多 Seed 试音/推荐 Take”节点与 Worker 批量端点：一次生成 2–8 个候选，在顺序卸载模式中采用 latent-first、统一切换解码器，可选逐条 ASR 回读后生成可审计推荐结果。
- 新增“定位证据片段/剪辑清单”节点，把长音频定位结构化 JSON 转成持久化 WAV、AudioBatch、时间范围与 manifest。
- “保存音频”升级为正式输出节点，结果区提供 ComfyUI 原生播放器/下载入口，并继续安全限制在 output 目录。
- 新增分阶段性能分析节点，显示冷/热启动、阶段耗时、RTF、总耗时和 CUDA 峰值显存。
- 时间线与证据裁切保留立体声；混合单声道素材时自动提升声道，不再整体降混。
- 新增第 17–19 组 UI/API 示例、每周上游代码/模型 revision 监测，以及 Python 3.10–3.13 / 宿主 Transformers 4.x–5.x CI 矩阵。
- RTX 5090 Laptop 双 Seed 真模型 Smoke 通过：2/2 WAV 有效，报告确认 `latent_first_decode_later`、第二条参考条件缓存命中和一次批量解码器切换。
- 节点总数增至 30；隔离 Worker 继续固定 Transformers 5.8，宿主 requirements 保持零依赖。

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
