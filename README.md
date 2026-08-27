# comfyui-fireredaudio-T8

FireRedAudio 的 43 个 ComfyUI V3 全能力节点，由 **T8star-Aix** 集成。节点菜单：

`T8star-Aix / Audio / FireRedAudio`

## 能力

- 多语言 ASR
- 长音频静音感知分段、重叠去重，输出 SRT、VTT、JSON 与 JSONL
- 原生长录音时间线、时间找内容、内容找时间和结构化摘要
- 单音频理解，以及 1–8 个动态音频比较问答，可选 thinking
- 中英文零样本声音克隆
- 参考音频一键 ASR 逐字稿；可连接到 TTS/音色档案，TTS 留空时也可自动转写后继续生成
- 2–8 个 Seed latent-first 批量试音，可选逐条 ASR 回读并按文本误差、削波、静音和时长稳定性推荐 Take
- 长录音参考候选：非破坏提取 3–15 秒片段，按削波、静音、能量对比、语音活动和时长排序，可选明确标注为代理指标的 ASR 可懂度复排
- 多 Take 试听评审板：使用 ComfyUI 原生播放器/下载入口，记录 1–5 分、备注和唯一采用项，不修改候选音频
- 自然语言声音设计
- 语义语音编辑
- 严格模板和参数化两种音高、速度、音量声学编辑
- 局部修复闭环：手工时间或定位 JSON 裁片 → 语义/声学编辑 → 等功率交叉淡化回填；保留源声道/采样率并输出原版/修复版 A/B 和哈希报告
- 参考音频时长、响度、削波、静音、直流偏移质检
- 另存 24 kHz 单声道的非破坏参考音频清理副本，可选静音裁剪、-23 LUFS 和语音高通，保留前后质检与处理记录
- 可复用音色档案与 1–8 角色音色库
- 载入桌面整合包导出的项目交换 JSON，复用音色库、脚本和 adopted take
- SRT、`角色：台词`、带时间码角色脚本和 JSON 的解析与生成前预检
- 朗读文本规范化：保留原文并单独记录实际送入 TTS 的文本，支持自定义替换词典、全角/空白清理、中文日期和可选数字展开
- Worker `tts-batch` 分块、逐条落盘、原子 manifest、内容指纹缓存和中断恢复的批量配音；不同角色/参考音频可进入同一批 latent-first/decode-later
- 同一参考音频的读取、重采样、RedAE 与 Patch Encoder 条件进程内复用；文件变化或模型卸载时安全失效
- 同步有效声音起点、两遍 EBU R128 响度匹配和等长补齐的公平 A/B 对比
- 顺序、时间码定位和叠加三种时间线渲染，相邻对白交叉淡化，以及可选 room tone 自动补真实空隙
- 保留立体声制作素材；混合单声道对白时自动提升声道数，不再把整条制作时间线降混为单声道
- 将长音频定位 JSON 直接裁成可试听证据片段、剪辑清单和可继续制作的 AudioBatch
- 有声书、播客、视频对白交付预设，统一控制排列、LUFS、LRA、True Peak、采样率和保存格式
- 成品逐条 ASR 回读、中文 CER/英文 WER、削波、静音和时长 QA
- 字幕时长适配：报告每句时间槽差异，在安全倍率内使用 FFmpeg atempo 保持音高地非破坏适配，超限条目进入重做清单
- 逐句制作审核台：在节点内按行试听、下载、过滤 QA 建议，选择通过/人工复核/重做并记录评分和备注；仅通过项形成独立交付 AudioBatch
- 从批量、返修、时长适配或审核 Manifest 恢复 AudioBatch，保留审核状态并显式标记缺失文件，支持跨会话继续
- QA 失败 line ID 定向返修；支持固定/递增 Seed、多次尝试、字幕时间槽门禁、独立返修文件和非破坏 AudioBatch/manifest 合并
- AudioBatch 可按序号、line ID 或角色试听选择；成功 Take 可批量导出、原生下载并打包 ZIP
- 分轨制作交付包：角色/场景等长对白分轨、Master、dialogue、BGM、room tone、实际 SRT/VTT、模型版本、参数与素材 SHA-256 一键 ZIP
- WAV、FLAC、MP3、OGG 音频与 SRT、VTT、TXT、JSONL 安全保存；保存音频节点在 ComfyUI 结果区提供原生播放器/下载入口
- 冷/热启动、分阶段耗时、RTF 和 CUDA 峰值显存性能报告
- 加速实测向导：固定参考、文本、Seed 和参数，对 `off`、FlashAttention、DeepSpeed 等模式分别暖机并至少正式测量 3 次，比较中位耗时、RTF、峰值显存、哈希和实际回退，只给建议不自动改设置
- 模型校验、真实 GPU/空闲显存报告、缓存清理、显存释放和 Worker 停止
- ComfyUI V3 进度同步、中断联动取消、快速/均衡/高质量预设

## 安装

在 ComfyUI-Manager 中搜索 `comfyui-fireredaudio-T8` 并安装，然后重启 ComfyUI。也可以手动安装：

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/comfyui-fireredaudio-T8.git
cd comfyui-fireredaudio-T8
python scripts/setup_runtime.py
```

Manager 安装不会改动 ComfyUI 的 Python 依赖。首次使用前仍需运行上面的 `setup_runtime.py`，创建专用于 FireRedAudio 的隔离 Worker 环境。

## Transformers 兼容方式

本节点的已验证隔离运行时固定为 Python 3.10、PyTorch 2.8.0+cu128 和 Transformers 5.8.0。许多 ComfyUI 节点仍依赖 Transformers 4.x，因此本节点**不会**把 FireRedAudio 依赖安装进 ComfyUI：

```text
ComfyUI Python / Transformers 4.x
        │ 标准 AUDIO + 本地 RPC
        ▼
隔离 Python 3.10 / torch 2.8 cu128 / Transformers 5.8 Worker
```

`requirements.txt` 有意保持为空。Windows 节点发行包随附固定版本 `uv`，请显式准备隔离环境：

```powershell
python scripts/setup_runtime.py
```

也可在模型加载器中填写其他隔离 `python.exe`，或连接桌面整合包启动的外部 Worker。节点绝不调用 `pip` 修改 ComfyUI 环境。

模型加载器的设备默认使用 `auto`。实际 GPU、空闲显存和最终选用的显存模式可通过“运行时控制”节点查看；`auto` 会在空闲显存不足 36 GiB 时选择顺序卸载，而不是只看显卡标称总显存。

加速模式默认 `auto_safe`，使用预编译 FlashAttention 2 wheel。DeepSpeed BF16 已通过单卡完整 TTS，但仍是手动实验模式；FLA+Liger 在没有可审计 `causal-conv1d` Windows wheel 时会明确显示部分 PyTorch 回退。安装器从固定 URL 下载预编译 wheel、校验 SHA-256 和 Python/Torch/CUDA ABI，不从源码编译，也不修改 ComfyUI 宿主环境。

普通单卡优先使用 `auto_safe`；`off` 用作 SDPA 对照与排障；DeepSpeed、FLA+Liger 和 Torch Compile 只有在相同真实工作流的暖机多轮中位数至少快 20% 时才建议手动选择。当前不启用多卡。

v0.7 起，同一 Worker 中重复使用相同参考音频会命中 CPU LRU 条件缓存，避免每句重复执行 RedAE/Patch Encoder。缓存按绝对路径、大小和修改时间失效，随模型 Worker 释放，不写入 ComfyUI 工程，也不长期占用 GPU 显存；命中和占用可在运行时状态中审计。

## 模型

默认扫描：

```text
ComfyUI/models/TTS/FireRedAudio/
  FireRedAudio/
    config.json
    model-00001-of-00005.safetensors
    ...
  RedAE_decoder/
    model.pt
```

Lite 仅下载主模型，支持 ASR/理解；Full 增加 RedAE decoder，支持生成和编辑：

```powershell
python scripts/download_models.py --target "D:\ComfyUI\models\TTS\FireRedAudio" --profile full
```

## 基础工作流

1. `Load Audio` → `FireRedAudio 零样本声音克隆` 的参考音频输入。
2. `FireRedAudio 模型/隔离运行时` → TTS 的模型输入。
3. 可选连接 `FireRedAudio 生成参数`。
4. TTS 输出连接 `FireRedAudio 保存音频`，可选择 WAV/FLAC/MP3/OGG。

示例见 `example_workflows/ui` 和 `example_workflows/api`。`15_reference_cleanup` 演示参考音频质检后再生成不覆盖源文件的清理副本；`16_synchronized_ab` 演示两个候选同步起点、匹配响度并分别保存；`17_reference_asr_tts` 演示自动逐字稿驱动 TTS；`18_seed_audition` 演示多 Seed 试音；`19_long_audio_evidence` 演示定位结果直接生成证据片段；`20_creator_qa_repair_delivery` 演示分块批配音、QA 失败返修、试听选择和批量 ZIP；`21_podcast_local_repair` 演示手工范围、语义编辑、回填与 A/B 保存；`22_production_package` 演示从项目交换 AudioBatch 导出角色/场景分轨和完整制作包；`23_long_reference_screening` 演示长录音筛选、人工试听、ASR 和 VoiceProfile；`24_seed_blind_review` 演示隐藏 Seed 文件名的多 Take 盲听评分与采用；`25_acceleration_benchmark` 演示真实加速实测；`26_production_review_loop` 演示规范化、字幕适配、QA、两轮逐句审核/返修和最终导出；`27_resume_review_session` 演示重启后从审核 Manifest 继续。参考清理和候选筛选都不会执行降噪、去混响或削波伪修复。

长音频字幕是静音感知的分段级近似时间，不是词级强制对齐。批量配音和多 Seed 试音都使用 Worker `tts-batch`；批量配音默认每批 8 条并允许不同角色参考，在顺序卸载模式中先完成本批 latent，再统一切换解码器，减少逐句切换大模型。每条结果仍独立落盘、记录和恢复。

## 配音生产工作流

推荐连接顺序：

```text
Load Audio → 音色档案 ┐
Load Audio → 音色档案 ├→ 音色库 → 角色脚本/SRT 预检 → 朗读文本规范化 → 可恢复批量配音
                       └───────────────────────────────────────────┬→ 字幕时长适配 → 成品语音 QA
模型/隔离运行时 ───────────────────────────────────────────────────┘                    │
                                                                                       ▼
                                                               逐句制作审核台 → 仅通过项交付
                                                                      │重做 line ID
                                                                      ▼
规范化脚本 + 音色库 + 带审核 AudioBatch ───────────────────────→ 定向返修 → 再次 QA/审核

原始长音频 → 局部修复范围 → 语义/声学编辑 → 局部修复回填 A/B → 分别试听/保存
```

制作交付可继续连接：

```text
候选 A ┐
       ├→ 同步 A/B 对比 → 分别试听/保存
候选 B ┘

可恢复批量配音 ────────────────┐
room tone ─────────────────────┼→ 时间线渲染 → 保存音频
有声书/播客/视频对白交付预设 ───┘        └→ 同一交付预设

AudioBatch + 可选 Master/BGM/room tone/字幕 → 分轨制作交付包 → Manifest + ZIP

长录音 → 参考候选筛选 → 多 Take 试听评审 → ASR 核对逐字稿 → VoiceProfile

固定参考/逐字稿/目标文本/Seed → 加速实测向导 → 人工把建议模式填回模型加载器

审核 Manifest → 恢复批次/审核会话 → 逐句制作审核台 → 继续交付或返修
```

交叉淡化只作用于相邻对白的真实重叠区域；`sequence` 模式启用交叉淡化时，相邻重叠会替代顺序间隔。自动补空隙必须连接 room tone，渲染器只在没有对白覆盖的时间段循环填充，并在边缘加入短淡化，不会把背景声铺到对白上。

交付预设会执行两遍 FFmpeg EBU R128 规范化并限制 True Peak。`audiobook` 默认 -20 LUFS / -3 dBTP / FLAC，`podcast` 默认 -16 LUFS / -1 dBTP / WAV，`video_dialogue` 默认 -23 LUFS / -1 dBTP / WAV；预设同时覆盖排列方式、交叉淡化和输出采样率，报告中会保存实际参数。

若脚本已在桌面整合包中整理，可点击“导出 ComfyUI 项目 JSON”，再用“FireRedAudio 桌面项目交换”节点一次还原 Voice Bank、Script Plan 和已采用的 Audio Batch。节点会验证参考音频 SHA-256；文件移动或被改写时会明确报错。

角色脚本支持以下形式：

```text
旁白：欢迎收听本期节目。
[00:00:03,000 --> 00:00:06,000] 小夏：大家好，我是小夏。
# 场景：访谈
旁白：这一句会记录到“访谈”场景分轨。
```

SRT 的正文可用 `[角色] 台词` 标记角色。角色脚本支持 `# 场景：名称`、`# Scene: name` 或 `## 名称` 场景标题。JSON 可直接传数组，或传包含 `lines` 数组的对象；每项支持 `speaker`、`scene`、`text`、`language`、`start_seconds`、`end_seconds`。

批量配音输出到 `ComfyUI/output/<subfolder>/<project_name>/`。`manifest.json` 会在每条成功或失败后原子更新；启用恢复后，只有台词、音色参考、生成参数和模型身份指纹均一致且 WAV 仍存在的条目才会命中缓存。未命中条目按 `batch_size` 分块进入 Worker `tts-batch`；报告保存每批真实 execution model 和性能数据。若 ComfyUI 中断当前批次，已完成并写入 manifest 的条目仍可在下次运行复用。

`SpeechQA.failed_line_ids` 可直接连接“QA 失败项定向返修”。返修始终写入独立目录和 `repair-manifest.json`，不会覆盖原始或已通过音频；成功条目才会替换输出 AudioBatch 中对应 line ID。默认启用字幕时间槽门禁：脚本带时间码时，每次新 Take 都会重新测量真实 WAV，仍超过允许值就继续下一 Seed，全部尝试仍不合格则保持失败状态，不会把“生成成功”误当作“返修合格”。返修后的 AudioBatch 可连接“试听选择”“批量保存/下载”或时间线；批量保存会生成相对独立的交付目录、`export-manifest.json` 和可选 ZIP，结果区默认显示最多 16 条原生播放器/下载入口。

“朗读文本规范化”不会改写输入脚本对象；输出 ScriptPlan 的 `source_text` 保留原文，`text` 是实际朗读文本，命中的规则写入 `normalization`。自定义词典使用 JSON 对象并按长词优先替换。中文数字展开默认关闭，因为型号、编号和专有名词可能需要不同读法；应先查看原文/朗读文本对照，再连接批量配音。

“字幕时长适配”只处理带 `start_seconds/end_seconds` 的成功 WAV。默认最大安全加速为 1.15 倍，使用 FFmpeg `atempo` 保持音高并写入新文件；所需倍率超过上限时不截断、不强拉伸，而是输出重做 line ID。`report_only` 只生成报告。所有分支都保留源路径、源 SHA-256、实际倍率和适配后时长。

“逐句制作审核台”把 QA 结果归为通过、人工复核或建议重做：仅静音比例异常默认人工复核，文字、削波或时间槽失败默认建议重做，人工选择始终可以覆盖。节点内表格的决定、1–5 分和备注会同步到工作流可序列化输入；重新运行节点后写入独立审核 Manifest。`approved_batch` 会把未通过项置为 `review_hold`，因此时间线和导出不会悄悄混入待复核素材。返修成功时旧审核结论进入 `previous_human_review`，新 Take 重新接受 QA/人工判断。

“恢复批次/审核会话”默认只允许读取 ComfyUI output 内的 Manifest；相对路径按 output 根解析。缺失 WAV 可选择标记为 `missing` 或立即报错，已记录的 SHA-256 可选复核。确需读取外部制作目录时必须显式打开高级选项；恢复不会运行模型，也不会修改 Manifest 或音频。

“局部修复范围”可直接填写秒数，也可连接“长音频时间定位”的结构化 JSON 并按序号选择范围。两侧上下文会一起送入编辑模型并作为整体非破坏回填；“局部修复回填 A/B”会验证源 SHA-256，自动把编辑片段重采样/适配回原声道数，在替换区域内部做等功率边缘混合，源文件不会被覆盖。编辑片段时长变化会明确写入报告，不会偷偷拉伸。

“分轨制作交付包”使用时间线的实际落点生成 SRT/VTT，角色和场景分轨会用静音补齐到同一 Master 终点，便于直接导入 DAW。连接的外部 Master、BGM 和 room tone 会原样收进包；该节点不会假装替用户完成未指定的 BGM 混音，Manifest 会明确记录每项素材是否由本节点混入。ZIP 同时保存上游代码/模型 revision、Worker revision、生成参数、交付预设和全部产物 SHA-256。

“长录音参考候选”保留源采样率和声道，源文件前后 SHA-256 必须一致。信号排序不是主观音质预测；启用 ASR 时也只根据非空文本、语速和重复度生成明确标注的可懂度代理，不能冒充有真值逐字稿的 WER/CER。推荐候选仍须进入“多 Take 试听评审板”人工试听，再通过“参考音频 ASR 逐字稿”核对文字后生成 VoiceProfile。

“多 Take 试听评审板”本身继续采用纯节点实现：候选通过 ComfyUI 官方原生音频预览协议显示，支持播放和下载；评分、备注与选中 line ID 写入新的 review manifest，每批只有一个 `adopted=true`，源 AudioBatch 和 WAV 不变。v0.14 的“逐句制作审核台”另带一个本地前端表格，但音频 URL 仍使用官方 `/view` 协议，不增加网络请求。多 Seed 试音的文件名改为不暴露 Seed 的 `take-001.wav` 等，Seed 只留在审计 Manifest，适合盲听。

“加速实测向导”会自动补入 `off` 基线，每种模式先暖机，再正式运行 3–9 次；缺轮子或 ABI 不匹配导致的回退不会被误当成该模式成绩。FlashAttention 等普通候选默认至少快 10%，DeepSpeed/FLA/Torch Compile 等实验模式至少快 20%，并可要求同 Seed 多轮 SHA-256 一致后才推荐。节点结束会释放模型，只输出建议和完整证据，不会修改模型加载器。

## 真模型长跑验收

可在目标机器运行节点仓库自带的验收脚本；模型保持外置：

```powershell
python scripts/validate_real_model_long_run.py `
  --model-root "D:\ComfyUI\models\TTS\FireRedAudio" `
  --reference-audio "D:\ComfyUI\input\voice_reference.wav" `
  --rounds 20 --device cuda:0 --memory-mode sequential --acceleration-mode auto_safe
```

v0.9.0 在 RTX 5090 Laptop 上完成 20/20 轮真实 TTS，所有 WAV 均通过大小、时长和 SHA-256 取证；暖机后任务中位耗时 15.184 秒，CUDA 峰值分配约 19.852 GiB，任务结束显存首五轮/末五轮中位数均为 574 MiB，漂移 0 MiB。该结果只证明这台目标机器和该工作流，不外推成所有显卡的固定速度。

v0.10.0 的多 Seed 专用链路另做了 2 条真模型 Smoke：Seed 4200/4201 均生成有效 24 kHz WAV，输出 5.28/4.96 秒且哈希不同；Worker 报告执行模型为 `latent_first_decode_later`，第二条复用参考条件并只进行一次批量解码器切换。冷启动总耗时包含 67.714 秒模型加载，不能拿该值代表暖机速度。

v0.11.0 的新 `BatchDubbing` 包装层使用两句真实 TTS 做了独立 Smoke：2/2 生成有效 24 kHz WAV，单次 Worker 批次报告为 `latent_first_decode_later`，冷启动总耗时 120.923 秒；同一节点重新执行时 2/2 命中 Manifest，耗时 0.030 秒且两条文件 SHA-256 均未变化。该测试证明分块、落盘与恢复链路，不把冷启动时长当作通用速度结论，也没有运行 ASR。

v0.13.0 的无真模型执行级验收覆盖长录音候选、原生试听、人工采用和加速决策算法：源哈希不变、唯一采用项成立，模拟 `off/flash_attention/deepspeed` 每模式 3 次正式测量共 9 条，正确推荐有效快路径并排除实际回退。具体机器的速度结论必须在该机器运行第 25 组工作流，不能用模拟验收数字代替。

v0.14.0 的执行级验收覆盖原文/朗读文本双轨、自定义词典与中文日期、1.10 秒音频安全适配到 1.00 秒时间槽、1.50 秒超限条目进入重做、QA 路由、人工覆盖、两行内嵌播放器/下载和审核 Manifest 跨会话恢复；源 WAV 前后 SHA-256 均未变化。

同版本在 RTX 5090 Laptop、24 GiB、`sequential + auto_safe` 上完成两句真模型完整创作链验收：2/2 批量生成、第一句以 1.08 倍安全适配、第二句以 1.25 倍需求进入重做、ASR 回读 CER 为 0、QA 正确拦截时间槽失败、只返修问题句、终审/恢复/双 WAV 与 ZIP 交付均完成；总耗时 129.828 秒，CUDA 峰值分配约 19.85 GiB，源文件 SHA-256 未变化。可用 `scripts/validate_v014_real_production.py` 在目标机器复验；真实声音审美仍需人工试听，自动指标不能代替听感。

同版本随后在 RTX 5090 Laptop、24 GiB、`sequential` 上完成真模型第 25 组等价实测：三种模式均无回退且每模式 3/3 正式 WAV 同 Seed 哈希可复现；`off` 中位 24.182 秒 / RTF 6.297，FlashAttention 21.179 秒 / RTF 5.515（快 12.418%），DeepSpeed 20.659 秒 / RTF 5.380（快 14.569%），峰值显存均约 19.850 GiB。按普通候选 10%、实验模式 20% 门槛推荐 FlashAttention；DeepSpeed 虽略快但未达到实验采用门槛。此结论仍只适用于这台机器和固定短句工作流。

仓库每周读取固定清单中的 FireRedAudio 代码与模型 revision，并与 GitHub/Hugging Face 最新 revision 比较。发现变化只创建兼容性审计提醒，不自动升级；仍需通过宿主 Python/Transformers 矩阵、ComfyUI schema 和真模型 Smoke Test 后才能更新固定版本。

## 许可证与安全

上游代码和模型采用 Apache-2.0。FireRedAudio 包含零样本声音克隆；仅在获得授权、符合法律和平台规则的场景使用。禁止欺诈、冒充、侵权或其他违法活动。本项目不是 FireRed Team 官方产品。
