# comfyui-fireredaudio-T8

FireRedAudio 的 24 个 ComfyUI V3 全能力节点，由 **T8star-Aix** 集成。节点菜单：

`T8star-Aix / Audio / FireRedAudio`

## 能力

- 多语言 ASR
- 长音频静音感知分段、重叠去重，输出 SRT、VTT、JSON 与 JSONL
- 原生长录音时间线、时间找内容、内容找时间和结构化摘要
- 单音频理解，以及 1–8 个动态音频比较问答，可选 thinking
- 中英文零样本声音克隆
- 自然语言声音设计
- 语义语音编辑
- 严格模板和参数化两种音高、速度、音量声学编辑
- 参考音频时长、响度、削波、静音、直流偏移质检
- 另存 24 kHz 单声道的非破坏参考音频清理副本，可选静音裁剪、-23 LUFS 和语音高通，保留前后质检与处理记录
- 可复用音色档案与 1–8 角色音色库
- 载入桌面整合包导出的项目交换 JSON，复用音色库、脚本和 adopted take
- SRT、`角色：台词`、带时间码角色脚本和 JSON 的解析与生成前预检
- 逐条落盘、原子 manifest、内容指纹缓存和中断恢复的批量配音
- 同一参考音频的读取、重采样、RedAE 与 Patch Encoder 条件进程内复用；文件变化或模型卸载时安全失效
- 顺序、时间码定位和叠加三种时间线渲染，以及峰值/时间槽溢出报告
- 成品逐条 ASR 回读、中文 CER/英文 WER、削波、静音和时长 QA
- WAV、FLAC、MP3、OGG 音频与 SRT、VTT、TXT、JSONL 安全保存
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

普通单卡优先使用 `auto_safe`；`off` 用作 SDPA 对照与排障；DeepSpeed、FLA+Liger 和 Torch Compile 只有在相同真实工作流的暖机多轮中位数证明更快时才建议手动选择。当前不启用多卡。

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

示例见 `example_workflows/ui` 和 `example_workflows/api`，其中 `15_reference_cleanup` 演示参考音频质检后再生成不覆盖源文件的清理副本。该节点不会执行降噪、去混响或削波伪修复。

长音频字幕是静音感知的分段级近似时间，不是词级强制对齐。上游生成目前仅支持单样本，批量工作流应顺序执行，不能视为原生 GPU batch。

## 配音生产工作流

推荐连接顺序：

```text
Load Audio → 音色档案 ┐
Load Audio → 音色档案 ├→ 音色库 → 角色脚本/SRT 预检 → 可恢复批量配音
                       └───────────────────────────────┬→ 时间线渲染 → 保存音频
模型/隔离运行时 ───────────────────────────────────────└→ 成品语音 QA
```

若脚本已在桌面整合包中整理，可点击“导出 ComfyUI 项目 JSON”，再用“FireRedAudio 桌面项目交换”节点一次还原 Voice Bank、Script Plan 和已采用的 Audio Batch。节点会验证参考音频 SHA-256；文件移动或被改写时会明确报错。

角色脚本支持以下形式：

```text
旁白：欢迎收听本期节目。
[00:00:03,000 --> 00:00:06,000] 小夏：大家好，我是小夏。
```

SRT 的正文可用 `[角色] 台词` 标记角色。JSON 可直接传数组，或传包含 `lines` 数组的对象；每项支持 `speaker`、`text`、`language`、`start_seconds`、`end_seconds`。

批量配音输出到 `ComfyUI/output/<subfolder>/<project_name>/`。`manifest.json` 会在每条成功或失败后原子更新；启用恢复后，只有台词、音色参考、生成参数和模型身份指纹均一致且 WAV 仍存在的条目才会命中缓存。由于上游生成接口当前只支持 `batch_size=1`，此节点采用顺序 Worker 请求，以获得可中断、可诊断和可恢复的行为，并不宣称原生张量 batch 加速。

## 许可证与安全

上游代码和模型采用 Apache-2.0。FireRedAudio 包含零样本声音克隆；仅在获得授权、符合法律和平台规则的场景使用。禁止欺诈、冒充、侵权或其他违法活动。本项目不是 FireRed Team 官方产品。
