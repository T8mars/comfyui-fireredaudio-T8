# comfyui-fireredaudio-T8

FireRedAudio 的 ComfyUI V3 全能力节点，由 **T8star-Aix** 集成。节点菜单：

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

FireRedAudio 官方代码固定要求 Python 3.10、PyTorch 2.11.0 和 Transformers 5.8.0。许多 ComfyUI 节点仍依赖 Transformers 4.x，因此本节点**不会**把 FireRedAudio 依赖安装进 ComfyUI：

```text
ComfyUI Python / Transformers 4.x
        │ 标准 AUDIO + 本地 RPC
        ▼
隔离 Python 3.10 / torch 2.11 / Transformers 5.8 Worker
```

`requirements.txt` 有意保持为空。Windows 节点发行包随附固定版本 `uv`，请显式准备隔离环境：

```powershell
python scripts/setup_runtime.py
```

也可在模型加载器中填写其他隔离 `python.exe`，或连接桌面整合包启动的外部 Worker。节点绝不调用 `pip` 修改 ComfyUI 环境。

模型加载器的设备默认使用 `auto`。实际 GPU、空闲显存和最终选用的显存模式可通过“运行时控制”节点查看；`auto` 会在空闲显存不足 36 GiB 时选择顺序卸载，而不是只看显卡标称总显存。

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

示例见 `example_workflows/ui` 和 `example_workflows/api`。

长音频字幕是静音感知的分段级近似时间，不是词级强制对齐。上游生成目前仅支持单样本，批量工作流应顺序执行，不能视为原生 GPU batch。

## 许可证与安全

上游代码和模型采用 Apache-2.0。FireRedAudio 包含零样本声音克隆；仅在获得授权、符合法律和平台规则的场景使用。禁止欺诈、冒充、侵权或其他违法活动。本项目不是 FireRed Team 官方产品。
