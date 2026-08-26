# comfyui-fireredaudio-T8

FireRedAudio 的 ComfyUI V3 全能力节点，由 **T8star-Aix** 集成。节点菜单：

`T8star-Aix / Audio / FireRedAudio`

## 能力

- 多语言 ASR
- 长音频固定窗口分段转写，输出时间戳 JSON 与 SRT
- 单/双音频理解与问答，可选 thinking
- 中英文零样本声音克隆
- 自然语言声音设计
- 语义语音编辑
- 音高、速度、音量声学编辑
- 模型校验、隔离环境诊断、显存释放和 Worker 停止
- 任务进度同步、ComfyUI 中断联动取消、快速/均衡/高质量预设

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
4. TTS 输出连接 ComfyUI 原生 `Save Audio`。

示例见 `example_workflows/ui` 和 `example_workflows/api`。

## 许可证与安全

上游代码和模型采用 Apache-2.0。FireRedAudio 包含零样本声音克隆；仅在获得授权、符合法律和平台规则的场景使用。禁止欺诈、冒充、侵权或其他违法活动。本项目不是 FireRed Team 官方产品。
