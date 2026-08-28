# 模型与隔离运行时

创建所有 FireRedAudio 节点共用的 Worker 句柄，不会改动 ComfyUI 的 Python/Transformers。

## 使用方法

1. 模型放在 `ComfyUI/models/TTS/FireRedAudio`，或填写自定义目录。
2. 普通单卡先选 `auto_safe`；`off` 用于对照排障。
3. DeepSpeed、FLA/Liger、Torch Compile 先用“加速实测向导”测完再手动选择。

## 注意

Lite 只支持识别/理解；生成、声音设计和编辑必须使用 Full。
