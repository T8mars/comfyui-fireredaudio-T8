# 环境诊断

显示宿主 Python、Torch、Transformers 和隔离 Worker 状态，用于证明依赖没有串环境。

## 使用方法

1. 安装或升级后先运行一次。
2. 宿主 Transformers 可以是 4.x；FireRedAudio Worker 使用固定 5.8.0。
3. 把报告附在 Issue 中比只发截图更容易定位。

## 注意

该节点只诊断，不下载模型、不安装依赖、不启动生成。
