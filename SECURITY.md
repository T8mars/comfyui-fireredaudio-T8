# Security

请勿公开外部 Worker Token。托管 Worker 仅绑定 `127.0.0.1`，并使用随机 Bearer Token。

发现漏洞时，请通过发布仓库的私密安全报告渠道提交，不要在公开 Issue 中附带令牌、个人音频或模型缓存路径。

本节点不会把隔离运行时依赖安装进 ComfyUI Python；`requirements.txt` 有意不声明宿主依赖。
