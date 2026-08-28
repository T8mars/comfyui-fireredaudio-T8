# 采用创意候选

把评审板选中的候选回填到原 AudioBatch，并保留旧路径与完整来源。

## 使用方法

1. 连接原 AudioBatch、已评审候选和采用 ID。
2. 只替换候选对应的 `source_line_id`。
3. 回填后继续 QA、逐句审核或批量交付。

## 注意

不会覆盖旧 WAV；采用 Manifest 会记录前后路径、Seed 和人工评分。
