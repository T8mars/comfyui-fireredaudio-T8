from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "web" / "docs"


NODES = {
    "T8_FireRedAudio_ModelLoader": {
        "zh": ("模型与隔离运行时", "创建所有 FireRedAudio 节点共用的 Worker 句柄，不会改动 ComfyUI 的 Python/Transformers。", ["模型放在 `ComfyUI/models/TTS/FireRedAudio`，或填写自定义目录。", "普通单卡先选 `auto_safe`；`off` 用于对照排障。", "DeepSpeed、FLA/Liger、Torch Compile 先用“加速实测向导”测完再手动选择。"], "Lite 只支持识别/理解；生成、声音设计和编辑必须使用 Full。"),
        "en": ("Model and isolated runtime", "Creates the worker handle shared by all FireRedAudio nodes without changing ComfyUI's Python or Transformers packages.", ["Place models under `ComfyUI/models/TTS/FireRedAudio`, or select a custom model directory.", "Use `auto_safe` first on a normal single GPU; use `off` as the troubleshooting baseline.", "Benchmark DeepSpeed, FLA/Liger, and Torch Compile before selecting them manually."], "Lite supports recognition and understanding only. Generation, voice design, and editing require Full."),
    },
    "T8_FireRedAudio_GenerationSettings": {
        "zh": ("生成参数", "集中复用质量预设、Seed、音频步数、文本 Token、扩散步数和 CFG。", ["先用 `balanced`。", "复现成品时固定 Seed 和全部参数。", "创意探索请使用候选池，不要把 QA 纠错当作随机抽卡。"], "高步数会明显增加耗时；参数变化会使批量缓存失效。"),
        "en": ("Generation settings", "Reusable quality preset, seed, audio steps, text tokens, diffusion steps, and CFG.", ["Start with `balanced`.", "Keep the seed and all settings fixed for reproducible delivery.", "Use the creative candidate pool for exploration instead of turning QA repair into random sampling."], "Higher steps cost more time. Any settings change invalidates matching batch cache entries."),
    },
    "T8_FireRedAudio_TTS": {
        "zh": ("零样本声音克隆", "用参考音频和逐字稿朗读目标文本。逐字稿留空时可自动 ASR。", ["连接 Full 模型和 `Load Audio`。", "有准确逐字稿时直接填写，可减少一次 ASR。", "输出 AUDIO 继续连接保存音频或后期节点。"], "参考音频尽量单人、清晰、无混响；自动 ASR 结果仍应人工核对。"),
        "en": ("Zero-shot voice cloning", "Reads target text using a reference recording and its transcript. The transcript can be generated automatically by ASR.", ["Connect a Full model and `Load Audio`.", "Provide an exact transcript when available to avoid an extra ASR pass.", "Connect AUDIO to a save or post-production node."], "Prefer a clean, single-speaker reference and manually verify automatic transcripts."),
    },
    "T8_FireRedAudio_SeedAudition": {
        "zh": ("多 Seed 试音", "为一段新文案生成多个匿名 Take，并可用 ASR 与音频代理指标给出建议。", ["候选文件名不包含 Seed，适合盲听。", "把 `all_takes` 接到多 Take 试听评审板。", "推荐结果是机器筛选，不替代人工听感。"], "用于新文案试音；已有批量台词的创意返修请使用“单句创意候选池”。"),
        "en": ("Multi-seed audition", "Generates anonymous takes for a new line and can produce a suggestion from ASR and audio proxy metrics.", ["Filenames hide the seed for blind listening.", "Connect `all_takes` to the Take Review Board.", "The automatic suggestion is a filter, not a substitute for listening."], "Use this for new copy. Use Creative Line Candidate Pool for a line already inside an AudioBatch."),
    },
    "T8_FireRedAudio_VoiceDesign": {
        "zh": ("声音设计", "用自然语言描述新音色并朗读指定文本。", ["描述年龄、音高、音色质感、音量、口音、情绪、流畅度、语速、清晰度、语调和性格。", "先用短句试音，再把满意结果保存为后续参考。", "相同描述也应通过多 Seed 盲听选 Take。"], "不要把角色设定或剧情要求混进音色描述；出现噪音时先核对 Full 模型与官方参数。"),
        "en": ("Voice design", "Creates a new voice from a natural-language description and reads the supplied text.", ["Describe age, pitch, timbre, volume, accent, emotion, fluency, speed, clarity, intonation, and personality.", "Audition with a short sentence, then save an accepted result as a later reference.", "Blind-review multiple seeds for the same description."], "Keep character plot instructions out of the acoustic description. If output is noise, verify the Full model and official parameter path."),
    },
    "T8_FireRedAudio_SpeechEdit": {
        "zh": ("语音编辑", "执行语义插入、删除、替换，或官方英文模板的声学编辑。", ["语义替换示例：`最后一句替换：你们在干嘛呀`。", "删除示例：`删除第二句话`；插入示例：`在第一句后插入：欢迎回来`。", "音高/速度/音量优先使用“参数化声学编辑”节点，避免手写格式错误。"], "编辑指令针对音频中的实际内容；先试听原音频并明确句子位置。"),
        "en": ("Speech edit", "Performs semantic insertion, deletion, replacement, or upstream acoustic edit templates.", ["Replace example: `Replace the last sentence with: What are you doing?`", "Delete example: `Delete the second sentence`; insert example: `Insert after the first sentence: Welcome back.`", "Use Parametric Acoustic Edit for pitch, speed, and volume to avoid malformed commands."], "The instruction targets content actually present in the audio. Listen first and identify the sentence unambiguously."),
    },
    "T8_FireRedAudio_ScriptParser": {
        "zh": ("角色脚本 / SRT 预检", "把 SRT、角色台词或 JSON 转为稳定 ScriptPlan，并在生成前检查角色和时间槽。", ["角色脚本写法：`旁白：欢迎收听。`", "时间码写法：`[00:00:03,000 --> 00:00:06,000] 小夏：大家好。`", "先查看预检报告，再连接朗读规范化和批量配音。"], "脚本改动会改变 line ID 或内容指纹；继续旧项目时必须确认匹配。"),
        "en": ("Role script / SRT preflight", "Converts SRT, role-script, or JSON input into a stable ScriptPlan and validates speakers and cue windows before generation.", ["Role-script form: `Narrator: Welcome.`", "Timed form: `[00:00:03,000 --> 00:00:06,000] Alex: Hello.`", "Inspect the preflight report before normalization and batch dubbing."], "Script edits can change line IDs or fingerprints. Verify identity before resuming an older project."),
    },
    "T8_FireRedAudio_BatchDubbing": {
        "zh": ("可恢复批量配音", "按脚本批量生成并在每条结束后原子写 Manifest，支持中断恢复和内容指纹缓存。", ["连接 Model、ScriptPlan、VoiceBank 和可选生成参数。", "24GB 显存通常从 batch size 4–8 开始。", "输出 AudioBatch 继续进入时长适配、QA、审核和交付。"], "只有模型、文本、音色、参数和文件都匹配时才命中缓存；不会用旧音频冒充新结果。"),
        "en": ("Resumable batch dubbing", "Generates a script in batches and atomically updates the manifest after every line, with interruption recovery and fingerprinted caching.", ["Connect Model, ScriptPlan, VoiceBank, and optional settings.", "Start around batch size 4–8 on a 24 GB GPU.", "Send AudioBatch to duration fitting, QA, review, and delivery."], "A cache hit requires matching model, text, voice, settings, and an existing file. Stale audio is not reused as a new result."),
    },
    "T8_FireRedAudio_SpeechQA": {
        "zh": ("成品语音 QA", "逐句 ASR 回读并检查 CER/WER、削波、静音和字幕超时。", ["`failed_line_ids` 可接纠错返修，或接逐句审核后人工决定。", "ASR 缓存只复用逐字稿；阈值每次重新计算。", "静音异常不等于表演错误，最终仍要人工试听。"], "QA 是证据门槛，不是音色、情绪和表演质量的最终裁判。"),
        "en": ("Rendered speech QA", "Runs per-line ASR and checks CER/WER, clipping, silence, and cue overrun.", ["Connect `failed_line_ids` to corrective retry or review the evidence line by line.", "ASR cache reuses only the transcript; thresholds are recalculated every run.", "A silence flag may be an intentional performance pause, so listen before rejecting."], "QA is an evidence gate, not the final judge of voice, emotion, or acting quality."),
    },
    "T8_FireRedAudio_DurationFit": {
        "zh": ("语音感知字幕时长适配", "让生成语音落入 SRT 时间槽，同时尽量保持自然语速。", ["默认 `speech_aware`：先检测并裁掉首尾多余静音。", "剩余超时时只加速语音片段；达到阈值的内部表演停顿保持原长。", "所需语音倍率超过独立自然语速上限时输出重做 line ID，不截断对白。"], "报告会列出受保护停顿、语音时长和实际倍率；源文件始终保留，最终自然度仍需试听。"),
        "en": ("Speech-aware subtitle duration fit", "Fits generated speech into SRT cue windows while preserving natural delivery where possible.", ["Default `speech_aware` detects and removes excess boundary silence first.", "For residual overrun, only speech spans are accelerated while qualifying performance pauses keep their original duration.", "If the required speech tempo exceeds the independent natural-speed limit, the line is sent for regeneration."], "The report lists protected pauses, speech duration, and actual tempo. Source files are preserved and final naturalness still requires listening."),
    },
    "T8_FireRedAudio_LineReview": {
        "zh": ("逐句制作审核台", "把自动 QA 和人工试听决定合并为通过、复核、重做三条路径。", ["逐句播放并填写决定、1–5 分和备注。", "`retry_line_ids` 接纠错返修或单句创意候选池。", "只把 `approved_batch` 送入最终交付。"], "`auto` 只是接受 QA 建议；交付前仍应对关键台词作人工确认。"),
        "en": ("Line production review", "Combines automatic QA evidence and human listening into approve, review, and retry paths.", ["Listen per line and record a decision, 1–5 rating, and note.", "Connect `retry_line_ids` to corrective retry or the creative candidate pool.", "Send only `approved_batch` to final delivery."], "`auto` follows the QA suggestion. Critical delivery lines still need human confirmation."),
    },
    "T8_FireRedAudio_BatchRetry": {
        "zh": ("QA 纠错返修", "只重生成失败 line ID，保持脚本、音色和质量目标不变，并做哈希与字幕时间槽门禁。", ["用于文字错误、生成失败、削波或超时等纠错。", "递增 Seed 会记录每次请求和结果。", "成功条目才会非破坏地合并回 AudioBatch。"], "这是纠错通道，不用于追求更有创意的表演；创意探索请用单句创意候选池。"),
        "en": ("Corrective QA retry", "Regenerates only failed line IDs while keeping script, voice, and quality goals fixed, with hash and cue-duration gates.", ["Use for text errors, generation failures, clipping, or cue overruns.", "Incremented seeds are recorded for every attempt.", "Only a passing replacement is merged non-destructively into the AudioBatch."], "This is a corrective path, not an acting-variation sampler. Use Creative Line Candidate Pool for exploration."),
    },
    "T8_FireRedAudio_CreativeCandidatePool": {
        "zh": ("单句创意候选池", "为一个现有 line ID 生成不同 Seed 的匿名表演候选，并可把原 Take 一并放入盲听。", ["显式填写一个 line ID，不要把 QA 纠错列表当作创意选择。", "默认保留原 Take，并用随机文件名避免从编号猜来源。", "查看声学差异预筛后仍须进入多 Take 评审板匿名盲听。"], "哈希、时长、包络和频谱只能发现明显重复，不能证明人耳可辨的表演差异；节点不会自动替换原音频。"),
        "en": ("Creative line candidate pool", "Generates anonymous performance variations for one existing line ID and can include the original take in the blind review.", ["Enter exactly one line ID instead of treating a QA failure list as creative direction.", "Keep the original take and use randomized filenames so provenance cannot be guessed from numbering.", "Inspect the acoustic pre-screen, then complete an anonymous Take Review Board listen."], "Hashes, duration, envelope, and spectrum only detect obvious duplicates; they do not prove a perceptible acting difference. Source audio is never replaced automatically."),
    },
    "T8_FireRedAudio_TakeReviewBoard": {
        "zh": ("多 Take 试听评审板", "用匿名乱序 A/B/C 播放器保存评分、备注和唯一采用项。", ["首次保持采用序号为 0，只运行到评审板并完成盲听。", "在节点内点击“采用”，它会写入隐藏的稳定候选 ID。", "再次运行后把已评审 AudioBatch 和采用 ID 连接候选采用节点。"], "节点默认不会采用第一条；只有人工点击采用并再次运行后，采用 ID 才会输出。"),
        "en": ("Take review board", "Uses shuffled anonymous A/B/C players to record ratings, notes, and one adopted take.", ["Keep selection position at 0 on the first run and listen at the board.", "Click Adopt in the node; it writes the hidden stable candidate ID.", "Run again, then connect the reviewed AudioBatch and adopted ID to Apply Creative Candidate."], "The first item is never adopted by default. An ID is emitted only after an explicit human choice and rerun."),
    },
    "T8_FireRedAudio_AudioBatchResume": {
        "zh": ("恢复批次与制作状态", "从既有 Manifest 恢复 AudioBatch，并显示继续创作所需的状态摘要。", ["选择批量、时长适配、评审、返修或候选 Manifest。", "查看可播放、已通过、待审核、待返修和缺失数量。", "按面板给出的下一步连接审核、返修或批量保存。"], "只有所有条目完整且已明确通过/采用时才显示导出就绪；缺失文件不会被静默忽略。"),
        "en": ("Resume batch and production state", "Restores an AudioBatch from a manifest and shows the production state needed to continue.", ["Choose a batch, duration-fit, review, repair, or candidate manifest.", "Inspect playable, approved, pending, retry, and missing counts.", "Follow the dashboard action into review, repair, or batch save."], "Export readiness requires every item to exist and be explicitly approved or adopted. Missing files are never hidden."),
    },
    "T8_FireRedAudio_CandidateApply": {
        "zh": ("采用创意候选", "把评审板选中的候选回填到原 AudioBatch，并保留旧路径与完整来源。", ["连接原 AudioBatch、已评审候选和采用 ID。", "只替换候选对应的 `source_line_id`。", "回填后继续 QA、逐句审核或批量交付。"], "不会覆盖旧 WAV；采用 Manifest 会记录前后路径、Seed 和人工评分。"),
        "en": ("Apply creative candidate", "Merges the reviewed selection back into the source AudioBatch while preserving the old path and full provenance.", ["Connect the source AudioBatch, reviewed candidates, and selected ID.", "Only the candidate's `source_line_id` is replaced.", "Continue with QA, line review, or batch delivery."], "Old WAV files are never overwritten. The adoption manifest records previous/new paths, seed, and human review."),
    },
    "T8_FireRedAudio_SaveAudioBatch": {
        "zh": ("批量保存 / 下载", "把成功 Take 导出到 ComfyUI output，生成便携 Manifest、原生试听下载和可选 ZIP。", ["选择 WAV/FLAC 用于无损制作；MP3/OGG 用于轻量预览。", "结果区每条音频都有播放器和下载入口。", "ZIP 适合交付或迁移，不包含模型。"], "导出是新副本；源 AudioBatch 和源音频不会被修改。"),
        "en": ("Batch save / download", "Exports successful takes under ComfyUI output with a portable manifest, native players/downloads, and an optional ZIP.", ["Use WAV/FLAC for lossless production and MP3/OGG for lightweight previews.", "Every previewed result has a native player and download entry.", "ZIP is for delivery or transfer and never contains model files."], "Export creates new copies. The source AudioBatch and source audio remain unchanged."),
    },
    "T8_FireRedAudio_AccelerationBenchmark": {
        "zh": ("加速实测向导", "在同一参考、文本、Seed 和参数下实测不同后端，给出可审计建议。", ["至少 1 次暖机、3 次正式测量。", "先比较 `off,flash_attention,deepspeed`。", "查看中位耗时、RTF、显存、实际回退和哈希复现。"], "向导只给建议，不会偷偷修改模型加载器；实验模式至少快 20% 才值得采用。"),
        "en": ("Acceleration benchmark", "Benchmarks backends with the same reference, text, seed, and settings and produces an auditable recommendation.", ["Use at least one warm-up and three measured runs.", "Compare `off,flash_attention,deepspeed` first.", "Inspect median latency, RTF, VRAM, actual fallback, and reproducible hashes."], "The benchmark never changes the loader. Experimental modes should improve by at least 20% before adoption."),
    },
    "T8_FireRedAudio_Environment": {
        "zh": ("环境诊断", "显示宿主 Python、Torch、Transformers 和隔离 Worker 状态，用于证明依赖没有串环境。", ["安装或升级后先运行一次。", "宿主 Transformers 可以是 4.x；FireRedAudio Worker 使用固定 5.8.0。", "把报告附在 Issue 中比只发截图更容易定位。"], "该节点只诊断，不下载模型、不安装依赖、不启动生成。"),
        "en": ("Environment diagnostics", "Shows host Python, Torch, Transformers, and isolated worker state to prove dependencies are not mixed.", ["Run once after install or upgrade.", "Host Transformers may remain on 4.x while the FireRedAudio worker uses pinned 5.8.0.", "Attach the report to issues; it is more actionable than a screenshot alone."], "This node is read-only: it does not download models, install packages, or run generation."),
    },
}


def render(title: str, purpose: str, steps: list[str], note: str, *, language: str) -> str:
    usage = "使用方法" if language == "zh" else "How to use"
    caution = "注意" if language == "zh" else "Important"
    lines = [f"# {title}", "", purpose, "", f"## {usage}", ""]
    lines.extend(f"{index}. {step}" for index, step in enumerate(steps, 1))
    lines.extend(["", f"## {caution}", "", note, ""])
    return "\n".join(lines)


def build() -> dict[str, int]:
    for node_id, localized in NODES.items():
        fallback = render(*localized["zh"], language="zh")
        target = DOCS / f"{node_id}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fallback, encoding="utf-8")
        locale_dir = DOCS / node_id
        locale_dir.mkdir(parents=True, exist_ok=True)
        locale_dir.joinpath("zh.md").write_text(fallback, encoding="utf-8")
        locale_dir.joinpath("en.md").write_text(
            render(*localized["en"], language="en"),
            encoding="utf-8",
        )
    return {"nodes": len(NODES), "files": len(NODES) * 3}


if __name__ == "__main__":
    print(build())
