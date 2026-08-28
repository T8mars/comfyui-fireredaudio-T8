from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FireRedAudio v0.17 人工听感验收</title><style>
:root{color-scheme:light;font-family:Inter,"Microsoft YaHei",sans-serif;background:#f5f1eb;color:#292725}body{max-width:980px;margin:auto;padding:30px}
h1{margin-bottom:6px}.muted{color:#726b64}.card{background:#fffdfa;border:1px solid #ded7cf;border-radius:16px;padding:20px;margin:18px 0;box-shadow:0 8px 26px #5d504415}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}.take{border:1px solid #ded7cf;border-radius:12px;padding:14px;background:#faf7f2}
audio{width:100%;margin:8px 0}label{display:block;margin:8px 0}select,textarea,button{font:inherit;padding:9px;border:1px solid #bdb4ab;border-radius:8px;background:white}textarea{width:100%;box-sizing:border-box}
button{background:#c9666d;color:white;border:0;font-weight:700;cursor:pointer}.gate{font-weight:700;color:#9b4249}
</style></head><body>
<h1>FireRedAudio v0.17 人工听感验收</h1><p class="muted">自动指标只负责预筛；请戴耳机完成自然度与匿名候选判断。</p>
<section class="card"><h2>1. 字幕时长适配</h2><p>原始生成</p><audio controls preload="metadata" src="duration-source.wav"></audio>
<p>v0.17 停顿保护/语音感知结果</p><audio controls preload="metadata" src="duration-v017.wav"></audio>
<label><input id="duration_artifacts" type="checkbox"> 无明显金属音、断裂或拼接伪影</label>
<label><input id="duration_pacing" type="checkbox"> 语速与停顿听起来自然</label>
<label><input id="duration_pronunciation" type="checkbox"> 发音内容保持完整</label>
<textarea id="duration_notes" rows="3" placeholder="时长适配备注"></textarea></section>
<section class="card"><h2>2. 匿名创意候选</h2><p class="muted">顺序已经打乱，文件名不包含 Seed 或来源。先试听，再选择。</p>
<div class="grid">__TAKES__</div>
<label>最满意候选 <select id="candidate_choice"><option value="">请选择</option>__OPTIONS__<option value="none">都不采用</option></select></label>
<label><input id="candidate_distinct" type="checkbox"> 至少能听出两种不同的表演/节奏</label>
<label><input id="candidate_quality" type="checkbox"> 最佳候选质量不低于原制作要求</label>
<label><input id="candidate_blind" type="checkbox"> 试听时没有查看 blind-map.json</label>
<textarea id="candidate_notes" rows="3" placeholder="候选听感备注"></textarea></section>
<section class="card"><p class="gate">三个时长勾选和三个候选勾选全部成立，才视为本轮人工门禁通过。</p><button id="save">下载 listening-result.json</button></section>
<script>
document.getElementById('save').addEventListener('click',()=>{const value={schema_version:1,release:'0.17.0',reviewed_at:new Date().toISOString(),duration_fit:{artifacts:duration_artifacts.checked,natural_pacing:duration_pacing.checked,pronunciation_preserved:duration_pronunciation.checked,notes:duration_notes.value},creative_candidates:{selected_blind_label:candidate_choice.value,audibly_distinct_take:candidate_distinct.checked,quality_not_worse:candidate_quality.checked,blind_review_respected:candidate_blind.checked,notes:candidate_notes.value}};value.passed=Object.values(value.duration_fit).slice(0,3).every(Boolean)&&candidate_distinct.checked&&candidate_quality.checked&&candidate_blind.checked&&!!candidate_choice.value;const blob=new Blob([JSON.stringify(value,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='listening-result-v017.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});
</script></body></html>"""


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the v0.17 human listening gate")
    parser.add_argument("--duration-source", required=True)
    parser.add_argument("--duration-output", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source = Path(args.duration_source).resolve()
    fitted = Path(args.duration_output).resolve()
    report_path = Path(args.candidate_report).resolve()
    output = Path(args.output_dir).resolve()
    if not source.is_file() or not fitted.is_file() or not report_path.is_file():
        raise FileNotFoundError("时长样本、适配结果或候选报告不存在")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    items = [item for item in report.get("items", []) if Path(str(item.get("output_path") or "")).is_file()]
    if not 2 <= len(items) <= 8:
        raise ValueError("候选报告必须包含 2–8 条可播放音频")
    items.sort(key=lambda item: hashlib.sha256((str(item.get("output_sha256")) + "v017-blind").encode()).hexdigest())
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output / "duration-source.wav")
    shutil.copy2(fitted, output / "duration-v017.wav")
    blind_map = []
    cards = []
    options = []
    for index, item in enumerate(items):
        label = chr(65 + index)
        filename = f"candidate-{label}.wav"
        path = Path(str(item["output_path"])).resolve()
        shutil.copy2(path, output / filename)
        cards.append(f'<div class="take"><strong>候选 {label}</strong><audio controls preload="metadata" src="{filename}"></audio></div>')
        options.append(f'<option value="{label}">候选 {label}</option>')
        blind_map.append({"blind_label": label, "line_id": item.get("line_id"), "seed": item.get("seed"), "origin": item.get("origin") or item.get("candidate_origin"), "sha256": file_digest(path)})
    (output / "listen.html").write_text(HTML.replace("__TAKES__", "".join(cards)).replace("__OPTIONS__", "".join(options)), encoding="utf-8")
    (output / "blind-map.json").write_text(json.dumps({"schema_version": 1, "items": blind_map}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"listen_html": str(output / "listen.html"), "candidate_count": len(items), "source_sha256": file_digest(source), "duration_output_sha256": file_digest(fitted)}
    (output / "review-package.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
