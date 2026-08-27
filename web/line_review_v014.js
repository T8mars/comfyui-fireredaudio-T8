const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const NODE_NAME = "T8_FireRedAudio_LineReview";
const STATE_WIDGETS = ["decisions_json", "ratings_json", "notes_json"];

function installStyle() {
    if (document.getElementById("t8-firered-line-review-style")) return;
    const style = document.createElement("style");
    style.id = "t8-firered-line-review-style";
    style.textContent = `
        .t8-fr-review { box-sizing: border-box; width: 100%; min-height: 180px; max-height: 620px;
            overflow: auto; padding: 10px; border: 1px solid var(--border-color, #4b4b4b);
            border-radius: 8px; background: var(--comfy-menu-bg, #202020); color: var(--input-text, #eee);
            font: 12px/1.4 system-ui, sans-serif; }
        .t8-fr-review * { box-sizing: border-box; }
        .t8-fr-review__toolbar { position: sticky; top: 0; z-index: 2; display: flex; gap: 8px;
            align-items: center; flex-wrap: wrap; padding: 7px; margin: -4px -4px 8px;
            background: var(--comfy-menu-bg, #202020); border-bottom: 1px solid var(--border-color, #4b4b4b); }
        .t8-fr-review__summary { flex: 1; min-width: 220px; font-weight: 600; }
        .t8-fr-review button, .t8-fr-review select, .t8-fr-review input {
            border: 1px solid var(--border-color, #555); border-radius: 5px;
            background: var(--comfy-input-bg, #111); color: var(--input-text, #eee); padding: 5px 7px; }
        .t8-fr-review button { cursor: pointer; }
        .t8-fr-review button:hover { border-color: var(--p-primary-color, #fb7299); }
        .t8-fr-review__row { display: grid; grid-template-columns: 58px minmax(180px, 1.4fr) minmax(190px, 1fr) 100px 118px 72px minmax(130px, .8fr);
            gap: 7px; align-items: center; padding: 8px 4px; border-bottom: 1px solid var(--border-color, #3b3b3b); }
        .t8-fr-review__meta { color: var(--descrip-text, #aaa); }
        .t8-fr-review__text { min-width: 0; overflow-wrap: anywhere; }
        .t8-fr-review__source { margin-top: 3px; color: var(--descrip-text, #aaa); font-size: 11px; }
        .t8-fr-review__qa { font-size: 11px; }
        .t8-fr-review__qa--approve { color: #67c587; }
        .t8-fr-review__qa--review { color: #e8b65b; }
        .t8-fr-review__qa--retry { color: #ef6b73; }
        .t8-fr-review audio { width: 100%; min-width: 170px; height: 32px; }
        .t8-fr-review__audio { display: grid; grid-template-columns: 1fr auto; gap: 5px; align-items: center; }
        .t8-fr-review__download { color: var(--p-primary-color, #fb7299); text-decoration: none; font-weight: 700; }
        .t8-fr-review__empty { padding: 24px 10px; text-align: center; color: var(--descrip-text, #aaa); }
        @media (max-width: 900px) { .t8-fr-review__row { grid-template-columns: 48px 1fr; } }
    `;
    document.head.append(style);
}

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function readMapping(node, name) {
    try {
        const value = JSON.parse(widget(node, name)?.value || "{}");
        return value && typeof value === "object" && !Array.isArray(value) ? value : {};
    } catch {
        return {};
    }
}

function writeMapping(node, name, value) {
    const target = widget(node, name);
    if (!target) return;
    target.value = JSON.stringify(value, null, 2);
    target.callback?.(target.value);
    node.setDirtyCanvas?.(true, true);
}

function audioUrl(descriptor) {
    if (!descriptor) return "";
    const params = new URLSearchParams({
        filename: descriptor.filename ?? "",
        subfolder: descriptor.subfolder ?? "",
        type: descriptor.type ?? "output",
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function option(value, label) {
    const element = document.createElement("option");
    element.value = value;
    element.textContent = label;
    return element;
}

function qaLabel(review) {
    const decision = review?.suggested_decision ?? "review";
    const labels = { approve: "QA 通过", review: "人工复核", retry: "建议重做" };
    return { decision, label: labels[decision] ?? "人工复核" };
}

function renderBoard(node, payload) {
    const root = node.__t8LineReviewRoot;
    if (!root) return;
    node.__t8LineReviewPayload = payload;
    root.replaceChildren();
    const rows = Array.isArray(payload?.rows) ? payload.rows : [];
    if (!rows.length) {
        const empty = document.createElement("div");
        empty.className = "t8-fr-review__empty";
        empty.textContent = "运行节点后，这里会显示逐句播放器、QA 建议和审核操作。";
        root.append(empty);
        return;
    }

    const decisions = readMapping(node, "decisions_json");
    const ratings = readMapping(node, "ratings_json");
    const notes = readMapping(node, "notes_json");
    const toolbar = document.createElement("div");
    toolbar.className = "t8-fr-review__toolbar";
    const summary = document.createElement("div");
    summary.className = "t8-fr-review__summary";
    summary.textContent = `逐句审核：${payload.previewed ?? rows.length}/${payload.total ?? rows.length} 条`;
    const filter = document.createElement("select");
    filter.append(option("all", "全部"), option("approve", "QA 通过"), option("review", "待复核"), option("retry", "建议重做"));
    const apply = document.createElement("button");
    apply.type = "button";
    apply.textContent = "同步审核数据";
    const hint = document.createElement("span");
    hint.className = "t8-fr-review__meta";
    hint.textContent = "修改会立即写回节点；重新运行后生成审核 Manifest。";
    toolbar.append(summary, filter, apply, hint);
    root.append(toolbar);

    const rendered = [];
    for (const row of rows) {
        const lineId = String(row.line_id ?? "");
        const currentReview = row.review ?? {};
        const qa = qaLabel(currentReview);
        const container = document.createElement("div");
        container.className = "t8-fr-review__row";
        container.dataset.suggestion = qa.decision;

        const meta = document.createElement("div");
        meta.className = "t8-fr-review__meta";
        meta.textContent = `#${row.index ?? row.position}\n${row.speaker ?? ""}`;
        meta.style.whiteSpace = "pre-line";

        const text = document.createElement("div");
        text.className = "t8-fr-review__text";
        text.textContent = row.text ?? "";
        if (row.source_text && row.source_text !== row.text) {
            const source = document.createElement("div");
            source.className = "t8-fr-review__source";
            source.textContent = `原文：${row.source_text}`;
            text.append(source);
        }

        const audioCell = document.createElement("div");
        audioCell.className = "t8-fr-review__audio";
        const url = audioUrl(row.audio);
        if (url) {
            const audio = document.createElement("audio");
            audio.controls = true;
            audio.preload = "none";
            audio.src = url;
            const download = document.createElement("a");
            download.className = "t8-fr-review__download";
            download.href = url;
            download.download = row.audio?.filename ?? `${lineId}.wav`;
            download.title = "下载当前音频";
            download.textContent = "↓";
            audioCell.append(audio, download);
        } else {
            audioCell.textContent = "音频不可预览";
            audioCell.classList.add("t8-fr-review__meta");
        }

        const qaCell = document.createElement("div");
        qaCell.className = `t8-fr-review__qa t8-fr-review__qa--${qa.decision}`;
        qaCell.textContent = qa.label;
        qaCell.title = currentReview.suggestion_reason ?? "";

        const decision = document.createElement("select");
        decision.setAttribute("aria-label", `${lineId} 审核决定`);
        decision.append(
            option("auto", `自动（${qa.label}）`),
            option("approve", "通过"),
            option("review", "人工复核"),
            option("retry", "重做"),
        );
        decision.value = decisions[lineId] ?? currentReview.requested_decision ?? "auto";

        const rating = document.createElement("select");
        rating.setAttribute("aria-label", `${lineId} 评分`);
        rating.append(option("", "未评分"));
        for (let score = 1; score <= 5; score += 1) rating.append(option(String(score), `${score} 分`));
        rating.value = String(ratings[lineId] ?? currentReview.rating ?? "");

        const note = document.createElement("input");
        note.type = "text";
        note.placeholder = "审核备注";
        note.setAttribute("aria-label", `${lineId} 审核备注`);
        note.value = String(notes[lineId] ?? currentReview.note ?? "");

        const sync = () => {
            decisions[lineId] = decision.value;
            if (rating.value) ratings[lineId] = Number(rating.value); else delete ratings[lineId];
            if (note.value.trim()) notes[lineId] = note.value.trim(); else delete notes[lineId];
            writeMapping(node, "decisions_json", decisions);
            writeMapping(node, "ratings_json", ratings);
            writeMapping(node, "notes_json", notes);
        };
        decision.addEventListener("change", sync);
        rating.addEventListener("change", sync);
        note.addEventListener("input", sync);
        note.addEventListener("change", sync);
        container.append(meta, text, audioCell, qaCell, decision, rating, note);
        root.append(container);
        rendered.push({ container, sync });
    }
    filter.addEventListener("change", () => {
        for (const entry of rendered) {
            entry.container.hidden = filter.value !== "all" && entry.container.dataset.suggestion !== filter.value;
        }
    });
    apply.addEventListener("click", () => {
        for (const entry of rendered) entry.sync();
        hint.textContent = "已同步。重新运行节点即可保存本次审核。";
    });
    requestAnimationFrame(() => {
        const size = node.computeSize?.() ?? node.size;
        node.setSize?.([Math.max(1080, size?.[0] ?? 1080), Math.max(520, size?.[1] ?? 520)]);
        node.setDirtyCanvas?.(true, true);
    });
}

function installBoard(node) {
    if (node.__t8LineReviewRoot) return;
    installStyle();
    const root = document.createElement("div");
    root.className = "t8-fr-review";
    node.__t8LineReviewRoot = root;
    node.addDOMWidget("line_review_board", "T8FireRedLineReview", root, {
        serialize: false,
        hideOnZoom: true,
    });
    renderBoard(node, node.__t8LineReviewPayload);
}

app.registerExtension({
    name: "t8star.fireredaudio.line-review",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            installBoard(this);
            return result;
        };
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            const payload = message?.fireredaudio_review?.[0];
            if (payload) renderBoard(this, payload);
            return result;
        };
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.__t8LineReviewRoot?.querySelectorAll("audio").forEach((audio) => {
                audio.pause();
                audio.removeAttribute("src");
            });
            return onRemoved?.apply(this, arguments);
        };
    },
    nodeCreated(node) {
        if (node.comfyClass === NODE_NAME || node.type === NODE_NAME) installBoard(node);
    },
});

export { NODE_NAME, STATE_WIDGETS };
