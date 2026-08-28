const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const NODE_NAME = "T8_FireRedAudio_TakeReviewBoard";

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

function writeWidget(node, name, value) {
    const target = widget(node, name);
    if (!target) return;
    target.value = value;
    target.callback?.(value);
    node.setDirtyCanvas?.(true, true);
}

function audioUrl(descriptor) {
    const params = new URLSearchParams({
        filename: descriptor?.filename ?? "",
        subfolder: descriptor?.subfolder ?? "",
        type: descriptor?.type ?? "output",
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function installStyle() {
    if (document.getElementById("t8-firered-take-review-style")) return;
    const style = document.createElement("style");
    style.id = "t8-firered-take-review-style";
    style.textContent = `
        .t8-fr-takes { display:grid; gap:10px; padding:10px; color:var(--input-text,#eee);
            background:var(--comfy-menu-bg,#202020); border:1px solid var(--border-color,#444);
            border-radius:8px; font:12px/1.4 system-ui,sans-serif; }
        .t8-fr-takes__hint { padding:8px 10px; border-radius:6px; background:var(--comfy-input-bg,#151515); }
        .t8-fr-takes__row { display:grid; grid-template-columns:42px minmax(180px,1fr) 90px minmax(120px,.6fr) 80px;
            gap:8px; align-items:center; padding:8px; border:1px solid var(--border-color,#444); border-radius:7px; }
        .t8-fr-takes__label { font-size:20px; font-weight:800; text-align:center; color:var(--p-primary-color,#fb7299); }
        .t8-fr-takes audio { width:100%; height:32px; }
        .t8-fr-takes select,.t8-fr-takes input,.t8-fr-takes button { min-width:0; padding:6px; border-radius:5px;
            border:1px solid var(--border-color,#555); background:var(--comfy-input-bg,#111); color:var(--input-text,#eee); }
        .t8-fr-takes button { cursor:pointer; font-weight:700; }
        .t8-fr-takes button[data-selected="true"] { border-color:#67c587; color:#67c587; }
    `;
    document.head.append(style);
}

function render(node, payload) {
    const root = node.__t8TakeReviewRoot;
    if (!root) return;
    node.__t8TakeReviewPayload = payload;
    root.replaceChildren();
    const rows = Array.isArray(payload?.rows) ? payload.rows : [];
    const hint = document.createElement("div");
    hint.className = "t8-fr-takes__hint";
    hint.textContent = !payload || rows.length === 0
        ? "尚未生成候选。运行工作流后，这里会显示匿名试听、评分、备注和采用操作。"
        : payload.selection_required
            ? "匿名盲听：先评分和备注，再点“采用”；随后重新运行工作流完成回填。"
            : "已记录采用项。重新运行后会写入评审 Manifest 并继续回填。";
    root.append(hint);
    const ratings = readMapping(node, "ratings_json");
    const notes = readMapping(node, "notes_json");
    const selected = String(widget(node, "selected_line_id")?.value ?? "");

    for (const row of rows) {
        const lineId = String(row.line_id ?? "");
        const container = document.createElement("div");
        container.className = "t8-fr-takes__row";
        const label = document.createElement("div");
        label.className = "t8-fr-takes__label";
        label.textContent = String(row.blind_label ?? "?");
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.preload = "none";
        audio.src = audioUrl(row.audio);
        const rating = document.createElement("select");
        for (const [value, text] of [["", "未评分"], ["1", "1 分"], ["2", "2 分"], ["3", "3 分"], ["4", "4 分"], ["5", "5 分"]]) {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = text;
            rating.append(option);
        }
        rating.value = String(ratings[lineId] ?? row.rating ?? "");
        const note = document.createElement("input");
        note.placeholder = "听感备注";
        note.value = String(notes[lineId] ?? row.note ?? "");
        const choose = document.createElement("button");
        choose.type = "button";
        choose.textContent = "采用";
        choose.dataset.selected = String(selected === lineId);

        const sync = () => {
            if (rating.value) ratings[lineId] = Number(rating.value); else delete ratings[lineId];
            if (note.value.trim()) notes[lineId] = note.value.trim(); else delete notes[lineId];
            writeWidget(node, "ratings_json", JSON.stringify(ratings, null, 2));
            writeWidget(node, "notes_json", JSON.stringify(notes, null, 2));
        };
        rating.addEventListener("change", sync);
        note.addEventListener("input", sync);
        choose.addEventListener("click", () => {
            sync();
            writeWidget(node, "selected_position", 0);
            writeWidget(node, "selected_line_id", lineId);
            root.querySelectorAll("button").forEach((button) => { button.dataset.selected = "false"; });
            choose.dataset.selected = "true";
            hint.textContent = `已选择匿名候选 ${row.blind_label}。重新运行工作流完成采用和导出。`;
        });
        container.append(label, audio, rating, note, choose);
        root.append(container);
    }
    requestAnimationFrame(() => {
        const size = node.computeSize?.() ?? node.size;
        node.setSize?.([Math.max(760, size?.[0] ?? 760), Math.max(360, size?.[1] ?? 360)]);
        node.setDirtyCanvas?.(true, true);
    });
}

function install(node) {
    if (node.__t8TakeReviewRoot) return;
    installStyle();
    const root = document.createElement("div");
    root.className = "t8-fr-takes";
    node.__t8TakeReviewRoot = root;
    node.addDOMWidget("take_review_board", "T8FireRedTakeReview", root, {
        serialize: false,
        hideOnZoom: true,
    });
    render(node, node.__t8TakeReviewPayload);
}

app.registerExtension({
    name: "t8star.fireredaudio.take-review-v017",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            install(this);
            return result;
        };
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            const payload = message?.fireredaudio_take_review?.[0];
            if (payload) render(this, payload);
            return result;
        };
    },
    nodeCreated(node) {
        if (node.comfyClass === NODE_NAME || node.type === NODE_NAME) install(node);
    },
});

export { NODE_NAME };
