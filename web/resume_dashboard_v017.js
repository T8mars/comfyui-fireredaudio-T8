const { app } = window.comfyAPI.app;

const NODE_NAME = "T8_FireRedAudio_AudioBatchResume";

function render(node, payload) {
    const root = node.__t8ResumeDashboard;
    if (!root) return;
    node.__t8ResumePayload = payload;
    root.replaceChildren();
    const metrics = [
        ["可播放", payload?.playable ?? 0],
        ["已通过", payload?.approved_line_ids?.length ?? 0],
        ["待审核", payload?.pending_review_line_ids?.length ?? 0],
        ["待返修", payload?.retry_line_ids?.length ?? 0],
        ["缺失", payload?.missing?.length ?? 0],
    ];
    for (const [label, value] of metrics) {
        const card = document.createElement("div");
        card.style.cssText = "padding:8px;border:1px solid var(--border-color,#444);border-radius:6px;text-align:center";
        card.textContent = `${label}\n${value}`;
        card.style.whiteSpace = "pre-line";
        root.append(card);
    }
    const next = document.createElement("div");
    next.style.cssText = "grid-column:1/-1;padding:8px;border-radius:6px;background:var(--comfy-input-bg,#151515)";
    next.textContent = payload?.export_ready
        ? `✓ 可导出：${payload?.next_action ?? ""}`
        : `下一步：${payload?.next_action ?? "继续制作"}`;
    root.append(next);
    node.setSize?.([Math.max(560, node.size?.[0] ?? 560), Math.max(300, node.size?.[1] ?? 300)]);
    node.setDirtyCanvas?.(true, true);
}

function install(node) {
    if (node.__t8ResumeDashboard) return;
    const root = document.createElement("div");
    root.style.cssText = "display:grid;grid-template-columns:repeat(5,1fr);gap:7px;padding:9px;color:var(--input-text,#eee);font:12px/1.4 system-ui,sans-serif";
    node.__t8ResumeDashboard = root;
    node.addDOMWidget("resume_dashboard", "T8FireRedResumeDashboard", root, {
        serialize: false,
        hideOnZoom: true,
    });
    render(node, node.__t8ResumePayload);
}

app.registerExtension({
    name: "t8star.fireredaudio.resume-dashboard-v017",
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
            const payload = message?.fireredaudio_resume_dashboard?.[0];
            if (payload) render(this, payload);
            return result;
        };
    },
    nodeCreated(node) {
        if (node.comfyClass === NODE_NAME || node.type === NODE_NAME) install(node);
    },
});

export { NODE_NAME };
