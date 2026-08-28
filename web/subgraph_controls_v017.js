const { app } = window.comfyAPI.app;

const PRIMITIVES = new Set(["PrimitiveInt", "PrimitiveFloat"]);

function clamp(value, constraint) {
    let next = Number(value);
    if (!Number.isFinite(next)) next = Number(constraint.min ?? 0);
    if (Number.isFinite(Number(constraint.min))) next = Math.max(Number(constraint.min), next);
    if (Number.isFinite(Number(constraint.max))) next = Math.min(Number(constraint.max), next);
    if (constraint.integer) next = Math.round(next);
    return next;
}

function applyConstraint(node) {
    if (!PRIMITIVES.has(node?.comfyClass) && !PRIMITIVES.has(node?.type)) return;
    const constraint = node.properties?.t8_firered_constraint;
    if (!constraint || node.__t8FireRedConstraintInstalled) return;
    const target = node.widgets?.find((widget) => widget.name === "value") ?? node.widgets?.[0];
    if (!target) return;

    target.options = {
        ...(target.options ?? {}),
        min: constraint.min,
        max: constraint.max,
        step: constraint.step,
        precision: constraint.integer ? 0 : target.options?.precision,
    };
    target.value = clamp(target.value, constraint);
    const originalCallback = target.callback;
    target.callback = function (value) {
        const bounded = clamp(value, constraint);
        target.value = bounded;
        return originalCallback?.call(this, bounded);
    };
    node.__t8FireRedConstraintInstalled = true;
    node.setDirtyCanvas?.(true, true);
}

function constrainWidget(target, constraint) {
    if (!target || !constraint || target.__t8FireRedConstraintInstalled) return;
    const boundedOptions = {
        min: constraint.min,
        max: constraint.max,
        step: constraint.step,
        precision: constraint.integer ? 0 : target.options?.precision,
    };
    if (target.options && typeof target.options === "object") {
        Object.assign(target.options, boundedOptions);
    } else {
        try { target.options = boundedOptions; } catch { /* getter-only proxy */ }
    }
    const valueDescriptor = Object.getOwnPropertyDescriptor(target, "value");
    if (valueDescriptor?.configurable && valueDescriptor.get && valueDescriptor.set) {
        const originalGet = valueDescriptor.get;
        const originalSet = valueDescriptor.set;
        Object.defineProperty(target, "value", {
            ...valueDescriptor,
            get() { return originalGet.call(this); },
            set(value) { originalSet.call(this, clamp(value, constraint)); },
        });
    }
    target.value = clamp(target.value, constraint);
    const originalCallback = target.callback;
    target.callback = function (value) {
        const bounded = clamp(value, constraint);
        target.value = bounded;
        return originalCallback?.call(this, bounded);
    };
    target.__t8FireRedConstraintInstalled = true;
}

function applyProxyConstraints(node) {
    const constraints = node?.properties?.t8_firered_proxy_constraints;
    if (!Array.isArray(constraints) || !Array.isArray(node.widgets)) return;
    // ComfyUI 1.49 migrates legacy proxyWidgets into promoted subgraph inputs.
    // Boundary widgets keep semantic names (for example target_line_id), while
    // Primitive proxies are projected as value/value_1/value_2... . Filter to
    // that stable primitive sequence so direct boundary widgets cannot shift
    // the numeric constraint mapping.
    const primitiveTargets = node.widgets.filter((target) => /^value(?:_\d+)?$/.test(target.name ?? ""));
    const numericTargets = primitiveTargets.filter((target) =>
        target.type === "number" || target.type === "slider"
    );
    const numericConstraints = constraints.filter(Boolean);
    const usedTargets = new Set();
    numericConstraints.forEach((constraint, index) => {
        const matchingInput = node.inputs?.find((input) =>
            input.label === constraint.label || input.name === constraint.label
        );
        const byInput = matchingInput?._widget
            ?? node._projectPromotedWidget?.(matchingInput);
        const byLabel = numericTargets.find((target) =>
            !usedTargets.has(target) && target.label === constraint.label
        );
        const target = byInput ?? byLabel ?? numericTargets[index];
        if (target) usedTargets.add(target);
        constrainWidget(target, constraint);
    });
    node.setDirtyCanvas?.(true, true);
}

function applyAllConstraints(node) {
    applyConstraint(node);
    applyProxyConstraints(node);
}

app.registerExtension({
    name: "t8star.fireredaudio.subgraph-controls-v017",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!PRIMITIVES.has(nodeData.name)) return;
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            this.__t8FireRedConstraintInstalled = false;
            applyConstraint(this);
            return result;
        };
    },
    setup() {
        for (const node of app.graph?._nodes ?? []) applyAllConstraints(node);
    },
    nodeCreated(node) {
        // Subgraph definitions restore properties after construction in some
        // frontend versions, so run once now and once after configuration.
        applyAllConstraints(node);
        queueMicrotask(() => applyAllConstraints(node));
        requestAnimationFrame(() => applyAllConstraints(node));
        // Promoted proxy widgets are materialized asynchronously by newer
        // ComfyUI frontends after subgraph migration/configuration completes.
        for (const delay of [50, 250, 1000]) {
            setTimeout(() => applyAllConstraints(node), delay);
        }
    },
});

export { applyAllConstraints, applyConstraint, applyProxyConstraints, clamp };
