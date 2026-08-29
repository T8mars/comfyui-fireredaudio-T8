from __future__ import annotations

import argparse
import copy
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "example_workflows" / "ui"
TEMPLATE_ROOT = ROOT / "example_workflows"
SUBGRAPH_ROOT = ROOT / "subgraphs"
NAMESPACE = uuid.UUID("0b1c8dbd-24ef-4d53-b0df-40a763e965e0")


@dataclass(frozen=True)
class TemplateSpec:
    output_name: str
    source_name: str
    title: str
    subtitle: str
    app_inputs: tuple[tuple[str, str], ...]
    app_outputs: tuple[str, ...]
    app_mode: bool = True


@dataclass(frozen=True)
class WidgetPort:
    node_id: int
    name: str
    kind: str
    label: str
    default: Any = None
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None


@dataclass(frozen=True)
class OutputPort:
    node_id: int
    slot: int
    name: str
    kind: str
    label: str


@dataclass(frozen=True)
class SubgraphSpec:
    output_name: str
    source_name: str
    title: str
    node_ids: tuple[int, ...]
    widgets: tuple[WidgetPort, ...]
    outputs: tuple[OutputPort, ...]


TEMPLATES = (
    TemplateSpec(
        "FireRedAudio_01_Quick_TTS.app.json",
        "01_zero_shot_tts.json",
        "快速声音克隆",
        "参考音频 → 自动逐字稿 → TTS → 保存",
        (
            ("2", "audio"),
            ("4", "prompt_text"),
            ("4", "target_text"),
            ("4", "language"),
            ("3", "quality_preset"),
            ("3", "seed"),
        ),
        ("4", "5"),
    ),
    TemplateSpec(
        "FireRedAudio_02_Voice_Design.app.json",
        "04_voice_design.json",
        "声音设计",
        "自然语言描述 → 新声音 → 直接保存",
        (
            ("3", "instruction"),
            ("3", "target_text"),
            ("2", "quality_preset"),
            ("2", "seed"),
        ),
        ("3", "4"),
    ),
    TemplateSpec(
        "FireRedAudio_03_Long_Reference_Screening.app.json",
        "23_long_reference_screening.json",
        "长录音筛选参考音色",
        "长录音 → 候选切片 → 盲听 → ASR → 音色档案",
        (
            ("2", "audio"),
            ("3", "minimum_seconds"),
            ("3", "preferred_seconds"),
            ("3", "maximum_seconds"),
            ("3", "candidate_limit"),
            ("4", "selected_position"),
        ),
        ("3", "4", "5", "6"),
    ),
    TemplateSpec(
        "FireRedAudio_04_Podcast_Local_Repair.app.json",
        "21_podcast_local_repair.json",
        "播客局部修复",
        "选择范围 → 语义编辑 → 无损回填 → A/B 保存",
        (
            ("2", "audio"),
            ("4", "range_mode"),
            ("4", "start_seconds"),
            ("4", "end_seconds"),
            ("5", "instruction"),
            ("5", "edit_type"),
            ("3", "seed"),
        ),
        ("6", "7", "8"),
    ),
    TemplateSpec(
        "FireRedAudio_05_Multirole_QA_Delivery.json",
        "20_creator_qa_repair_delivery.json",
        "多角色 QA 返修交付",
        "角色脚本 → 批量生成 → QA → 定向返修 → ZIP",
        (),
        (),
        app_mode=False,
    ),
    TemplateSpec(
        "FireRedAudio_06_SRT_Production_Review.json",
        "26_production_review_loop.json",
        "SRT 制作审核闭环",
        "规范化 → 时长适配 → QA → 人工审核 → 返修 → 交付",
        (),
        (),
        app_mode=False,
    ),
    TemplateSpec(
        "FireRedAudio_07_Creative_Line_Candidates.json",
        "28_creative_line_candidates.json",
        "单句创意候选与盲听采用",
        "人工指定台词 → 多 Seed 匿名候选 → 盲听评分 → 明确回填",
        (),
        (),
        app_mode=False,
    ),
)


SUBGRAPHS = (
    SubgraphSpec(
        "FireRedAudio Quick TTS.json",
        "01_zero_shot_tts.json",
        "FireRedAudio · 快速声音克隆",
        (4, 5),
        (
            WidgetPort(4, "prompt_text", "STRING", "参考音频逐字稿（可留空自动 ASR）", "同时，他强调微调要科学有序。"),
            WidgetPort(4, "target_text", "STRING", "目标文本", "欢迎使用 FireRedAudio，来自 T8star-Aix。"),
            WidgetPort(4, "language", "COMBO", "语言", "zh"),
            WidgetPort(4, "auto_transcribe_reference", "BOOLEAN", "逐字稿留空时自动 ASR", True),
        ),
        (
            OutputPort(5, 0, "audio", "AUDIO", "生成音频"),
            OutputPort(4, 1, "report", "STRING", "生成报告"),
            OutputPort(4, 2, "reference_transcript", "STRING", "实际参考逐字稿"),
        ),
    ),
    SubgraphSpec(
        "FireRedAudio Batch QA Repair Delivery.json",
        "20_creator_qa_repair_delivery.json",
        "FireRedAudio · 批量 QA 返修交付",
        (9, 10, 11, 13),
        (
            WidgetPort(9, "project_name", "STRING", "批量项目名", "creator-loop"),
            WidgetPort(9, "batch_size", "INT", "每批条数", 8, 1, 32, 1),
            WidgetPort(10, "max_text_error_rate", "FLOAT", "最大 CER/WER", 0.2, 0.0, 1.0, 0.01),
            WidgetPort(11, "max_attempts", "INT", "最多返修次数", 2, 1, 5, 1),
            WidgetPort(13, "audio_format", "COMBO", "交付格式", "wav"),
            WidgetPort(13, "create_zip", "BOOLEAN", "生成 ZIP", True),
        ),
        (
            OutputPort(13, 0, "audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", "交付批次"),
            OutputPort(13, 1, "manifest_path", "STRING", "交付 Manifest"),
            OutputPort(13, 2, "zip_path", "STRING", "交付 ZIP"),
            OutputPort(10, 1, "qa_report", "STRING", "首轮 QA 报告"),
            OutputPort(11, 2, "repair_report", "STRING", "返修报告"),
        ),
    ),
    SubgraphSpec(
        "FireRedAudio SRT Draft QA.json",
        "26_production_review_loop.json",
        "FireRedAudio · SRT 初稿与 QA",
        (6, 8, 9, 10),
        (
            WidgetPort(6, "replacement_dictionary_json", "STRING", "朗读替换词典 JSON", '{"API":"A P I"}'),
            WidgetPort(8, "project_name", "STRING", "项目名", "v014-review-demo"),
            WidgetPort(8, "batch_size", "INT", "每批条数", 8, 1, 32, 1),
            WidgetPort(9, "strategy", "COMBO", "时长适配策略", "speech_aware"),
            WidgetPort(9, "maximum_speed", "FLOAT", "最大安全加速倍率", 1.15, 1.0, 2.0, 0.01),
            WidgetPort(10, "max_text_error_rate", "FLOAT", "最大 CER/WER", 0.2, 0.0, 1.0, 0.01),
        ),
        (
            OutputPort(9, 0, "audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", "时长适配批次"),
            OutputPort(9, 1, "manifest_path", "STRING", "时长适配 Manifest"),
            OutputPort(9, 2, "duration_retry_line_ids", "STRING", "时长重做 line ID"),
            OutputPort(10, 0, "qa", "T8_FIREREDAUDIO_SPEECH_QA", "Speech QA"),
            OutputPort(10, 1, "qa_report", "STRING", "QA 报告"),
            OutputPort(10, 2, "qa_failed_line_ids", "STRING", "QA 失败 line ID"),
        ),
    ),
    SubgraphSpec(
        "FireRedAudio Local Speech Repair.json",
        "21_podcast_local_repair.json",
        "FireRedAudio · 局部语音修复",
        (4, 5, 6),
        (
            WidgetPort(4, "range_mode", "COMBO", "范围来源", "manual"),
            WidgetPort(4, "start_seconds", "FLOAT", "开始时间（秒）", 2.0, 0.0, 86400.0, 0.01),
            WidgetPort(4, "end_seconds", "FLOAT", "结束时间（秒）", 5.0, 0.01, 86400.0, 0.01),
            WidgetPort(4, "context_ms", "INT", "上下文（毫秒）", 250, 0, 5000, 1),
            WidgetPort(5, "instruction", "STRING", "编辑指令", "replace the last sentence with: 这是一段已经修复的台词"),
            WidgetPort(5, "edit_type", "COMBO", "编辑类型", "semantic"),
            WidgetPort(6, "crossfade_ms", "INT", "回填交叉淡化（毫秒）", 40, 0, 2000, 1),
        ),
        (
            OutputPort(6, 0, "original_audio", "AUDIO", "原始音频"),
            OutputPort(6, 1, "repaired_audio", "AUDIO", "修复音频"),
            OutputPort(6, 2, "replacement_report", "STRING", "替换报告"),
            OutputPort(5, 1, "edited_text", "STRING", "编辑后文本"),
        ),
    ),
    SubgraphSpec(
        "FireRedAudio Creative Line Candidates.json",
        "28_creative_line_candidates.json",
        "FireRedAudio · 单句创意候选与采用",
        (9, 10, 11),
        (
            WidgetPort(9, "candidate_count", "INT", "新候选数量", 3, 2, 7, 1),
            WidgetPort(9, "seed_start", "INT", "起始 Seed", 1001, 0, 0xFFFFFFFF - 700000, 1),
            WidgetPort(9, "seed_step", "INT", "Seed 间隔", 97, 1, 100000, 1),
            WidgetPort(9, "include_original", "BOOLEAN", "把原 Take 加入盲听", True),
            WidgetPort(9, "run_asr_qa", "BOOLEAN", "逐个 ASR 回读", False),
            WidgetPort(10, "selected_position", "INT", "采用候选序号（0=仅盲听）", 0, 0, 8, 1),
            WidgetPort(10, "ratings_json", "STRING", "评分 JSON", "{}"),
            WidgetPort(10, "notes_json", "STRING", "备注 JSON", "{}"),
        ),
        (
            OutputPort(11, 0, "audio_batch", "T8_FIREREDAUDIO_AUDIO_BATCH", "回填后批次"),
            OutputPort(11, 1, "selected_audio", "AUDIO", "已采用音频"),
            OutputPort(11, 2, "manifest_path", "STRING", "采用 Manifest"),
            OutputPort(11, 3, "adoption_report", "STRING", "采用报告"),
            OutputPort(9, 3, "candidate_report", "STRING", "Seed 与候选证据"),
        ),
    ),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stable_uuid(name: str) -> str:
    return str(uuid.uuid5(NAMESPACE, name))


def _build_template(spec: TemplateSpec) -> Path:
    payload = _read_json(UI_ROOT / spec.source_name)
    payload.setdefault("extra", {})
    payload["extra"].update(
        {
            "frontendVersion": "1.49.6",
            "t8starTemplate": {
                "schema_version": 1,
                "title": spec.title,
                "subtitle": spec.subtitle,
                "source_workflow": f"ui/{spec.source_name}",
            },
        }
    )
    if spec.app_mode:
        payload["extra"]["linearMode"] = True
        payload["extra"]["linearData"] = {
            "inputs": [[node_id, name] for node_id, name in spec.app_inputs],
            "outputs": list(spec.app_outputs),
        }
    target = TEMPLATE_ROOT / spec.output_name
    _write_json(target, payload)
    return target


def _render_thumbnail(path: Path, title: str, subtitle: str, *, app_mode: bool) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - development dependency only
        raise RuntimeError("生成模板预览图需要 Pillow") from exc

    width, height = 640, 360
    image = Image.new("RGB", (width, height), "#11131a")
    draw = ImageDraw.Draw(image)
    for y in range(height):
        ratio = y / max(1, height - 1)
        start = (24, 26, 38)
        end = (38, 35, 51)
        color = tuple(round(start[i] * (1 - ratio) + end[i] * ratio) for i in range(3))
        draw.line((0, y, width, y), fill=color)
    draw.rounded_rectangle((34, 34, 606, 326), radius=28, fill="#191c27", outline="#eb7197", width=3)
    draw.rounded_rectangle((58, 58, 164, 92), radius=17, fill="#eb7197")
    draw.ellipse((526, 42, 604, 120), outline="#799cff", width=5)
    draw.ellipse((548, 64, 626, 142), outline="#b985d7", width=3)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    title_font = ImageFont.truetype(str(font_path), 40) if font_path.is_file() else ImageFont.load_default()
    body_font = ImageFont.truetype(str(font_path), 21) if font_path.is_file() else ImageFont.load_default()
    small_font = ImageFont.truetype(str(font_path), 16) if font_path.is_file() else ImageFont.load_default()
    draw.text((78, 66), "T8STAR", font=small_font, fill="#ffffff")
    draw.text((62, 138), title, font=title_font, fill="#ffffff")
    draw.text((64, 208), subtitle, font=body_font, fill="#b8bfd2")
    badge = "APP MODE" if app_mode else "GRAPH WORKFLOW"
    badge_width = draw.textlength(badge, font=small_font) + 34
    draw.rounded_rectangle((62, 270, 62 + badge_width, 306), radius=18, fill="#242838", outline="#799cff")
    draw.text((79, 278), badge, font=small_font, fill="#dbe4ff")
    image.save(path, "JPEG", quality=92, optimize=True)


def _link_object(raw: list[Any]) -> dict[str, Any]:
    link_id, source_id, source_slot, target_id, target_slot, kind = raw
    return {
        "id": int(link_id),
        "origin_id": int(source_id),
        "origin_slot": int(source_slot),
        "target_id": int(target_id),
        "target_slot": int(target_slot),
        "type": kind,
    }


def _build_subgraph(spec: SubgraphSpec) -> Path:
    source = _read_json(UI_ROOT / spec.source_name)
    selected_ids = set(spec.node_ids)
    source_nodes = {int(node["id"]): node for node in source["nodes"]}
    nodes = [copy.deepcopy(source_nodes[node_id]) for node_id in spec.node_ids]
    node_by_id = {int(node["id"]): node for node in nodes}
    source_links = [_link_object(link) for link in source.get("links", [])]
    internal_links = [
        copy.deepcopy(link)
        for link in source_links
        if link["origin_id"] in selected_ids and link["target_id"] in selected_ids
    ]
    incoming = [
        link
        for link in source_links
        if link["origin_id"] not in selected_ids and link["target_id"] in selected_ids
    ]
    outgoing_ids = {
        link["id"]
        for link in source_links
        if link["origin_id"] in selected_ids and link["target_id"] not in selected_ids
    }
    internal_ids = {link["id"] for link in internal_links}
    for node in nodes:
        for item in node.get("inputs", []):
            if item.get("link") not in internal_ids:
                item["link"] = None
        for item in node.get("outputs", []):
            item["links"] = [
                int(link_id)
                for link_id in (item.get("links") or [])
                if int(link_id) in internal_ids and int(link_id) not in outgoing_ids
            ] or None

    next_link = max([int(source.get("last_link_id", 0)), *(link["id"] for link in internal_links)], default=0) + 1
    next_node = max(spec.node_ids) + 1
    subgraph_inputs: list[dict[str, Any]] = []
    top_inputs: list[dict[str, Any]] = []
    proxy_widgets: list[list[str]] = []
    proxy_constraints: list[dict[str, Any] | None] = []
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for link in incoming:
        grouped.setdefault((link["origin_id"], link["origin_slot"]), []).append(link)
    used_names: set[str] = set()
    for input_index, (_source_key, links) in enumerate(grouped.items()):
        first = links[0]
        target_node = node_by_id[first["target_id"]]
        target_input = target_node["inputs"][first["target_slot"]]
        base_name = str(target_input.get("name") or f"input_{input_index + 1}")
        name = base_name
        suffix = 2
        while name in used_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(name)
        port_links: list[int] = []
        for external in links:
            link_id = next_link
            next_link += 1
            port_links.append(link_id)
            target = node_by_id[external["target_id"]]
            target["inputs"][external["target_slot"]]["link"] = link_id
            internal_links.append(
                {
                    "id": link_id,
                    "origin_id": -10,
                    "origin_slot": input_index,
                    "target_id": external["target_id"],
                    "target_slot": external["target_slot"],
                    "type": external["type"],
                }
            )
        label = str(target_input.get("localized_name") or target_input.get("label") or name)
        subgraph_inputs.append(
            {
                "id": _stable_uuid(f"{spec.output_name}:input:{name}"),
                "name": name,
                "type": first["type"],
                "linkIds": port_links,
                "localized_name": name,
                "label": label,
                "pos": [-180, 80 + input_index * 30],
            }
        )
        top_input = {
            "name": name,
            "type": first["type"],
            "link": None,
            "label": label,
            "localized_name": name,
        }
        if first["type"] in {"STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"}:
            top_input["widget"] = {"name": name}
            proxy_widgets.append(["-1", name])
        top_inputs.append(top_input)

    for widget in spec.widgets:
        node = node_by_id[widget.node_id]
        slot = len(node.setdefault("inputs", []))
        link_id = next_link
        next_link += 1
        node["inputs"].append(
            {
                "name": widget.name,
                "type": widget.kind,
                "widget": {"name": widget.name},
                "link": link_id,
            }
        )
        if widget.kind == "COMBO":
            # A combo's option list belongs to the destination node schema, so
            # keep it on the subgraph boundary.  All curated combo defaults are
            # deliberately the first legal option (zh/wav/speech_aware/etc.).
            port_index = len(subgraph_inputs)
            internal_links.append(
                {
                    "id": link_id,
                    "origin_id": -10,
                    "origin_slot": port_index,
                    "target_id": widget.node_id,
                    "target_slot": slot,
                    "type": widget.kind,
                }
            )
            subgraph_inputs.append(
                {
                    "id": _stable_uuid(
                        f"{spec.output_name}:widget:{widget.node_id}:{widget.name}"
                    ),
                    "name": widget.name,
                    "type": widget.kind,
                    "linkIds": [link_id],
                    "localized_name": widget.name,
                    "label": widget.label,
                    "pos": [-180, 80 + port_index * 30],
                }
            )
            top_inputs.append(
                {
                    "name": widget.name,
                    "type": widget.kind,
                    "widget": {"name": widget.name},
                    "link": None,
                    "label": widget.label,
                    "localized_name": widget.name,
                }
            )
            proxy_widgets.append(["-1", widget.name])
            continue

        primitive_type = {
            "INT": "PrimitiveInt",
            "FLOAT": "PrimitiveFloat",
            "BOOLEAN": "PrimitiveBoolean",
            "STRING": "PrimitiveStringMultiline",
        }.get(widget.kind)
        if primitive_type is None:
            raise ValueError(f"{spec.output_name}: 不支持的代理控件类型 {widget.kind}")
        primitive_id = next_node
        next_node += 1
        primitive_values = (
            [widget.default, "fixed"] if widget.kind == "INT" else [widget.default]
        )
        properties: dict[str, Any] = {"Node name for S&R": primitive_type}
        proxy_constraint: dict[str, Any] | None = None
        if widget.kind in {"INT", "FLOAT"} and (
            widget.minimum is not None or widget.maximum is not None
        ):
            proxy_constraint = {
                "min": widget.minimum,
                "max": widget.maximum,
                "step": widget.step,
                "integer": widget.kind == "INT",
                "label": widget.label,
            }
            properties["t8_firered_constraint"] = proxy_constraint
        primitive = {
            "id": primitive_id,
            "type": primitive_type,
            "pos": [-420, 80 + len(proxy_widgets) * 92],
            "size": [300, 120 if widget.kind == "STRING" else 64],
            "flags": {},
            "order": len(nodes),
            "mode": 0,
            "inputs": [
                {
                    "label": widget.label,
                    "localized_name": "value",
                    "name": "value",
                    "type": widget.kind,
                    "widget": {"name": "value"},
                    "link": None,
                }
            ],
            "outputs": [
                {
                    "localized_name": widget.kind,
                    "name": widget.kind,
                    "type": widget.kind,
                    "links": [link_id],
                }
            ],
            "properties": properties,
            "widgets_values": primitive_values,
            "title": widget.label,
        }
        nodes.append(primitive)
        node_by_id[primitive_id] = primitive
        internal_links.append(
            {
                "id": link_id,
                "origin_id": primitive_id,
                "origin_slot": 0,
                "target_id": widget.node_id,
                "target_slot": slot,
                "type": widget.kind,
            }
        )
        proxy_widgets.append([str(primitive_id), "value"])
        proxy_constraints.append(proxy_constraint)

    subgraph_outputs: list[dict[str, Any]] = []
    top_outputs: list[dict[str, Any]] = []
    for output_index, output in enumerate(spec.outputs):
        node = node_by_id[output.node_id]
        while len(node.setdefault("outputs", [])) <= output.slot:
            raise ValueError(f"{spec.output_name}: node {output.node_id} 没有输出槽 {output.slot}")
        link_id = next_link
        next_link += 1
        links = list(node["outputs"][output.slot].get("links") or [])
        links.append(link_id)
        node["outputs"][output.slot]["links"] = links
        internal_links.append(
            {
                "id": link_id,
                "origin_id": output.node_id,
                "origin_slot": output.slot,
                "target_id": -20,
                "target_slot": output_index,
                "type": output.kind,
            }
        )
        subgraph_outputs.append(
            {
                "id": _stable_uuid(f"{spec.output_name}:output:{output.name}"),
                "name": output.name,
                "type": output.kind,
                "linkIds": [link_id],
                "localized_name": output.name,
                "label": output.label,
                "pos": [1420, 80 + output_index * 30],
            }
        )
        top_outputs.append(
            {
                "name": output.name,
                "type": output.kind,
                "links": [],
                "label": output.label,
                "localized_name": output.name,
            }
        )

    subgraph_id = _stable_uuid(f"subgraph:{spec.output_name}")
    top_node = {
        "id": 1,
        "type": subgraph_id,
        "pos": [40, 40],
        "size": [360, max(180, 90 + max(len(top_inputs), len(top_outputs)) * 26)],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": top_inputs,
        "outputs": top_outputs,
        "properties": {
            "proxyWidgets": proxy_widgets,
            "t8_firered_proxy_constraints": proxy_constraints,
            "cnr_id": "comfyui-fireredaudio-t8",
            "ver": "0.19.1",
        },
        "widgets_values": [],
        "title": spec.title,
    }
    definition = {
        "id": subgraph_id,
        "version": 1,
        "state": {
            "lastGroupId": 0,
            "lastNodeId": max(int(node["id"]) for node in nodes),
            "lastLinkId": next_link - 1,
            "lastRerouteId": 0,
        },
        "revision": 0,
        "config": {},
        "name": spec.title,
        "inputNode": {"id": -10, "bounding": [-220, 40, 160, max(100, len(subgraph_inputs) * 30 + 40)]},
        "outputNode": {"id": -20, "bounding": [1420, 40, 160, max(100, len(subgraph_outputs) * 30 + 40)]},
        "inputs": subgraph_inputs,
        "outputs": subgraph_outputs,
        "widgets": [],
        "nodes": nodes,
        "groups": [],
        "links": internal_links,
        "extra": {"t8starSourceWorkflow": f"example_workflows/ui/{spec.source_name}"},
    }
    payload = {
        "revision": 0,
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [top_node],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {"frontendVersion": "1.49.6"},
        "version": 0.4,
        "definitions": {"subgraphs": [definition]},
    }
    target = SUBGRAPH_ROOT / spec.output_name
    _write_json(target, payload)
    return target


def build(*, thumbnails: bool = True) -> dict[str, Any]:
    templates = []
    for spec in TEMPLATES:
        path = _build_template(spec)
        if thumbnails:
            _render_thumbnail(path.with_suffix(".jpg"), spec.title, spec.subtitle, app_mode=spec.app_mode)
        templates.append(path.name)
    subgraphs = [_build_subgraph(spec).name for spec in SUBGRAPHS]
    report = {
        "schema_version": 1,
        "templates": templates,
        "subgraphs": subgraphs,
        "app_mode_templates": [spec.output_name for spec in TEMPLATES if spec.app_mode],
        "graph_templates": [spec.output_name for spec in TEMPLATES if not spec.app_mode],
    }
    _write_json(ROOT / "scripts" / "v017-experience-manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FireRedAudio v0.17 ComfyUI templates and subgraphs")
    parser.add_argument("--no-thumbnails", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(thumbnails=not args.no_thumbnails), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
