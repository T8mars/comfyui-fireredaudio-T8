from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_schemas(comfy_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(comfy_root.resolve()))
    package_name = "comfyui_fireredaudio_t8_validation"
    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("无法载入节点包")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    extension = asyncio.run(module.comfy_entrypoint())
    node_classes = asyncio.run(extension.get_node_list())
    schemas = {}
    for node_class in node_classes:
        schema = node_class.GET_SCHEMA()
        schemas[schema.node_id] = schema
    return schemas


def validate_examples(schemas: dict[str, Any]) -> dict[str, int]:
    known_core_outputs = {"LoadAudio": 1, "SaveAudio": 1}
    ui_count = 0
    api_count = 0
    for path in sorted((ROOT / "example_workflows" / "ui").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        nodes = {int(node["id"]): node for node in payload["nodes"]}
        for node in nodes.values():
            node_type = node["type"]
            if node_type.startswith("T8_FireRedAudio_") and node_type not in schemas:
                raise ValueError(f"{path.name}: 未注册节点 {node_type}")
        for link_id, source_id, source_slot, target_id, target_slot, _kind in payload["links"]:
            if source_id not in nodes or target_id not in nodes:
                raise ValueError(f"{path.name}: link {link_id} 指向不存在节点")
            source_outputs = nodes[source_id].get("outputs", [])
            target_inputs = nodes[target_id].get("inputs", [])
            if int(source_slot) >= len(source_outputs):
                raise ValueError(f"{path.name}: link {link_id} 源输出槽越界")
            if int(target_slot) >= len(target_inputs):
                raise ValueError(f"{path.name}: link {link_id} 目标输入槽越界")
        ui_count += 1

    for path in sorted((ROOT / "example_workflows" / "api").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for node_id, node in payload.items():
            node_type = node["class_type"]
            schema = schemas.get(node_type)
            if node_type.startswith("T8_FireRedAudio_") and schema is None:
                raise ValueError(f"{path.name}: 未注册节点 {node_type}")
            if schema is not None:
                allowed = {item.id for item in schema.inputs}
                for key in node.get("inputs", {}):
                    if key not in allowed and not any(key.startswith(f"{prefix}.") for prefix in allowed):
                        raise ValueError(f"{path.name}:{node_id}: 未知输入 {node_type}.{key}")
            for key, value in node.get("inputs", {}).items():
                if not (
                    isinstance(value, list)
                    and len(value) == 2
                    and str(value[0]) in payload
                    and isinstance(value[1], int)
                ):
                    continue
                source_type = payload[str(value[0])]["class_type"]
                output_count = (
                    len(schemas[source_type].outputs)
                    if source_type in schemas
                    else known_core_outputs.get(source_type)
                )
                if output_count is not None and value[1] >= output_count:
                    raise ValueError(
                        f"{path.name}:{node_id}.{key}: {source_type} 输出槽 {value[1]} 越界"
                    )
        api_count += 1
    return {"schemas": len(schemas), "ui_workflows": ui_count, "api_workflows": api_count}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FireRedAudio nodes against a real ComfyUI checkout")
    parser.add_argument("--comfy-root", required=True)
    args = parser.parse_args()
    schemas = load_schemas(Path(args.comfy_root))
    result = validate_examples(schemas)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
