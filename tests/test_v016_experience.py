from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class V016ExperienceTests(unittest.TestCase):
    def test_curated_templates_are_directly_discoverable_and_have_previews(self) -> None:
        manifest = json.loads(
            (ROOT / "scripts" / "v017-experience-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(manifest["templates"]), 7)
        self.assertEqual(len(manifest["app_mode_templates"]), 4)
        self.assertEqual(len(manifest["graph_templates"]), 3)
        for name in manifest["templates"]:
            path = ROOT / "example_workflows" / name
            self.assertTrue(path.is_file(), name)
            self.assertTrue(path.with_suffix(".jpg").is_file(), name)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("t8starTemplate", payload["extra"])
            if name in manifest["app_mode_templates"]:
                self.assertTrue(payload["extra"]["linearMode"])
                self.assertTrue(payload["extra"]["linearData"]["inputs"])
                self.assertTrue(payload["extra"]["linearData"]["outputs"])

    def test_subgraph_blueprints_have_no_dangling_links(self) -> None:
        paths = sorted((ROOT / "subgraphs").glob("*.json"))
        self.assertEqual(len(paths), 5)
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 0.4)
            self.assertEqual(len(payload["definitions"]["subgraphs"]), 1)
            definition = payload["definitions"]["subgraphs"][0]
            self.assertEqual(payload["nodes"][0]["type"], definition["id"])
            self.assertLessEqual(
                len(payload["nodes"][0].get("widgets_values", [])),
                len(payload["nodes"][0]["properties"].get("proxyWidgets", [])),
                path.name,
            )
            definition_node_ids = {str(node["id"]) for node in definition["nodes"]}
            for node_id, widget_name in payload["nodes"][0]["properties"].get(
                "proxyWidgets", []
            ):
                self.assertTrue(node_id == "-1" or node_id in definition_node_ids, path.name)
                self.assertTrue(widget_name, path.name)
            node_ids = {-10, -20, *(int(node["id"]) for node in definition["nodes"])}
            links = definition["links"]
            link_ids = [int(link["id"]) for link in links]
            self.assertEqual(len(link_ids), len(set(link_ids)), path.name)
            for link in links:
                self.assertIn(int(link["origin_id"]), node_ids, path.name)
                self.assertIn(int(link["target_id"]), node_ids, path.name)
            referenced = {
                int(link_id)
                for item in (*definition["inputs"], *definition["outputs"])
                for link_id in item["linkIds"]
            }
            self.assertTrue(referenced)
            self.assertTrue(referenced <= set(link_ids), path.name)
        creative = json.loads(
            (ROOT / "subgraphs" / "FireRedAudio Creative Line Candidates.json").read_text(
                encoding="utf-8"
            )
        )
        creative_nodes = {
            int(node["id"]): node for node in creative["definitions"]["subgraphs"][0]["nodes"]
        }
        self.assertEqual(creative_nodes[9]["widgets_values"][:5], [3, 1001, 97, True, False])
        self.assertEqual(creative_nodes[10]["widgets_values"][0], 0)
        primitive_values = {
            node.get("title"): node["widgets_values"]
            for node in creative_nodes.values()
            if str(node.get("type", "")).startswith("Primitive")
        }
        self.assertEqual(primitive_values["新候选数量"], [3, "fixed"])
        self.assertEqual(primitive_values["起始 Seed"], [1001, "fixed"])
        self.assertEqual(primitive_values["Seed 间隔"], [97, "fixed"])
        self.assertEqual(primitive_values["把原 Take 加入盲听"], [True])
        self.assertEqual(primitive_values["逐个 ASR 回读"], [False])
        self.assertEqual(primitive_values["采用候选序号（0=仅盲听）"], [0, "fixed"])
        constraints = {
            node.get("title"): node.get("properties", {}).get(
                "t8_firered_constraint"
            )
            for node in creative_nodes.values()
            if str(node.get("type", "")).startswith("Primitive")
        }
        self.assertEqual(
            constraints["新候选数量"],
            {"min": 2, "max": 7, "step": 1, "integer": True, "label": "新候选数量"},
        )
        self.assertEqual(constraints["采用候选序号（0=仅盲听）"]["min"], 0)
        self.assertEqual(constraints["采用候选序号（0=仅盲听）"]["max"], 8)
        proxy_constraints = creative["nodes"][0]["properties"][
            "t8_firered_proxy_constraints"
        ]
        self.assertEqual(len(proxy_constraints), 8)
        self.assertEqual(proxy_constraints[0]["max"], 7)
        self.assertEqual(proxy_constraints[5]["max"], 8)

    def test_subgraph_number_constraints_have_frontend_enforcement(self) -> None:
        source = (ROOT / "web" / "subgraph_controls_v017.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("t8_firered_constraint", source)
        self.assertIn("t8_firered_proxy_constraints", source)
        self.assertIn("applyProxyConstraints", source)
        self.assertIn("Math.max", source)
        self.assertIn("Math.min", source)
        self.assertIn("PrimitiveInt", source)
        self.assertIn("PrimitiveFloat", source)

    def test_take_review_frontend_requires_explicit_blind_selection(self) -> None:
        source = (ROOT / "web" / "take_review_v017.js").read_text(encoding="utf-8")
        self.assertIn("fireredaudio_take_review", source)
        self.assertIn("selected_line_id", source)
        self.assertIn("匿名盲听", source)
        self.assertIn("重新运行工作流", source)

    def test_creative_candidate_workflow_keeps_generation_review_and_adoption_separate(self) -> None:
        payload = json.loads(
            (ROOT / "example_workflows" / "api" / "28_creative_line_candidates.json").read_text(
                encoding="utf-8"
            )
        )
        by_type = {value["class_type"]: (node_id, value) for node_id, value in payload.items()}
        pool_id, pool = by_type["T8_FireRedAudio_CreativeCandidatePool"]
        review_id, review = by_type["T8_FireRedAudio_TakeReviewBoard"]
        apply_id, apply_node = by_type["T8_FireRedAudio_CandidateApply"]
        self.assertEqual(review["inputs"]["audio_batch"], [pool_id, 0])
        self.assertEqual(apply_node["inputs"]["reviewed_candidates"], [review_id, 1])
        self.assertEqual(apply_node["inputs"]["selected_candidate_id"], [review_id, 2])
        export = by_type["T8_FireRedAudio_SaveAudioBatch"][1]
        self.assertEqual(export["inputs"]["audio_batch"], [apply_id, 0])
        self.assertTrue(pool["inputs"]["include_original"])
        self.assertEqual(pool["inputs"]["seed_step"], 97)

    def test_creative_template_widget_values_match_current_node_schema(self) -> None:
        payload = json.loads(
            (
                ROOT
                / "example_workflows"
                / "FireRedAudio_07_Creative_Line_Candidates.json"
            ).read_text(encoding="utf-8")
        )
        by_type = {node["type"]: node for node in payload["nodes"]}
        self.assertEqual(
            by_type["T8_FireRedAudio_ModelLoader"]["widgets_values"],
            [
                "FireRedAudio",
                "auto",
                "auto",
                "auto_safe",
                "full",
                "managed",
                False,
                False,
                "",
                "",
                "",
                "",
            ],
        )
        self.assertEqual(
            by_type["T8_FireRedAudio_GenerationSettings"]["widgets_values"],
            ["balanced", 1001, "fixed", 750, 6, 512, 10, 2.0],
        )
        self.assertEqual(
            by_type["T8_FireRedAudio_TakeReviewBoard"]["widgets_values"],
            [0, "{}", "{}", "creative-blind-review", "fireredaudio/reviews", 8, ""],
        )

    def test_core_nodes_ship_default_chinese_and_english_help(self) -> None:
        required = {
            "T8_FireRedAudio_ModelLoader",
            "T8_FireRedAudio_TTS",
            "T8_FireRedAudio_SpeechEdit",
            "T8_FireRedAudio_DurationFit",
            "T8_FireRedAudio_BatchRetry",
            "T8_FireRedAudio_CreativeCandidatePool",
            "T8_FireRedAudio_CandidateApply",
            "T8_FireRedAudio_AccelerationBenchmark",
        }
        for node_id in required:
            fallback = ROOT / "web" / "docs" / f"{node_id}.md"
            zh = ROOT / "web" / "docs" / node_id / "zh.md"
            en = ROOT / "web" / "docs" / node_id / "en.md"
            self.assertTrue(fallback.is_file(), node_id)
            self.assertTrue(zh.is_file(), node_id)
            self.assertTrue(en.is_file(), node_id)
            self.assertIn("# ", fallback.read_text(encoding="utf-8"))
            self.assertIn("## How to use", en.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
