from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from webui.label_registry import LabelRegistry
from webui.server import TactileApplication


class LabelRegistryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        manifest = self.root / "datasets/failrecovery/manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps({
                "id": "failure_001", "task_key": "usb_insert",
                "ground_truth_outcome": "failure", "total_frames": 30,
                "camera_video_paths": {"cam_high": "video.mp4"},
            }) + "\n" + json.dumps({
                "id": "success_001", "task_key": "usb_insert",
                "ground_truth_outcome": "success", "total_frames": 30,
                "camera_video_paths": {"cam_high": "video.mp4"},
            }) + "\n", encoding="utf-8",
        )
        self.app = TactileApplication(self.root)
        self.app.load_rollouts()

    def payload(self):
        return self.app.labels.snapshot()

    def test_defaults_and_display_edits_do_not_change_saved_event_keys(self):
        self.assertEqual([row["id"] for row in self.payload()["labels"]], [1, 2, 3, 4])
        original = [{"event_key": 3, "start_frame": 4, "end_frame": 10}]
        self.app.save_rollout_annotations("failure_001", original)
        data = self.payload()
        data["labels"][2]["name"] = "Object dropped"
        data["labels"][2]["description"] = "Expanded explanation"
        data["labels"][2]["color"] = "#aabbcc"
        self.app.labels.update(data)
        reread = LabelRegistry(self.root / "config/annotation_labels.json")
        self.assertEqual(reread.get(3)["name"], "Object dropped")
        self.assertEqual(reread.get(3)["color"], "#aabbcc")
        self.assertEqual([row["event_key"] for row in self.app.annotations_by_rollout()["failure_001"]], [3])

    def test_registered_labels_work_on_any_rollout(self):
        self.app.save_rollout_annotations("success_001", [
            {"event_key": 3, "start_frame": 2, "end_frame": 4},
            {"event_key": 4, "start_frame": 5, "end_frame": 7},
        ])
        self.assertEqual(
            [event["event_key"] for event in self.app.annotations_by_rollout()["success_001"]],
            [3, 4],
        )
        data = self.payload()
        data["labels"].append({
            "id": 5, "name": "Contact lost", "description": "",
            "color": "#123456",
        })
        self.app.labels.update(data)
        self.app.save_rollout_annotations("success_001", [
            {"event_key": 5, "start_frame": 8, "end_frame": 10},
        ])
        self.assertEqual(self.app.annotations_by_rollout()["success_001"][0]["event_key"], 5)
        with self.assertRaisesRegex(ValueError, "existing IDs"):
            self.app.labels.update({
                **self.payload(), "labels": self.payload()["labels"][:4]
            })

    def test_old_failure_and_inactive_metadata_is_ignored(self):
        data = self.payload()
        data["labels"][2]["scope"] = "failure"
        data["labels"][2]["active"] = False
        data["labels"][3]["scope"] = "failure"
        self.app.labels.update(data)
        self.assertNotIn("scope", self.app.labels.get(3))
        self.assertNotIn("active", self.app.labels.get(3))
        self.app.save_rollout_annotations("success_001", [
            {"event_key": 3, "start_frame": 1, "end_frame": 3},
            {"event_key": 4, "start_frame": 4, "end_frame": 6},
        ])
        self.assertEqual(len(self.app.annotations_by_rollout()["success_001"]), 2)

    def test_local_registry_file_from_old_version_loads_without_restrictions(self):
        config = self.root / "config/annotation_labels.json"
        content = self.payload()
        for label in content["labels"]:
            label["scope"] = "failure"
            label["active"] = False
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps(content), encoding="utf-8")
        self.app.labels = LabelRegistry(config)
        self.assertTrue(all("scope" not in row and "active" not in row
                            for row in self.payload()["labels"]))
        self.app.save_rollout_annotations("success_001", [
            {"event_key": 3, "start_frame": 1, "end_frame": 3}
        ])

    def test_unregistered_historical_label_kept_but_not_coerced_on_save(self):
        seed = self.root / "outputs/usb_event_intervals/old/intervals.jsonl"
        seed.parent.mkdir(parents=True)
        seed.write_text(json.dumps({
            "rollout_id": "failure_001", "event_key": 7, "start_frame": 0, "end_frame": 2
        }) + "\n", encoding="utf-8")
        self.assertEqual([row["event_key"] for row in self.app.annotations_by_rollout()["failure_001"]], [7])
        with self.assertRaisesRegex(ValueError, "unregistered"):
            self.app.save_rollout_annotations("failure_001", self.app.annotations_by_rollout()["failure_001"])
        self.assertFalse(self.app.annotation_record_path("failure_001").exists())

    def test_registry_rejects_invalid_color_and_duplicate_keys(self):
        data = self.payload()
        data["labels"][0]["color"] = "red; background:url(x)"
        with self.assertRaisesRegex(ValueError, "color"):
            self.app.labels.update(data)
        data = self.payload()
        data["labels"].append(dict(data["labels"][0]))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self.app.labels.update(data)

    def test_ui_has_dynamic_label_metadata_and_no_unknown_to_success_coercion(self):
        static = Path(__file__).resolve().parents[1] / "static"
        html = (static / "index.html").read_text(encoding="utf-8")
        js = (static / "shell.js").read_text(encoding="utf-8")
        self.assertIn('id="labelRegistryRows"', html)
        self.assertIn('id="annotationLabelGuide"', html)
        self.assertIn('id="resultsLabelFilters"', html)
        self.assertIn("function renderLabelSettings()", js)
        self.assertIn("function renderAnnotationLabelGuide()", js)
        self.assertIn("function labelMeta(id)", js)
        self.assertNotIn('label.scope === "failure"', js)
        self.assertNotIn("data-field=\"scope\"", js)
        self.assertNotIn("data-field=\"active\"", js)
        self.assertIn("event.event_key = Number(event.event_key)", js)
        self.assertNotIn("LABEL_KEYS.indexOf(Number(event.event_key))", js)


if __name__ == "__main__":
    unittest.main()
