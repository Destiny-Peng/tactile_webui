from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from webui.server import TactileApplication


class WorkspaceAnnotationTest(unittest.TestCase):
    def make_app(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        manifest = root / "datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "id": "usb_001",
                    "task_key": "usb_insert",
                    "task_description": "Insert USB",
                    "total_frames": 100,
                    "fps": 30,
                    "camera_video_paths": {"cam_high": "datasets/video.mp4"},
                }
            )
            + "\n"
        )
        app = TactileApplication(root)
        app.load_rollouts()
        return temporary, app

    def test_four_event_types_roundtrip(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        events = [
            {"event_key": 6, "start_frame": 10, "end_frame": 20},
            {"event_key": 7, "start_frame": 30, "end_frame": 40},
            {"event_key": 8, "start_frame": 50, "end_frame": 60},
            {"event_key": 9, "start_frame": 70, "end_frame": 80},
        ]
        saved = app.save_rollout_annotations("usb_001", events)
        self.assertEqual([row["event_key"] for row in saved], [6, 7, 8, 9])
        self.assertEqual(
            [row["event_name"] for row in saved],
            ["align_failure", "insert_failure", "align_success", "insert_success"],
        )
        self.assertEqual(len(app.annotations_by_rollout()["usb_001"]), 4)
        self.assertTrue(app.annotation_target_path().is_file())

    def test_rejects_invalid_interval(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        with self.assertRaises(ValueError):
            app.save_rollout_annotations(
                "usb_001", [{"event_key": 6, "start_frame": 80, "end_frame": 120}]
            )

    def test_historical_seed_is_copied_not_modified(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        seed = app.root / "outputs/usb_event_intervals/old/intervals.jsonl"
        seed.parent.mkdir(parents=True)
        original = {
            "event_id": "usb_001:event:0",
            "rollout_id": "usb_001",
            "event_index": 0,
            "event_key": 6,
            "start_frame": 5,
            "end_frame": 9,
        }
        seed.write_text(json.dumps(original) + "\n")
        before = seed.read_text()
        app.save_rollout_annotations(
            "usb_001", [{"event_key": 8, "start_frame": 15, "end_frame": 19}]
        )
        self.assertEqual(seed.read_text(), before)
        loaded = app.annotations_by_rollout()["usb_001"]
        self.assertEqual([row["event_key"] for row in loaded], [8])


if __name__ == "__main__":
    unittest.main()
