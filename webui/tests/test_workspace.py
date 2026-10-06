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
        manifest = root / "datasets/failrecovery/manifest.jsonl"
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
        self.assertTrue(all("event_name" not in row for row in saved))
        self.assertEqual(len(app.annotations_by_rollout()["usb_001"]), 4)
        self.assertEqual(app.annotation_target_path(), app.root / "annotations/failrecovery/records")
        self.assertTrue(app.annotation_record_path("usb_001").is_file())
        record = json.loads(app.annotation_record_path("usb_001").read_text())
        self.assertEqual([row["event_key"] for row in record["tactile_intervals"]], [6, 7, 8, 9])

    def test_existing_lf3r_record_is_not_overwritten(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        records = app.root / "annotations/failrecovery/records"
        records.mkdir(parents=True)
        legacy = records / "usb_001.json"
        legacy.write_text(json.dumps({"schema_version": 2, "rollout_id": "usb_001", "notes": "keep me"}))
        app.save_rollout_annotations("usb_001", [{"event_key": 8, "start_frame": 10, "end_frame": 20}])
        self.assertEqual(json.loads(legacy.read_text())["notes"], "keep me")
        self.assertTrue((records / "usb_001.tactile.json").is_file())

    def test_external_source_project_root_resolves_manifest_paths(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        external = app.root / "old_project"
        manifest = external / "datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        video = external / "datasets/shared/video.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        manifest.write_text(json.dumps({
            "id": "usb_external", "task_key": "usb_insert", "total_frames": 1, "fps": 30,
            "camera_video_paths": {"cam_high": "datasets/shared/video.mp4"}
        }) + "\n")
        app.save_settings({**app.settings, "source_project_root": str(external)})
        self.assertEqual(app.manifest_path, manifest.resolve())
        self.assertEqual(app.source_file("datasets/shared/video.mp4"), video.resolve())
        self.assertIn("usb_external", app.rollout_map())


    def test_symlinked_manifest_infers_original_project_root(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        external_temporary = tempfile.TemporaryDirectory()
        self.addCleanup(external_temporary.cleanup)
        external = Path(external_temporary.name)

        old_manifest = (
            external
            / "datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl"
        )
        old_manifest.parent.mkdir(parents=True)
        video_rel = (
            "datasets/lf3r_failure_rollouts/v1/"
            "failrecovery/episode_001/videos/cam_high.mp4"
        )
        video = external / video_rel
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video")
        old_manifest.write_text(
            json.dumps(
                {
                    "id": "usb_symlink",
                    "task_key": "usb_insert",
                    "total_frames": 1,
                    "fps": 30,
                    "camera_video_paths": {"cam_high": video_rel},
                }
            )
            + "\n"
        )
        seed = external / "outputs/usb_event_intervals/run/intervals.jsonl"
        seed.parent.mkdir(parents=True)
        seed.write_text(
            json.dumps(
                {
                    "rollout_id": "usb_symlink",
                    "event_key": 6,
                    "start_frame": 0,
                    "end_frame": 0,
                }
            )
            + "\n"
        )

        local_manifest = app.root / "datasets/failrecovery/manifest.jsonl"
        local_manifest.unlink()
        local_manifest.symlink_to(old_manifest)

        app = TactileApplication(app.root)
        self.assertEqual(app.manifest_path, old_manifest.resolve())
        self.assertEqual(app.data_root, external.resolve())
        self.assertEqual(app.source_file(video_rel), video.resolve())
        self.assertEqual(app.annotation_seed_path(), seed.resolve())
        self.assertIn("usb_symlink", app.rollout_map())

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
