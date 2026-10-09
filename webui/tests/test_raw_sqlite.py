"""Original recorder SQLite, peer manifests and no-export WebUI tests."""
from __future__ import annotations

import json
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from webui.server import TactileApplication


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def make_sqlite(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, source TEXT, "
                   "sensor_ts_ns INTEGER, receive_wall_ns INTEGER, receive_mono_ns INTEGER, valid INTEGER)")
        db.execute("CREATE TABLE tactile_frames (event_id INTEGER, side TEXT, finger TEXT, "
                   "f6_json TEXT, raw_shape_json TEXT, deform_shape_json TEXT, raw_blob BLOB, deform_blob BLOB)")
        db.execute("CREATE TABLE synchronized_frames (frame_index INTEGER, elapsed_s REAL, "
                   "tick_wall_ns INTEGER, tick_mono_ns INTEGER, complete INTEGER, snapshot_json TEXT)")
        db.execute("CREATE TABLE camera_frames (source TEXT, frame_index INTEGER)")
        for fid, finger in enumerate(FINGERS, 1):
            db.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                       (fid, f"tactile:right:{finger}", 1000 + fid, 2000 + fid, 3000 + fid, 1))
            pixels = bytes([0, 255, 255, 0])
            db.execute("INSERT INTO tactile_frames VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (fid, "right", finger, "[1,2,3,4,5,6]", "[2,2]", "[2,2]", pixels, pixels))
        for frame_number in range(3):
            snapshot = {
                "camera:realsense_color": {"frame_index": frame_number},
                "camera:wrist_right": {"frame_index": frame_number + 4}
            }
            for fid, finger in enumerate(FINGERS, 1):
                snapshot[f"tactile:right:{finger}"] = {
                    "event_id": fid, "valid": True, "age_ms": 1, "stale": False
                }
            db.execute("INSERT INTO synchronized_frames VALUES (?, ?, ?, ?, ?, ?)",
                       (frame_number, frame_number / 30, 1, 2, 1, json.dumps(snapshot)))
            db.execute("INSERT INTO camera_frames VALUES (?, ?)",
                       ("camera:realsense_color", frame_number))
        db.commit()


def add_raw_manifest(webui_root: Path, label: str, source_root: Path, identifier: str):
    dataset = webui_root / "datasets" / label
    dataset.mkdir(parents=True)
    video_dir = source_root / "ep01/videos"
    video_dir.mkdir(parents=True)
    for name in ("realsense_color.mp4", "wrist_right.mp4"):
        (video_dir / name).write_bytes(b"mp4")
    db = source_root / "ep01/episode.sqlite3"
    make_sqlite(db)
    row = {
        "id": identifier, "data_root": str(source_root),
        "source_database_path": "ep01/episode.sqlite3",
        "camera_video_paths": {
            "cam_high": "ep01/videos/realsense_color.mp4",
            "cam_wrist": "ep01/videos/wrist_right.mp4",
        },
        "total_frames": 3, "fps": 30, "task_key": "usb_insert"
    }
    (dataset / "manifest.jsonl").write_text(json.dumps(row) + "\n")
    return db


class OriginalSqliteTests(unittest.TestCase):
    def test_direct_sqlite_full_tactile_api_and_read_only_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "viewer"
            source = Path(temp) / "raw"
            db = add_raw_manifest(root, "raw_dataset", source, "raw-one")
            before = db.read_bytes()
            app = TactileApplication(root)
            self.assertEqual(len(app.load_rollouts()), 1)
            self.assertEqual(app.source_file("ep01/videos/realsense_color.mp4", rollout_id="raw-one"),
                             source / "ep01/videos/realsense_color.mp4")
            service = app.tactile
            self.assertTrue(service.has_rollout("raw-one"))
            frame = service.frame("raw-one", "cam_high", 1)
            self.assertEqual(frame["matched_video_frame"], 1)
            self.assertEqual(frame["fingers"]["thumb"]["f6"], [1, 2, 3, 4, 5, 6])
            self.assertEqual(frame["fingers"]["thumb"]["image_kinds"], ["raw", "deform"])
            series = service.series("raw-one", "cam_high")
            self.assertEqual(len(series["sync_frames"]), 3)
            self.assertEqual(len(series["fingers"]["thumb"]), 1)
            frame_wrist = service.frame("raw-one", "cam_wrist", 6)
            self.assertEqual(frame_wrist["matched_video_frame"], 6)
            png = service.image("raw-one", "thumb", "1", "deform")
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", png[16:24]), (2, 2))
            sprite = service.sprite("raw-one", "cam_high", 1, "deform")
            self.assertEqual(struct.unpack(">II", sprite[16:24]), (10, 2))
            self.assertEqual(db.read_bytes(), before)
            self.assertFalse((source / "ep01/frames.jsonl").exists())
            self.assertFalse((source / "ep01/tactile").exists())

    def test_metadata_indexer_writes_only_tiny_manifest(self):
        from tools.index_sqlite_dataset import index_dataset
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "viewer"
            raw = Path(temp) / "raw_dataset"
            db = add_raw_manifest(root, "old_setup", raw, "original-one")
            (raw / "ep01/manifest.json").write_text(
                json.dumps({"status": "complete", "task": "Insert USB", "episode_label": "ep01"})
            )
            before = db.read_bytes()
            target = root / "datasets" / "new_raw" / "manifest.jsonl"
            with mock.patch("tools.index_sqlite_dataset.video_probe", return_value={
                "fps": 30.0, "frames": 3, "width": 320, "height": 240, "codec": "h264"
            }):
                indexed = index_dataset(raw, target, "new_raw")
            self.assertEqual(indexed[0]["id"], "new_raw--ep01")
            self.assertEqual(db.read_bytes(), before)
            self.assertLess(target.stat().st_size, 4096)
            self.assertTrue(target.with_suffix(".summary.json").is_file())
            app = TactileApplication(root)
            self.assertEqual(len(app.rollout_map()), 2)
            self.assertEqual(app.tactile.frame("new_raw--ep01", "cam_high", 1)["matched_video_frame"], 1)

    def test_peer_manifests_and_independent_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "viewer"
            db1 = add_raw_manifest(root, "first", Path(temp) / "original_a", "rollout-a")
            db2 = add_raw_manifest(root, "second", Path(temp) / "original_b", "rollout-b")
            app = TactileApplication(root)
            self.assertEqual(set(app.rollout_map()), {"rollout-a", "rollout-b"})
            self.assertEqual(len(app.rollout_payload()["manifests"]), 2)
            self.assertEqual(app.tactile.image("rollout-b", "index", "2", "raw")[:8],
                             b"\x89PNG\r\n\x1a\n")
            self.assertEqual(
                app.source_file("ep01/videos/wrist_right.mp4", rollout_id="rollout-b"),
                db2.parent / "videos/wrist_right.mp4",
            )
            self.assertTrue(db1.is_file())
            self.assertTrue(db2.is_file())

    def test_symlink_manifest_and_legacy_original_database_override(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "viewer"
            original_root = Path(temp) / "original"
            db = add_raw_manifest(root, "temporary", original_root, "old-id")
            link = root / "datasets/temporary/manifest.jsonl"
            link.unlink()
            linked_manifest = Path(temp) / "LF3R/datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl"
            linked_manifest.parent.mkdir(parents=True)
            row = {
                "id": "old-id", "task_key": "usb", "source_database_path": str(db),
                "camera_video_paths": {"cam_high": "datasets/copy.mp4"},
                "synchronized_frames_path": "datasets/frames.jsonl",
                "tactile_events_path": "datasets/tactile/events.jsonl"
            }
            linked_manifest.write_text(json.dumps(row) + "\n")
            link.symlink_to(linked_manifest)
            app = TactileApplication(root)
            rollout = app.rollout_map()["old-id"]
            self.assertEqual(
                app.source_file(rollout["camera_video_paths"]["cam_high"], ".mp4", "old-id"),
                db.parent / "videos/realsense_color.mp4"
            )
            self.assertEqual(app.tactile.frame("old-id", "cam_high", 1)["matched_video_frame"], 1)

    def test_duplicate_rollout_ids_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "viewer"
            add_raw_manifest(root, "a", Path(temp) / "raw_a", "same-id")
            add_raw_manifest(root, "b", Path(temp) / "raw_b", "same-id")
            app = TactileApplication(root)
            self.assertEqual(len(app.rollout_map()), 1)
            self.assertTrue(any(
                not item["valid"] and "duplicate rollout" in item["error"]
                for item in app.rollout_payload()["manifests"]
            ))

    def test_catalog_detects_added_dataset_after_startup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "viewer"
            add_raw_manifest(root, "a", Path(temp) / "raw_a", "a")
            app = TactileApplication(root)
            add_raw_manifest(root, "b", Path(temp) / "raw_b", "b")
            self.assertEqual(set(app.rollout_map()), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
