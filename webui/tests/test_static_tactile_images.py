import json
import tempfile
import unittest
from pathlib import Path

from tools.materialize_tactile_images import infer_data_root, materialize_rollout
from webui.tactile_service import FailRecoveryTactileService


class StaticTactileImageTests(unittest.TestCase):
    def test_materialized_image_is_preferred(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        episode = root / "datasets/failrecovery/episode_001"
        tactile = episode / "tactile"
        tactile.mkdir(parents=True)

        deform = bytes([0, 64, 128, 255])
        (tactile / "index.deform.u8").write_bytes(deform)
        (tactile / "events.jsonl").write_text(json.dumps({
            "event_id": 11,
            "finger": "index",
            "sample_index": 3,
            "f6": [0, 0, 0, 0, 0, 0],
            "deform_offset_bytes": 0,
            "deform_length_bytes": len(deform),
            "deform_shape": [2, 2],
        }) + "\n", encoding="utf-8")
        (episode / "frames.jsonl").write_text(json.dumps({
            "camera_frame_indices": {"cam_high": 0},
            "tactile": {"index": {"event_id": 11, "valid": True}},
        }) + "\n", encoding="utf-8")

        row = {
            "id": "episode_001",
            "synchronized_frames_path": "datasets/failrecovery/episode_001/frames.jsonl",
            "tactile_events_path": "datasets/failrecovery/episode_001/tactile/events.jsonl",
            "tactile_stream_paths": {
                "index": {"deform": "datasets/failrecovery/episode_001/tactile/index.deform.u8"}
            },
        }
        manifest = root / "datasets/failrecovery/manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

        summary = materialize_rollout(root, row, ("deform",), False)
        self.assertEqual(summary["deform"], 1)
        expected = tactile / "images/deform/index/000003.png"
        self.assertTrue(expected.is_file())

        service = FailRecoveryTactileService(root, manifest)
        self.assertEqual(service.static_image_path("episode_001", "index", "11", "deform"), expected)
        self.assertEqual(service.image("episode_001", "index", "11", "deform"), expected.read_bytes())

    def test_infer_data_root_follows_external_manifest_paths(self):
        standalone_tmp = tempfile.TemporaryDirectory()
        external_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(standalone_tmp.cleanup)
        self.addCleanup(external_tmp.cleanup)
        standalone = Path(standalone_tmp.name)
        external = Path(external_tmp.name)

        relative = "datasets/lf3r_failure_rollouts/v1/failrecovery/ep/frames.jsonl"
        referenced = external / relative
        referenced.parent.mkdir(parents=True)
        referenced.write_text("{}\n", encoding="utf-8")
        manifest = external / "datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "id": "ep",
            "synchronized_frames_path": relative,
        }) + "\n", encoding="utf-8")

        local_entry = standalone / "datasets/failrecovery/manifest.jsonl"
        local_entry.parent.mkdir(parents=True)
        local_entry.symlink_to(manifest)
        self.assertEqual(infer_data_root(local_entry.resolve(), standalone), external.resolve())


if __name__ == "__main__":
    unittest.main()
