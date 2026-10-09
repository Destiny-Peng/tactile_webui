"""Browser-video transcode: per-manifest job, original preservation, cache reuse."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from webui.server import TactileApplication
from webui.video_cache import VideoTranscodeManager, cache_path, compatible_video
from tools.transcode_manifest_videos_h264 import probe

ROOT = Path(__file__).resolve().parents[2]


class VideoCacheTests(unittest.TestCase):
    def test_cache_name_changes_with_original_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.mp4"
            source.write_bytes(b"one")
            cached = cache_path(source, root / "cache")
            self.assertEqual(compatible_video(source, root / "cache"), source)
            cached.parent.mkdir()
            cached.write_bytes(b"compatible")
            self.assertEqual(compatible_video(source, root / "cache"), cached)
            source.write_bytes(b"changed original data")
            self.assertNotEqual(cache_path(source, root / "cache"), cached)
            self.assertEqual(compatible_video(source, root / "cache"), source)

    def test_manifest_scope_deduplicates_and_disallows_unknown_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "raw/video.mp4"
            original.parent.mkdir()
            original.write_bytes(b"mp4")
            row = {"id": "test",
                   "data_root": str(original.parent),
                   "camera_video_paths": {"cam_high": "video.mp4", "cam_wrist": "video.mp4"},
                   "total_frames": 5}
            manifest = root / "datasets/test/manifest.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps(row) + "\n")
            app = TactileApplication(root)
            manager = VideoTranscodeManager(root)
            with self.assertRaises(ValueError):
                manager.submit(app, ["not-in-catalog"])
            # Avoid spawning a child here; only inspect the accepted batch.
            from unittest.mock import patch
            with patch("webui.video_cache.subprocess.Popen") as fake:
                fake.return_value.poll.return_value = None
                submitted = manager.submit(app, [app._manifest_info[0]["path"]])
            self.assertEqual(submitted["total"], 1)
            plan = json.loads((root / "logs/webui/transcode" / submitted["job_id"] / "plan.json").read_text())
            self.assertEqual(plan["videos"], [str(original)])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg and ffprobe required for live transcode smoke test")
    def test_real_mpeg4_to_h264_and_idempotent_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / "raw/video.mp4"
            original.parent.mkdir()
            subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
                "-i", "color=c=black:s=64x64:r=10:d=0.4",
                "-c:v", "mpeg4", "-pix_fmt", "yuv420p", str(original),
            ], check=True)
            self.assertEqual(probe(original)["codec_name"], "mpeg4")
            saved = original.read_bytes()
            plan = root / "plan.json"
            status = root / "status.json"
            cachedir = root / "cache"
            plan.write_text(json.dumps({"videos": [str(original)], "manifests": ["test"]}))
            from tools.transcode_manifest_videos_h264 import run
            self.assertEqual(run(plan, status, cachedir), 0)
            stats = json.loads(status.read_text())
            self.assertEqual(stats["converted"], 1)
            self.assertEqual(probe(cache_path(original, cachedir))["codec_name"], "h264")
            self.assertEqual(original.read_bytes(), saved)
            self.assertEqual(compatible_video(original, cachedir), cache_path(original, cachedir))
            self.assertEqual(run(plan, status, cachedir), 0)
            replay = json.loads(status.read_text())
            self.assertEqual(replay["cached"], 1)
            self.assertEqual(replay["converted"], 0)
            self.assertEqual(original.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
