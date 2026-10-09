"""Smoke test for the standalone, read-only Deform WebP benchmark."""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

try:
    from PIL import features
    import benchmark_deform_webp
except ImportError:
    benchmark_deform_webp = None


@unittest.skipUnless(
    benchmark_deform_webp is not None and features.check("webp"),
    "Pillow with WebP support is required",
)
class DeformWebPBenchmarkTest(unittest.TestCase):
    def test_one_episode_produces_results_without_touching_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "episode-001" / "episode.sqlite3"
            source.parent.mkdir()
            with sqlite3.connect(source) as db:
                db.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, source TEXT)")
                db.execute(
                    "CREATE TABLE tactile_frames "
                    "(event_id INTEGER, finger TEXT, deform_shape_json TEXT, deform_blob BLOB)"
                )
                for number in range(1, 16):
                    finger = ("thumb", "index", "middle", "ring", "pinky")[number % 5]
                    data = bytes((x * 7 + number * 11) % 256 for x in range(32 * 32))
                    db.execute("INSERT INTO events VALUES (?, ?)",
                               (number, "tactile:right:" + finger))
                    db.execute("INSERT INTO tactile_frames VALUES (?, ?, ?, ?)",
                               (number, finger, "[32,32]", data))
                db.commit()
            before = source.read_bytes()
            output = base / "test-results"
            result = benchmark_deform_webp.run(
                Namespace(path=source, output=output, max_images=0,
                          quality_every=3, random_reads=8, keep_cache=False)
            )
            self.assertEqual(result, output)
            self.assertEqual(before, source.read_bytes())
            self.assertTrue((output / "report.md").is_file())
            self.assertTrue((output / "comparison.png").is_file())
            self.assertFalse((output / "cache_q80.sqlite3").exists())
            record = json.loads((output / "results.json").read_text())
            self.assertEqual(record["tested_images"], 15)
            self.assertEqual(record["raw_rowid_read_ms"]["n"], 8)
            self.assertGreater(record["q80_cache_sqlite_bytes"], 0)
            self.assertGreater(record["qualities"]["PNG"]["compressed_bytes"], 0)
            self.assertGreater(record["qualities"]["Q80"]["compressed_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
