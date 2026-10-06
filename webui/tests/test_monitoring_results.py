from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from webui.monitoring import WebUIMonitor
from webui.results_service import OnlineResultsService


class MonitoringAndResultsTests(unittest.TestCase):
    def test_monitor_logs_live_under_webui_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            monitor = WebUIMonitor(root / "logs" / "webui")
            monitor.record_request(method="GET", path="/api/videos/x", status=206, resource="video", duration_ms=12.5)
            monitor.record_client_events([{"event": "stalled", "level": "error", "video": "annotate"}])
            snapshot = monitor.snapshot()
            self.assertEqual(Path(snapshot["log_paths"]["server"]), root / "logs" / "webui" / "server")
            self.assertEqual(Path(snapshot["log_paths"]["client"]), root / "logs" / "webui" / "client")
            self.assertEqual(snapshot["resource_counts"]["video"], 1)
            self.assertEqual(len(snapshot["client_errors"]), 1)
            self.assertTrue(list((root / "logs" / "webui" / "server").glob("*.jsonl")))
            self.assertTrue(list((root / "logs" / "webui" / "client").glob("*.jsonl")))

    def test_online_results_reads_existing_prediction_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "outputs" / "align_online" / "runs" / "group" / "seed_42" / "f6_gru"
            run.mkdir(parents=True)
            path = run / "test_predictions.csv"
            fields = [
                "rollout_id", "n", "step", "sample_frames", "sample_ticks",
                "label", "prediction", "p_in_progress", "p_success", "p_failure",
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "rollout_id": "ep_a", "n": 16, "step": 1,
                    "sample_frames": json.dumps(list(range(10, 26))),
                    "sample_ticks": json.dumps(list(range(16))),
                    "label": 0, "prediction": 2,
                    "p_in_progress": .1, "p_success": .2, "p_failure": .7,
                })
                writer.writerow({
                    "rollout_id": "ep_b", "n": 16, "step": 1,
                    "sample_frames": json.dumps(list(range(20, 36))),
                    "sample_ticks": json.dumps(list(range(16))),
                    "label": 1, "prediction": 1,
                    "p_in_progress": .1, "p_success": .8, "p_failure": .1,
                })
            service = OnlineResultsService(root)
            sources = service.list_sources("outputs")
            self.assertEqual(len(sources), 1)
            result = service.curve("outputs", sources[0]["id"], "ep_a")
            self.assertEqual(result["probability_labels"], [0, 1, 2])
            self.assertEqual(len(result["points"]), 1)
            self.assertEqual(result["points"][0]["frame"], 25)
            self.assertEqual(result["points"][0]["label"], 0)
            self.assertEqual(result["points"][0]["prediction"], 2)
            self.assertEqual(result["points"][0]["probabilities"], [0.1, 0.2, 0.7])


if __name__ == "__main__":
    unittest.main()
