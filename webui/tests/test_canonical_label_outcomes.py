"""Canonical migration must not filter labels by rollout outcome."""
import unittest

from tools.sharpa_tactile.canonical_annotations import merge_intervals, validate_event


class UnrestrictedCanonicalLabelsTest(unittest.TestCase):
    def test_success_failure_and_unknown_outcomes_preserve_all_labels(self):
        records = {
            "s": {"id": "s", "ground_truth_outcome": "success", "total_frames": 20},
            "f": {"id": "f", "ground_truth_outcome": "failure", "total_frames": 20},
            "u": {"id": "u", "ground_truth_outcome": "unknown", "total_frames": 20},
        }
        inputs = [
            (
                {"rollout_id": rid, "event_key": key, "start_frame": 1, "end_frame": 4},
                {"source": f"{rid}.json"},
            )
            for rid in records for key in (1, 2, 3, 4)
        ]
        events, summary, audit = merge_intervals(inputs, records)
        self.assertEqual(len(events), 12)
        self.assertEqual(summary["exact_duplicates"], 0)
        self.assertEqual(summary["output_intervals"], 12)
        self.assertNotIn("discarded_success_34", summary)
        self.assertNotIn("discarded_success_34", audit)
        for rid in records:
            self.assertEqual(
                {e["event_key"] for e in events if e["rollout_id"] == rid},
                {1, 2, 3, 4},
            )

    def test_interval_bounds_still_validated(self):
        record = {"id": "x", "ground_truth_outcome": "success", "total_frames": 20}
        validate_event({"event_key": 4, "start_frame": 1, "end_frame": 4}, record)
        with self.assertRaisesRegex(ValueError, "Invalid closed frame"):
            validate_event({"event_key": 4, "start_frame": -1, "end_frame": 4}, record)


if __name__ == "__main__":
    unittest.main()
