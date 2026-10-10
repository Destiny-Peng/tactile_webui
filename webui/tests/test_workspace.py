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
                    "ground_truth_outcome": "failure",
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
            {"event_key": 1, "start_frame": 10, "end_frame": 20},
            {"event_key": 2, "start_frame": 30, "end_frame": 40},
            {"event_key": 3, "start_frame": 50, "end_frame": 60},
            {"event_key": 4, "start_frame": 70, "end_frame": 80},
        ]
        saved = app.save_rollout_annotations("usb_001", events)
        self.assertEqual([row["event_key"] for row in saved], [1, 2, 3, 4])
        self.assertTrue(all("event_name" not in row for row in saved))
        self.assertEqual(len(app.annotations_by_rollout()["usb_001"]), 4)
        self.assertEqual(app.annotation_target_path(), app.root / "annotations/failrecovery/records")
        self.assertTrue(app.annotation_record_path("usb_001").is_file())
        record = json.loads(app.annotation_record_path("usb_001").read_text())
        self.assertEqual([row["event_key"] for row in record["tactile_intervals"]], [1, 2, 3, 4])

    def test_independent_labels_on_all_outcomes_keep_provenance(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        events = [{"event_key": key, "start_frame": 10, "end_frame": 20,
                   "provenance": [{"source": "canonical source"}]} for key in (2, 3, 4)]
        saved = app.save_rollout_annotations("usb_001", events)
        self.assertEqual(len(saved), 3)
        self.assertTrue(all(event["provenance"] for event in saved))
        for outcome in ("success", "failure", "unknown", None):
            app.rollout_map()["usb_001"]["ground_truth_outcome"] = outcome
            saved = app.save_rollout_annotations("usb_001", events)
            self.assertEqual({event["event_key"] for event in saved}, {2, 3, 4})

    def test_existing_lf3r_record_is_not_overwritten(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        records = app.root / "annotations/failrecovery/records"
        records.mkdir(parents=True)
        legacy = records / "usb_001.json"
        legacy.write_text(json.dumps({"schema_version": 2, "rollout_id": "usb_001", "notes": "keep me"}))
        app.save_rollout_annotations("usb_001", [{"event_key": 3, "start_frame": 10, "end_frame": 20}])
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
                    "event_key": 1,
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

    def test_symlinked_manifest_uses_datasets_parent_even_without_probe_hits(self):
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
        # Deliberately use a manifest row whose referenced payload is absent.
        # Root inference must come from the resolved manifest location, not
        # from successful existence probes.
        old_manifest.write_text(
            json.dumps(
                {
                    "id": "usb_symlink_path_only",
                    "task_key": "usb_insert",
                    "camera_video_paths": {
                        "cam_high": (
                            "datasets/lf3r_failure_rollouts/v1/"
                            "failrecovery/not_materialized/cam_high.mp4"
                        )
                    },
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
        self.assertTrue(app.tactile.has_rollout("usb_symlink_path_only"))

    def test_manifest_outcome_and_embedded_annotation_are_not_review_ground_truth(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        # The source catalog can carry an inferred outcome or legacy annotation.
        # Neither may label a rollout after the review migration.
        manifest = app.manifest_path
        row = json.loads(manifest.read_text())
        row["ground_truth_outcome"] = "failure"
        row["annotation"] = {
            "outcome_label": "success",
            "review_status": "complete",
            "annotator": "manifest_import",
        }
        manifest.write_text(json.dumps(row) + "\n")
        app = TactileApplication(app.root)
        payload = app.rollout_payload()["rollouts"][0]
        self.assertEqual(payload["ground_truth_outcome"], "failure")
        self.assertNotIn("outcome_label", payload["annotation"])
        self.assertEqual(payload["annotation_status"], "unreviewed")

        # The migrated sidecar is the only authoritative source, including
        # for rollout outcomes that contradict the old manifest.
        target = app.annotation_record_path("usb_001")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "rollout_id": "usb_001",
            "outcome_label": "recovered_success",
            "annotator": "human",
            "review_status": "complete",
            "tactile_intervals": [],
        }))
        reviewed = app.rollout_payload()["rollouts"][0]
        self.assertEqual(reviewed["annotation"]["outcome_label"], "recovered_success")
        self.assertEqual(reviewed["annotation_status"], "complete")
        self.assertEqual(reviewed["annotation"]["annotator"], "human")
        self.assertEqual(reviewed["ground_truth_outcome"], "failure")

    def test_manifest_outcome_is_not_inferred_when_review_only_has_intervals(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        app.save_rollout_annotations(
            "usb_001", [{"event_key": 1, "start_frame": 5, "end_frame": 8}]
        )
        row = app.rollout_payload()["rollouts"][0]
        self.assertEqual(row["ground_truth_outcome"], "failure")
        self.assertEqual(row["annotation_status"], "complete")
        self.assertNotIn("outcome_label", row["annotation"])
        self.assertEqual(len(row["annotation_events"]), 1)

    def test_outcome_filter_has_unlabeled_distinct_from_uncertain(self):
        static = Path(__file__).resolve().parents[1] / "static"
        js = (static / "shell.js").read_text(encoding="utf-8")
        html = (static / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("record.ground_truth_outcome", js)
        self.assertNotIn("Source manifest outcome:", js)
        self.assertIn('return ["success", "failure", "recovered_success", "uncertain"]', js)
        self.assertIn('"unlabeled"', js)
        self.assertIn('<option value="">Unlabeled</option>', html)

    def test_full_rollout_review_is_separate_from_interval_labels(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        source_manifest = app.manifest_path
        original_manifest = source_manifest.read_bytes()
        saved = app.save_rollout_annotations(
            "usb_001",
            [{"event_key": 1, "start_frame": 4, "end_frame": 10},
             {"event_key": 2, "start_frame": 20, "end_frame": 30}],
            review={
                "outcome_label": "recovered_success",
                "review_status": "complete",
                "annotator": "reviewer_A",
                "confidence": 0.9,
                "notes": "Failed on first attempt, later recovered.",
            },
        )
        self.assertEqual(len(saved), 2)
        sidecar = json.loads(app.annotation_record_path("usb_001").read_text())
        self.assertEqual(sidecar["outcome_label"], "recovered_success")
        self.assertEqual(sidecar["review_status"], "complete")
        self.assertEqual(sidecar["annotator"], "reviewer_A")
        self.assertEqual(sidecar["confidence"], 0.9)
        self.assertEqual(sidecar["notes"], "Failed on first attempt, later recovered.")
        self.assertEqual([row["event_key"] for row in sidecar["tactile_intervals"]], [1, 2])
        self.assertEqual(original_manifest, source_manifest.read_bytes())
        row = app.rollout_payload()["rollouts"][0]
        self.assertEqual(row["ground_truth_outcome"], "failure")
        self.assertEqual(row["annotation"]["outcome_label"], "recovered_success")
        self.assertEqual(row["annotation_status"], "complete")
        self.assertEqual([x["event_key"] for x in row["annotation_events"]], [1, 2])

        # An old interval-only client must preserve the new review metadata.
        app.save_rollout_annotations("usb_001", saved)
        updated = app.rollout_payload()["rollouts"][0]
        self.assertEqual(updated["annotation"]["outcome_label"], "recovered_success")
        self.assertEqual(updated["annotation"]["annotator"], "reviewer_A")

    def test_review_edit_preserves_legacy_confidence_without_requiring_it(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        app.save_rollout_annotations("usb_001", [], review={
            "outcome_label": "success", "annotator": "r1",
            "review_status": "in_progress", "confidence": 0.4,
        })
        app.save_rollout_annotations("usb_001", [], review={
            "outcome_label": "failure", "annotator": "r1",
            "review_status": "complete", "notes": "Confirmed failure",
        })
        record = json.loads(app.annotation_record_path("usb_001").read_text())
        self.assertEqual(record["confidence"], 0.4)
        self.assertEqual(record["review_status"], "complete")
        self.assertEqual(record["outcome_label"], "failure")

    def test_review_without_intervals_can_be_complete(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        app.save_rollout_annotations("usb_001", [], review={
            "outcome_label": "success", "review_status": "complete",
            "annotator": "reviewer_B", "confidence": None, "notes": "",
        })
        row = app.rollout_payload()["rollouts"][0]
        self.assertEqual(row["annotation_events"], [])
        self.assertEqual(row["annotation_status"], "complete")
        self.assertEqual(row["annotation"]["outcome_label"], "success")
        self.assertEqual(row["ground_truth_outcome"], "failure")
        # The new annotation survives a complete application restart.
        reopened = TactileApplication(app.root)
        self.assertEqual(
            reopened.rollout_payload()["rollouts"][0]["annotation"]["outcome_label"],
            "success",
        )

    def test_reviewer_can_set_uncertain_without_relabeling_source(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        for outcome in ("failure", "success", "recovered_success", "uncertain"):
            app.save_rollout_annotations("usb_001", [], review={
                "outcome_label": outcome, "review_status": "in_progress",
                "annotator": "human_1",
            })
            actual = app.rollout_payload()["rollouts"][0]
            self.assertEqual(actual["annotation"]["outcome_label"], outcome)
            self.assertEqual(actual["annotation_status"], "in_progress")
            self.assertEqual(actual["ground_truth_outcome"], "failure")
        app.save_rollout_annotations("usb_001", [], review={
            "outcome_label": None, "review_status": "unreviewed", "annotator": "",
        })
        actual = app.rollout_payload()["rollouts"][0]
        self.assertNotIn("outcome_label", actual["annotation"])
        self.assertEqual(actual["annotation_status"], "unreviewed")

    def test_invalid_review_is_rejected_without_writing(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        cases = (
            {"outcome_label": "failure", "annotator": ""},
            {"outcome_label": "success", "annotator": "someone", "confidence": 1.4},
            {"outcome_label": "something_else", "annotator": "someone"},
            {"outcome_label": "success", "annotator": "someone", "review_status": "bogus"},
        )
        for review in cases:
            with self.assertRaises(ValueError):
                app.save_rollout_annotations("usb_001", [], review=review)
        self.assertFalse(app.annotation_record_path("usb_001").exists())

    def test_deleted_intervals_do_not_reappear_from_historical_seed(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        seed = app.root / "outputs/usb_event_intervals/sample/intervals.jsonl"
        seed.parent.mkdir(parents=True)
        item = {
            "rollout_id": "usb_001", "event_key": 2,
            "event_index": 0, "start_frame": 1, "end_frame": 3
        }
        seed.write_text(json.dumps(item) + "\n")
        self.assertEqual(len(app.annotations_by_rollout()["usb_001"]), 1)
        app.save_rollout_annotations("usb_001", [], review={
            "outcome_label": "success", "review_status": "complete",
            "annotator": "reviewer_A",
        })
        self.assertEqual(app.annotations_by_rollout().get("usb_001", []), [])
        self.assertEqual(app.rollout_payload()["rollouts"][0]["annotation_status"], "complete")

    def test_annotate_html_contains_rollout_review_controls(self):
        static = Path(__file__).resolve().parents[1] / "static"
        html = (static / "index.html").read_text(encoding="utf-8")
        js = (static / "shell.js").read_text(encoding="utf-8")
        for ident in ("annotateOutcomeLabel", "annotateReviewer",
                      "annotateReviewStatus", "annotateReviewNotes",
                      "annotateSourceOutcome"):
            self.assertIn(f'id="{ident}"', html)
            self.assertIn(f'byId("{ident}")', js)
        self.assertNotIn('id="annotateConfidence"', html)
        self.assertNotIn('byId("annotateConfidence")', js)
        self.assertNotIn('<select id="annotateReviewStatus"', html)
        self.assertIn('review_status: "complete"', js)
        self.assertIn("loadRolloutReviewForm(record)", js)
        self.assertIn("rolloutReviewPayload()", js)

    def test_results_annotation_filter_is_dynamic(self):
        static = Path(__file__).resolve().parents[1] / "static"
        html = (static / "index.html").read_text(encoding="utf-8")
        javascript = (static / "shell.js").read_text(encoding="utf-8")
        self.assertIn('id="resultsLabelFilters"', html)
        self.assertIn("resultAnnotationLabels", javascript)
        self.assertIn("visibleAnnotationLabels", javascript)
        self.assertIn("renderResultsAnnotationTimeline", javascript)
        self.assertIn("Array.isArray(labelKeys) ? labelKeys : Array.from(new Set", javascript)

    def test_rejects_invalid_interval(self):
        temporary, app = self.make_app()
        self.addCleanup(temporary.cleanup)
        with self.assertRaises(ValueError):
            app.save_rollout_annotations(
                "usb_001", [{"event_key": 1, "start_frame": 80, "end_frame": 120}]
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
            "event_key": 1,
            "start_frame": 5,
            "end_frame": 9,
        }
        seed.write_text(json.dumps(original) + "\n")
        before = seed.read_text()
        app.save_rollout_annotations(
            "usb_001", [{"event_key": 3, "start_frame": 15, "end_frame": 19}]
        )
        self.assertEqual(seed.read_text(), before)
        loaded = app.annotations_by_rollout()["usb_001"]
        self.assertEqual([row["event_key"] for row in loaded], [3])


if __name__ == "__main__":
    unittest.main()
