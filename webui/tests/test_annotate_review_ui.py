"""Static frontend regressions for Annotate video layout and F6 review."""
import unittest
from pathlib import Path


class AnnotateReviewUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        static = Path(__file__).resolve().parents[1] / "static"
        cls.html = (static / "index.html").read_text(encoding="utf-8")
        cls.js = (static / "shell.js").read_text(encoding="utf-8")
        cls.css = (static / "shell.css").read_text(encoding="utf-8")

    def test_sidebar_is_collapsible_without_removing_navigation(self):
        self.assertIn('id="toggleRolloutSidebar"', self.html)
        self.assertIn('id="annotateSidebar"', self.html)
        self.assertIn('id="annotatePrevious"', self.html)
        self.assertIn('id="annotateNext"', self.html)
        self.assertIn("setRolloutSidebarCollapsed", self.js)
        self.assertIn(".review-workspace.sidebar-collapsed", self.css)
        self.assertIn(".review-workspace .review-stage { grid-column: 2; min-width: 0; }", self.css)
        self.assertIn("grid-template-columns: 2.75rem minmax(0,1fr)", self.css)
        self.assertIn('.queue-panel > :not(.review-queue-toggle)', self.css)
        self.assertIn('localStorage.setItem("tactile.annotate.queueCollapsed"', self.js)
        queue = self.html.split('<aside id="annotateSidebar"')[1].split('</aside>')[0]
        self.assertIn('id="toggleRolloutSidebar"', queue)
        self.assertEqual(self.html.count('id="toggleRolloutSidebar"'), 1)
        task = self.html.split('<section class="task-header">')[1].split('</section>')[0]
        self.assertNotIn('toggleRolloutSidebar', task)

    def test_lf3r_outcome_filter_uses_migrated_annotation_only(self):
        self.assertIn('id="annotateOutcomeFilter"', self.html)
        self.assertIn('<option value="success">Success</option>', self.html)
        self.assertIn('<option value="failure">Failure</option>', self.html)
        self.assertIn('id="annotateReviewFilter"', self.html)
        self.assertIn('id="annotateTaskFilter"', self.html)
        self.assertIn('function rolloutOutcome(record)', self.js)
        self.assertNotIn('record.ground_truth_outcome', self.js)
        self.assertIn('outcome === "success" || outcome === "failure"', self.js)
        self.assertNotIn('<option value="recovered_success">', self.html)
        self.assertNotIn('<option value="uncertain">', self.html)
        self.assertIn('<option value="success">Success</option>', self.html)
        self.assertIn('"unlabeled"', self.js)
        self.assertIn('(outcome === "all" || rolloutOutcome(record) === outcome)', self.js)
        self.assertIn('populateOutcomeFilter(); filterRollouts()', self.js)
        self.assertIn('byId("annotateOutcomeFilter").addEventListener("change", filterRollouts)', self.js)


    def test_header_badges_removed_and_camera_buttons_preserve_frame(self):
        self.assertNotIn('id="annotateBadges"', self.html)
        self.assertNotIn('byId("annotateBadges")', self.js)
        self.assertIn('role="group" aria-label="Video camera view"', self.html)
        self.assertNotIn('<select id="annotateCamera"', self.html)
        self.assertIn('data-camera=', self.js)
        self.assertIn('aria-pressed=', self.js)
        self.assertIn('loadAnnotateVideo(frame, wasPlaying)', self.js)
        self.assertIn('video.currentTime = frame /', self.js)

    def test_f6_values_and_six_channel_history_restore(self):
        self.assertIn('id="annotateF6Finger"', self.html)
        self.assertIn('id="annotateF6Curve"', self.html)
        self.assertIn('formatAnnotateF6(cell.f6)', self.js)
        self.assertIn('state.tactileF6Lookup', self.js)
        self.assertIn('renderAnnotateF6Curve()', self.js)
        self.assertIn('updateAnnotateF6Playhead(value)', self.js)
        self.assertIn('data-f6-playhead', self.js)
        self.assertIn('Math.ceil(points.length / 1400)', self.js)
        self.assertIn('await Promise.all(loads)', self.js)

    def test_lf3r_shortcuts_and_auto_review_are_exposed(self):
        self.assertIn('id="annotateSearch"', self.html)
        self.assertIn('All keyboard shortcuts', self.html)
        self.assertIn('<kbd>A</kbd> add interval', self.html)
        self.assertIn('ten frames', self.html)
        self.assertNotIn('id="annotateConfidence"', self.html)
        self.assertNotIn('<select id="annotateReviewStatus"', self.html)
        self.assertIn('review_status: "complete"', self.js)
        self.assertIn('if (!event.repeat) saveIntervals()', self.js)
        self.assertIn('if (!event.repeat) navigateRollout', self.js)
        self.assertIn('step * (event.shiftKey ? 10 : 1)', self.js)
        self.assertIn('byId("annotateSearch").focus()', self.js)
        self.assertIn('if (!event.repeat) setActiveBoundary(', self.js)
        self.assertIn('if (!event.repeat) setActiveEventType(', self.js)
        self.assertIn('if (label) addInterval(label.id)', self.js)
        self.assertIn('["range", "checkbox", "radio", "button", "submit", "reset"]', self.js)

    def test_stylesheet_has_no_leftover_merge_conflicts(self):
        for token in ('<<<<<<<', '=======', '>>>>>>>'):
            self.assertNotIn(token, self.css)


if __name__ == "__main__":
    unittest.main()
