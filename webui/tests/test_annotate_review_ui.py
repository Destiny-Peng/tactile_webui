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
        self.assertIn(".review-workspace .review-stage { grid-column: 2; }", self.css)

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

    def test_stylesheet_has_no_leftover_merge_conflicts(self):
        for token in ('<<<<<<<', '=======', '>>>>>>>'):
            self.assertNotIn(token, self.css)


if __name__ == "__main__":
    unittest.main()
