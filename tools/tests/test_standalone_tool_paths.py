"""Standalone tools keep outputs local and resolve shared raw data explicitly."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sharpa_tactile.common import ROOT, CODE_ROOT, DATA_ROOT, project_path, relative_path


class StandaloneToolPathsTest(unittest.TestCase):
    def test_new_output_stays_local(self):
        self.assertEqual(ROOT,CODE_ROOT)
        self.assertEqual(project_path('outputs','new_run','predictions.csv'),ROOT/'outputs/new_run/predictions.csv')

    def test_manifest_and_recorded_sensor_paths(self):
        manifest=project_path('datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl')
        if not manifest.is_file():self.skipTest('Shared dataset not installed')
        with manifest.open() as handle:record=json.loads(handle.readline())
        self.assertTrue(manifest.samefile(DATA_ROOT/'datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl'))
        raw=project_path(record['synchronized_frames_path'])
        self.assertTrue(raw.is_file())
        self.assertTrue(raw.samefile(DATA_ROOT/record['synchronized_frames_path']))
        stored=relative_path(raw)
        self.assertTrue((stored if stored.is_absolute() else ROOT/stored).samefile(raw))

    def test_historical_absolute_result_reference_relocates(self):
        old=DATA_ROOT/'outputs/sharpa_tactile_three_class/20261004_160624/data_manifest.json'
        expected=ROOT/'outputs/sharpa_tactile_three_class/20261004_160624/data_manifest.json'
        self.assertEqual(project_path(old),expected)
        if expected.is_file():self.assertEqual(relative_path(old),expected.relative_to(ROOT))

    def test_local_annotation_and_checkpoint_paths(self):
        self.assertEqual(project_path('annotations/tactile_canonical/v1'),ROOT/'annotations/tactile_canonical/v1')
        self.assertEqual(project_path('checkpoints/T-Rex/encoders'),ROOT/'checkpoints/T-Rex/encoders')


if __name__=='__main__':unittest.main()
