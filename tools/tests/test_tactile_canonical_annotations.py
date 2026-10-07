"""Migration contracts: union, exact dedup, authoritative outcome, immutable bounds."""
import json
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa_tactile.canonical_annotations import merge_intervals, validate_event, validate_output
from sharpa_tactile.common import ROOT, project_path, canonical_key, binary_target, is_stage


class CanonicalMigrationTests(unittest.TestCase):
    def setUp(self):
        self.records = {'s': dict(id='s', ground_truth_outcome='success', total_frames=30),
                        'f': dict(id='f', ground_truth_outcome='failure', total_frames=30)}

    def row(self, rid, key, start=2, end=8):
        return (dict(rollout_id=rid, event_key=key, start_frame=start, end_frame=end),
                dict(source='test.json', source_event_index=0, source_failure_type='timeout_no_progress'))

    def test_union_mapping_filter_and_exact_dedup(self):
        inputs = [self.row('f', k) for k in (6,7,8,9,3,4)]
        inputs += [self.row('f',3,3,8), self.row('f',3,2,9), self.row('f',0)]
        inputs += [self.row('s',3), self.row('s',4), self.row('s',6)]
        events, summary, audit = merge_intervals(inputs, self.records)
        self.assertEqual(summary['exact_duplicates'],2)
        self.assertEqual(summary['discarded_success_34'],2)
        self.assertEqual(summary['background_records_ignored'],1)
        self.assertEqual(len(events),7)
        self.assertEqual({e['event_key'] for e in events},{1,2,3,4})
        self.assertEqual(sum(e['event_key']==3 for e in events),3)
        self.assertEqual(sum(len(e['provenance']) for e in events),9)
        self.assertTrue(any(e['rollout_id']=='s' and e['event_key']==2 for e in events))
        self.assertTrue(all(e['rollout_id']=='f' for e in events if e['event_key'] in (3,4)))
        self.assertEqual(inputs[0][0]['event_key'],6)

    def test_invalid_frames_and_missing_authority_fail_closed(self):
        for start,end in ((-1,2),(4,2),(0,30),(1.5,2),(True,2)):
            with self.subTest(start=start,end=end), self.assertRaises(ValueError):
                merge_intervals([self.row('f',6,start,end)],self.records)
        self.records['f']['ground_truth_outcome']='unknown'
        with self.assertRaises(ValueError):
            merge_intervals([self.row('f',3)],self.records)
        with self.assertRaises(ValueError):
            merge_intervals([self.row('missing',6)],self.records)

    def test_consumers_reject_legacy_and_independent_labels_as_binary(self):
        for key in (6,7,8,9,0,5,'1'):
            with self.assertRaises(ValueError):canonical_key(dict(event_key=key))
        for key in (3,4):
            with self.assertRaises(ValueError):binary_target(dict(event_key=key))
        self.assertEqual(binary_target(dict(event_key=1)),0)
        self.assertEqual(binary_target(dict(event_key=2)),1)
        self.assertTrue(is_stage(dict(event_key=2,stages=['align']), 'align'))

    def test_actual_output(self):
        path=ROOT/'annotations/tactile_canonical/v1'
        if not path.exists():self.skipTest('Canonical artifact not present in this checkout')
        events=validate_output(path,project_path('datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl'))
        summary=json.loads((path/'migration_summary.json').read_text())
        self.assertEqual(len(events),summary['output_intervals'])
        self.assertEqual(summary['input_intervals'],summary['output_intervals']+summary['exact_duplicates']+
                         summary['discarded_success_34']+summary['background_records_ignored'])
        for key,stats in summary['final'].items():
            chosen=[e for e in events if e['event_key']==int(key)]
            self.assertEqual(len(chosen),stats['intervals'])
            self.assertEqual(len({e['rollout_id'] for e in chosen}),stats['rollouts'])


class CanonicalConsumerTests(unittest.TestCase):
    def test_independent_overlapping_channels_and_binary_projection(self):
        from sharpa_tactile.common import annotation_timeline, binary_timeline, three_class_timeline
        events=[dict(event_key=k,start_frame=1,end_frame=3) for k in (2,3,4)]
        self.assertEqual(annotation_timeline(events,5).tolist(),[[0,0,0,0]]+[[0,1,1,1]]*3+[[0,0,0,0]])
        labels,conflict=binary_timeline(events,5)
        self.assertEqual(labels.tolist(),[-1,1,1,1,-1]);self.assertFalse(conflict.any())
        self.assertEqual(three_class_timeline(events,5).tolist(),[0,2,2,2,0])
        with self.assertRaises(ValueError):annotation_timeline([dict(event_key=6,start_frame=1,end_frame=2)],5)

    def test_actual_loader_retains_all_four_labels(self):
        from sharpa_tactile.data import load_sources
        sources=load_sources(ROOT/'annotations/tactile_canonical/v1/intervals.jsonl',
                             project_path('datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl'))
        events=[e for _,events in sources.values() for e in events]
        self.assertEqual(len(sources),120);self.assertEqual(len(events),353)
        self.assertEqual({e['event_key'] for e in events},{1,2,3,4})
        self.assertEqual(sum(e['event_key'] in (3,4) for e in events),94)


if __name__=='__main__':unittest.main()
