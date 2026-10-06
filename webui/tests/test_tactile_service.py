"""Lazy tactile index loading and endpoint compatibility."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tactile_service import FailRecoveryTactileService


class TactileServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for rid in ('first', 'second'):
            folder = self.root / rid
            folder.mkdir()
            (folder / 'frames.jsonl').write_text(json.dumps({
                'camera_frame_indices': {'cam_high': 8}, 'complete': True,
                'tactile': {'thumb': {'event_id': 'e1', 'valid': True}},
            }) + '\n')
            (folder / 'events.jsonl').write_text(json.dumps({
                'event_id': 'e1', 'finger': 'thumb', 'valid': True,
                'f6': [1, 2, 3, 4, 5, 6], 'raw_offset_bytes': 0,
                'raw_length_bytes': 2, 'raw_shape': [1, 2],
            }) + '\n')
            (folder / 'raw.bin').write_bytes(bytes([10, 20]))
            self.rows.append({
                'id': rid, 'synchronized_frames_path': f'{rid}/frames.jsonl',
                'tactile_events_path': f'{rid}/events.jsonl',
                'tactile_stream_paths': {'thumb': {'raw': f'{rid}/raw.bin'}},
            })
        self.manifest = self.root / 'datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl'
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text(''.join(json.dumps(r) + '\n' for r in self.rows))

    def test_startup_reads_manifest_only_and_requests_reuse_one_episode(self):
        with mock.patch.object(FailRecoveryTactileService, '_jsonl', wraps=FailRecoveryTactileService._jsonl) as reader:
            service = FailRecoveryTactileService(self.root)
            self.assertEqual([c.args[0] for c in reader.call_args_list], [self.manifest])
            self.assertTrue(service.has_rollout('first'))
            self.assertEqual(service.episodes, {})
            frame = service.frame('first', 'cam_high', 9)
            self.assertEqual(frame['matched_video_frame'], 8)
            self.assertEqual(frame['fingers']['thumb']['f6'], [1, 2, 3, 4, 5, 6])
            series = service.series('first', 'cam_high')
            self.assertEqual(series['fingers']['thumb'][0]['frame'], 8)
            image = service.image('first', 'thumb', 'e1', 'raw')
            sprite = service.sprite('first', 'cam_high', 8, 'raw')
            self.assertEqual(image[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual(struct.unpack('>II', image[16:24]), (2, 1))
            self.assertEqual(struct.unpack('>II', sprite[16:24]), (10, 1))
            service.frame('first', 'cam_high', 8)
            self.assertEqual(set(service.episodes), {'first'})
            self.assertEqual([c.args[0] for c in reader.call_args_list], [
                self.manifest, self.root/'first/frames.jsonl', self.root/'first/events.jsonl'])

    def test_every_endpoint_can_trigger_first_load(self):
        for endpoint in ('frame', 'series', 'image', 'sprite'):
            with self.subTest(endpoint=endpoint):
                service = FailRecoveryTactileService(self.root)
                args = {'frame': ('cam_high', 8), 'series': ('cam_high',),
                        'image': ('thumb', 'e1', 'raw'), 'sprite': ('cam_high', 8, 'raw')}[endpoint]
                getattr(service, endpoint)('second', *args)
                self.assertEqual(set(service.episodes), {'second'})

    def test_concurrent_requests_load_episode_once(self):
        service = FailRecoveryTactileService(self.root)
        barrier = threading.Barrier(8)
        def request():
            barrier.wait(timeout=5)
            return service.frame('first', 'cam_high', 8)
        with mock.patch.object(service, '_load_episode', wraps=service._load_episode) as loader:
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: request(), range(8)))
            self.assertEqual(loader.call_count, 1)
            self.assertTrue(all(r['matched_video_frame'] == 8 for r in results))

    def test_unrelated_episodes_do_not_wait_for_each_other(self):
        service = FailRecoveryTactileService(self.root)
        entered, release = threading.Event(), threading.Event()
        original = service._load_episode
        def load(rid, row):
            if rid == 'first':
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('test release missing')
            return original(rid, row)
        with mock.patch.object(service, '_load_episode', side_effect=load):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(service.frame, 'first', 'cam_high', 8)
                try:
                    self.assertTrue(entered.wait(5))
                    second = pool.submit(service.frame, 'second', 'cam_high', 8)
                    self.assertEqual(second.result(timeout=2)['matched_video_frame'], 8)
                finally:
                    release.set()
                self.assertEqual(first.result(timeout=5)['matched_video_frame'], 8)

    def test_missing_or_corrupt_episode_does_not_break_startup_or_other_episodes(self):
        path = self.root / 'second/events.jsonl'
        path.write_text('{invalid json}\n')
        service = FailRecoveryTactileService(self.root)
        with self.assertRaises(KeyError):
            service.frame('unknown', 'cam_high', 8)
        with self.assertRaises(KeyError):
            service.frame('second', 'cam_high', 8)
        self.assertNotIn('second', service.episodes)
        self.assertEqual(service.frame('first', 'cam_high', 8)['matched_video_frame'], 8)
        path.write_text((self.root / 'first/events.jsonl').read_text())
        self.assertEqual(service.frame('second', 'cam_high', 8)['matched_video_frame'], 8)


if __name__ == '__main__':
    unittest.main()
