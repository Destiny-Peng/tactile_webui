from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .common import ROOT, FINGERS, read_jsonl, binary_timeline, three_class_timeline


def load_sources(interval_path, manifest_path):
    events = read_jsonl(interval_path)
    grouped = defaultdict(list)
    for event in events:
        if int(event['event_key']) in (6, 7, 8, 9):
            grouped[event['rollout_id']].append(event)
    manifest = {r['id']: r for r in read_jsonl(manifest_path)}
    missing = set(grouped) - manifest.keys()
    if missing:
        raise ValueError(f'Missing USB episodes: {sorted(missing)}')
    return {rid: (manifest[rid], grouped[rid]) for rid in sorted(grouped)}


def episode_arrays(record, intervals, camera='cam_high', num_classes=2):
    """Recorded tick/event join: never nearest-neighbor or future-fill tactile data."""
    frames = read_jsonl(ROOT / record['synchronized_frames_path'])
    event_rows = read_jsonl(ROOT / record['tactile_events_path'])
    events = {(r['finger'], int(r['event_id'])): r for r in event_rows}
    if num_classes == 3:
        labels = three_class_timeline(intervals, record['total_frames'])
        conflicts = np.zeros(len(labels), dtype=bool)
    else:
        labels, conflicts = binary_timeline(intervals, record['total_frames'])
    n = len(frames)
    f6 = np.zeros((n, 5, 6), dtype=np.float32)
    valid = np.ones(n, dtype=bool)
    camera_frames = np.full(n, -1, dtype=np.int64)
    targets = np.full(n, 0 if num_classes == 3 else -1, dtype=np.int64)
    references = []
    for i, row in enumerate(frames):
        camera_value = row.get('camera_frame_indices', {}).get(camera)
        if camera_value is not None:
            camera_frames[i] = int(camera_value)
            if 0 <= int(camera_value) < len(labels):
                targets[i] = labels[int(camera_value)]
            else:
                valid[i] = False
        else:
            valid[i] = False
        refs = []
        for j, finger in enumerate(FINGERS):
            sync = row.get('tactile', {}).get(finger, {})
            event = events.get((finger, sync.get('event_id')))
            good = bool(event and sync.get('valid') and not sync.get('stale') and event.get('valid'))
            if good:
                values = np.asarray(event.get('f6'), dtype=np.float32)
                good = (values.shape == (6,) and np.isfinite(values).all()
                        and int(event['receive_mono_ns']) <= int(row['tick_mono_ns'])
                        and event.get('deform_shape') == [240, 240]
                        and event.get('deform_length_bytes') == 240*240
                        and event.get('deform_offset_bytes', -1) >= 0)
            if good:
                f6[i, j] = values
            else:
                valid[i] = False
            refs.append(event)
        references.append(refs)
    # A prediction at t uses exactly t-15 ... t. Require the whole window valid.
    usable = np.zeros(n, dtype=bool)
    if n >= 16:
        usable[15:] = np.convolve(valid.astype(np.int32), np.ones(16, dtype=np.int32), 'valid') == 16
    supervised = np.flatnonzero((targets >= 0) & usable)
    if not len(supervised):
        raise ValueError(f'No valid labelled samples: {record["id"]}')
    # Binary retains past unlabeled context through the last interval.
    # Three-class supervises the complete usable rollout, including background suffixes.
    endpoints = np.flatnonzero(usable) if num_classes == 3 else np.flatnonzero(usable & (np.arange(n) <= supervised[-1]))
    gaps = np.r_[True, np.diff(endpoints) != 1]
    return {'f6': f6, 'valid': valid, 'endpoints': endpoints, 'labels': targets[endpoints],
            'video_frames': camera_frames[endpoints], 'ticks': endpoints,
            'segment_starts': gaps, 'references': references,
            'audit': {'sync_rows': n, 'valid_ticks': int(valid.sum()),
                      'usable_feature_rows': len(endpoints), 'failure_rows': int((targets[endpoints] == (2 if num_classes == 3 else 1)).sum()),
                      'success_rows': int((targets[endpoints] == (1 if num_classes == 3 else 0)).sum()),
                      'background_rows': int((targets[endpoints] == 0).sum()) if num_classes == 3 else 0,
                      'conflicting_camera_frames': int(conflicts.sum()),
                      'discarded_labelled_ticks': int(((targets >= 0) & ~usable).sum())}}


class DeformStreams:
    def __init__(self, record):
        self.maps = [np.memmap(ROOT / record['tactile_stream_paths'][finger]['deform'],
                               mode='r', dtype=np.uint8) for finger in FINGERS]

    def batch(self, references):
        result = np.empty((len(references), 5, 1, 240, 240), dtype=np.float32)
        for i, refs in enumerate(references):
            for j, event in enumerate(refs):
                start = int(event['deform_offset_bytes'])
                stop = start + 240*240
                if stop > len(self.maps[j]):
                    raise ValueError('Short deform stream read')
                result[i, j, 0] = self.maps[j][start:stop].reshape(240, 240)
        return result


def rollout_split(records, labels, seed=42):
    """Stratify by task and available binary classes; each rollout stays in one set."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for rid in records:
        y = labels[rid]
        key = (records[rid]['task_key'], bool((y == 1).any()), bool((y == 0).any()))
        groups[key].append(rid)
    splits = {'train': [], 'val': [], 'test': []}
    strata = {}
    for key, ids in sorted(groups.items()):
        ids = list(sorted(ids)); rng.shuffle(ids); n = len(ids)
        holdout = max(1, int(round(n * 0.15))) if n >= 3 else 0
        holdout = min(holdout, (n-1)//2)
        splits['test'].extend(ids[:holdout])
        splits['val'].extend(ids[holdout:2*holdout])
        splits['train'].extend(ids[2*holdout:])
        strata[str(key)] = n
    sets = [set(splits[name]) for name in ('train', 'val', 'test')]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(records)
    for name, ids in splits.items():
        if not ids or set(np.concatenate([labels[rid][labels[rid]>=0] for rid in ids])) != {0, 1}:
            raise ValueError(f'{name} does not contain both binary classes')
    return {'seed': seed, 'unit': 'rollout', 'strata': strata, **{k: sorted(v) for k, v in splits.items()}}
