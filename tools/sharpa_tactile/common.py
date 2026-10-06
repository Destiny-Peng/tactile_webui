from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for repo in ('T-Rex', 'sharpawave-deform-encoder'):
    sys.path.insert(0, str(ROOT / 'repos' / repo))
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
INPUTS = ('f6', 'deform', 'f6_deform')
HEADS = ('mlp', 'lstm')


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binary_timeline(events, total):
    import numpy as np
    labels = np.full(total, -1, dtype=np.int64)
    conflict = np.zeros(total, dtype=bool)
    for event in events:
        key = int(event['event_key'])
        if key not in (6, 7, 8, 9):
            continue
        start, end = int(event['start_frame']), int(event['end_frame'])
        if not 0 <= start <= end < total:
            raise ValueError(f'Invalid interval: {event}')
        value = int(key in (6, 7))
        conflict[start:end+1] |= (labels[start:end+1] >= 0) & (labels[start:end+1] != value)
        labels[start:end+1] = value
    labels[conflict] = -1
    return labels, conflict


def three_class_timeline(events, total):
    """Closed hard-label intervals; background elsewhere. Reject ambiguous overlaps."""
    import numpy as np
    labels = np.zeros(total, dtype=np.int64)
    for event in events:
        key = int(event['event_key'])
        if key not in (6, 7, 8, 9):
            continue
        start, end = int(event['start_frame']), int(event['end_frame'])
        if not 0 <= start <= end < total:
            raise ValueError(f'Invalid interval: {event}')
        value = 2 if key in (6, 7) else 1
        region = labels[start:end + 1]
        if ((region != 0) & (region != value)).any():
            raise ValueError(f'Conflicting success/failure intervals: {event}')
        region[:] = value
    return labels
