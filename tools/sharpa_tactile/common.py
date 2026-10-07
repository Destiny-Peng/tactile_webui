from __future__ import annotations
import hashlib
import os
import json
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_ROOT = CODE_ROOT / 'annotations/tactile_canonical/v1'
ROOT = CODE_ROOT
DATA_ROOT = Path(os.environ.get('TACTILE_DATA_ROOT', CODE_ROOT)).resolve()
linked_manifest = CODE_ROOT/'datasets/failrecovery/manifest.jsonl'
if 'TACTILE_DATA_ROOT' not in os.environ and linked_manifest.is_symlink():
    target = linked_manifest.resolve()
    if target.parent.name == 'v1' and target.parent.parent.name == 'lf3r_failure_rollouts':
        DATA_ROOT = target.parents[3]


def project_path(*parts):
    """Local outputs/code; linked LF3R raw data; relocated historical output paths."""
    path = Path(*parts)
    if path.is_absolute():
        if DATA_ROOT != ROOT:
            try:
                relative = path.relative_to(DATA_ROOT)
            except ValueError:
                return path
            if relative.parts and relative.parts[0] == 'outputs':
                return ROOT / relative
        return path
    local = ROOT / path
    if path.parts and path.parts[0] == 'outputs':
        return local
    if local.exists():
        return local
    # Only dataset and original annotation sources fall back to the linked tree.
    if path.parts and path.parts[0] in ('datasets', 'annotations'):
        original = DATA_ROOT / path
        if original.exists():
            return original
    return local


def relative_path(path):
    """Store local resources relative to this checkout, external raw sources absolute."""
    path = project_path(path)
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


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
    return [json.loads(line) for line in project_path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with project_path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binary_timeline(events, total):
    import numpy as np
    labels = np.full(total, -1, dtype=np.int64)
    conflict = np.zeros(total, dtype=bool)
    for event in events:
        key = int(event['event_key'])
        if canonical_key(event) not in (1, 2):
            continue
        start, end = int(event['start_frame']), int(event['end_frame'])
        if not 0 <= start <= end < total:
            raise ValueError(f'Invalid interval: {event}')
        value = binary_target(event)
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
        if canonical_key(event) not in (1, 2):
            continue
        start, end = int(event['start_frame']), int(event['end_frame'])
        if not 0 <= start <= end < total:
            raise ValueError(f'Invalid interval: {event}')
        value = canonical_key(event)
        region = labels[start:end + 1]
        if ((region != 0) & (region != value)).any():
            raise ValueError(f'Conflicting success/failure intervals: {event}')
        region[:] = value
    return labels


def canonical_key(event):
    """Never interpret legacy labels as targets; migrate the source first."""
    key = event['event_key']
    if type(key) is not int or key not in (1, 2, 3, 4):
        raise ValueError(f'Expected canonical label 1/2/3/4; run canonical_annotations: {key}')
    return key


def binary_target(event):
    key = canonical_key(event)
    if key not in (1, 2):
        raise ValueError('Labels 3/4 require their own task; cannot map to success/failure')
    return key - 1


def is_stage(event, stage):
    """Task stage comes from provenance, independently of canonical outcome."""
    canonical_key(event)
    return stage in event.get('stages', [])


def annotation_timeline(events, total):
    """Independent 1/2/3/4 channels; overlaps retain all positive labels."""
    import numpy as np
    target = np.zeros((total, 4), dtype=np.int64)
    for event in events:
        key = canonical_key(event)
        start, end = event['start_frame'], event['end_frame']
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < total:
            raise ValueError(f'Invalid interval: {event}')
        target[start:end+1, key-1] = 1
    return target
