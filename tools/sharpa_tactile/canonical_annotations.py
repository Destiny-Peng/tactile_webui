"""Canonical tactile intervals: strict labels, closed bounds, no lossy overlap merge."""
from __future__ import annotations
from .common import project_path
from .common import ANNOTATION_ROOT

from collections import Counter, defaultdict
import argparse
import datetime
import json
from pathlib import Path

from .common import ROOT, dump, read_jsonl, sha

LABELS = {1: 'success', 2: 'failure', 3: 'dropped_object', 4: 'wrong_object'}
LEGACY_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 2, 7: 2, 8: 1, 9: 1}
# This map applies ONLY to the original LF3R failure-type schema, not numeric canonical labels.
SOURCE_TYPES = {'timeout_no_progress': 6, 'control_error': 7,
                'observation_error': 8, 'other': 9, 'dropped_object': 3, 'wrong_object': 4}
DEFAULT_OUTPUT = ANNOTATION_ROOT
DEFAULT_MANIFEST = project_path('datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl')


def validate_event(event, record):
    key = event['event_key']
    if type(key) is not int or key not in LABELS:
        raise ValueError(f'Expected canonical event_key 1/2/3/4: {event}')
    start, end = event['start_frame'], event['end_frame']
    total = record['total_frames']
    if type(start) is not int or type(end) is not int or not 0 <= start <= end < total:
        raise ValueError(f'Invalid closed frame interval (total={total}): {event}')
    outcome = record.get('ground_truth_outcome')
    if outcome not in ('success', 'failure'):
        raise ValueError(f'Missing authoritative outcome: {record["id"]}')
    if key in (3, 4) and outcome != 'failure':
        raise ValueError(f'3/4 on non-failure rollout: {event}')


def merge_intervals(inputs, records):
    """Inputs are (raw event, source reference); dedup AFTER mapping, per rollout."""
    merged = {}; before = Counter(); mapped = Counter(); dropped = []; duplicates = []
    background = []; eligible = []
    for raw, source in inputs:
        rid = raw['rollout_id']; old = raw['event_key']
        if type(old) is not int or old not in LEGACY_MAPPING:
            raise ValueError(f'Unsupported source label: {raw}')
        if rid not in records:
            raise ValueError(f'Unknown rollout: {rid}')
        before[old] += 1
        key = LEGACY_MAPPING[old]
        provenance = dict(source, original_event_key=old,
                          original_start_frame=raw['start_frame'], original_end_frame=raw['end_frame'])
        if key == 0:
            background.append(dict(rollout_id=rid, provenance=provenance)); continue
        mapped[key] += 1
        event = dict(rollout_id=rid, event_key=key, label=LABELS[key],
                     start_frame=raw['start_frame'], end_frame=raw['end_frame'])
        if key in (3, 4) and records[rid].get('ground_truth_outcome') == 'success':
            dropped.append(dict(event, provenance=provenance)); continue
        validate_event(event, records[rid])
        identity = (rid, key, event['start_frame'], event['end_frame'])
        eligible.append(identity)
        if identity in merged:
            duplicates.append(dict(event, provenance=provenance))
            merged[identity]['provenance'].append(provenance)
        else:
            merged[identity] = dict(event, provenance=[provenance])
    events = sorted(merged.values(), key=lambda e: (e['rollout_id'], e['start_frame'], e['end_frame'], e['event_key']))
    for rid in records:
        for i, e in enumerate(e for e in events if e['rollout_id'] == rid):
            e['event_index'] = i
            e['source_record'] = f'annotations/tactile_canonical/v1/records/{rid}.json'
            # Align/Insert is provenance metadata, never an outcome label.
            types = {p.get('source_failure_type') for p in e['provenance']}
            stages = {'align' if t in ('timeout_no_progress', 'observation_error') else 'insert'
                      for t in types if t in ('timeout_no_progress', 'observation_error', 'control_error', 'other')}
            e['stages'] = sorted(stages)
    if set(eligible) != set(merged) or len(eligible) != len(events) + len(duplicates):
        raise ValueError('Merge lost nonduplicate intervals')
    for e in events:
        validate_event(e, records[e['rollout_id']])
        for p in e['provenance']:
            if (LEGACY_MAPPING[p['original_event_key']], p['original_start_frame'], p['original_end_frame']) != (e['event_key'], e['start_frame'], e['end_frame']):
                raise ValueError('Mapping or bounds changed')
    final = {str(k): dict(rollouts=len({e['rollout_id'] for e in events if e['event_key'] == k}),
                         intervals=sum(e['event_key'] == k for e in events)) for k in LABELS}
    summary = dict(before_label_intervals={str(k): before[k] for k in LEGACY_MAPPING},
                   after_mapping_before_filter={str(k): mapped[k] for k in LABELS},
                   discarded_success_34=len(dropped), discarded_success_34_by_label={str(k): sum(e['event_key']==k for e in dropped) for k in (3,4)},
                   background_records_ignored=len(background), exact_duplicates=len(duplicates), final=final,
                   input_intervals=sum(before.values()), output_intervals=len(events),
                   annotated_rollouts=len({e['rollout_id'] for e in events}))
    return events, summary, dict(discarded_success_34=dropped, exact_duplicates=duplicates, background=background)


def migrate(args):
    manifest = args.manifest.resolve(); source_dir = args.annotations.resolve(); relabel = args.relabel.resolve()
    records = {r['id']: r for r in read_jsonl(manifest) if 'usb' in r['task_key']}
    files = {}; inputs = []; excluded = []; counts = defaultdict(Counter)
    def register(path):
        path = path.resolve(); files[str(path)] = sha(path); return path
    register(manifest)
    def add_annotation(path, expected_sha=None):
        path = register(path)
        if expected_sha and files[str(path)] != expected_sha:
            raise ValueError(f'Snapshot hash mismatch: {path}')
        annotation = json.loads(path.read_text()); rid = annotation['rollout_id']
        if rid not in records: return
        for index, event in enumerate(annotation.get('failure_events') or [annotation]):
            typ = event.get('failure_type'); start = event.get('causal_onset_frame'); end = event.get('observable_onset_frame')
            if typ not in SOURCE_TYPES:
                excluded.append(dict(source=str(path), rollout_id=rid, event_index=index, source_failure_type=typ,
                                     reason='outside historical tactile schema (not canonical numeric labels)')); continue
            if start is None and end is None:
                excluded.append(dict(source=str(path), rollout_id=rid, event_index=index, source_failure_type=typ, reason='unannotated placeholder: both bounds null')); continue
            if start is None or end is None:
                raise ValueError(f'Incomplete interval: {path}/{index}')
            key = SOURCE_TYPES[typ]
            inputs.append((dict(rollout_id=rid, event_key=key, start_frame=start, end_frame=end),
                           dict(source=str(path), sha256=files[str(path)], source_event_index=index, source_failure_type=typ)))
            counts[str(path.parent)][key] += 1
    # The old batch was overwritten in live records. Restore it from its verified native backup.
    backup = args.legacy_backup.resolve(); meta = register(backup/'backup_manifest.json')
    for row in json.loads(meta.read_text())['records']:
        add_annotation(backup/row['backup_record'], row['sha256'])
    for rid in sorted(records):
        path = source_dir/f'{rid}.json'
        if not path.exists(): raise ValueError(f'Missing live annotation: {path}')
        add_annotation(path)
    runs = sorted(relabel.glob('*/source_intervals.json'))
    if not runs: raise ValueError(f'No relabel snapshots: {relabel}')
    for path in runs:
        register(path); snapshot = register(path.parent/'annotation_snapshot.json')
        refs = {r['rollout_id']: r for r in json.loads(snapshot.read_text())['records']}
        for row in refs.values(): add_annotation(path.parent/row['backup'], row['sha256'])
        for index, event in enumerate(json.loads(path.read_text())):
            rid = event['rollout_id']
            if rid not in records: raise ValueError(f'Unknown snapshot rollout: {rid}')
            # Verify derived export against its native source, including its precise bounds.
            ref = refs[rid]; original = json.loads((path.parent/ref['backup']).read_text())['failure_events'][event['event_index']]
            if (SOURCE_TYPES[original['failure_type']], original['causal_onset_frame'], original['observable_onset_frame']) != (event['event_key'], event['start_frame'], event['end_frame']):
                raise ValueError(f'Export/source mismatch: {path}/{index}')
            inputs.append((event, dict(source=str(path.resolve()), sha256=files[str(path.resolve())], source_event_index=index,
                                       source_failure_type=original['failure_type'], native_source=str((path.parent/ref['backup']).resolve()))))
            counts[str(path)][event['event_key']] += 1
    events, summary, audit = merge_intervals(inputs, records)
    summary.update(scope='All 125 USB rollouts in authoritative failrecovery manifest; union of all snapshots; no latest-wins',
                   manifest_outcomes=dict(Counter(r['ground_truth_outcome'] for r in records.values())),
                   per_source_before={p: {str(k):v for k,v in c.items()} for p,c in counts.items()},
                   excluded_non_tactile_source_events=len(excluded), validation='PASS',
                   created_at=datetime.datetime.now().astimezone().isoformat())
    # Check immutable sources before creating any deliverable.
    for path, digest in files.items():
        if sha(path) != digest: raise ValueError(f'Source changed during migration: {path}')
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    for e in events: e['source_record'] = str(output/'records'/f'{e["rollout_id"]}.json')
    by_id = defaultdict(list)
    for e in events: by_id[e['rollout_id']].append(e)
    for rid, record in records.items():
        dump(output/'records'/f'{rid}.json', dict(schema_version='tactile_canonical_v1', rollout_id=rid,
             ground_truth_outcome=record['ground_truth_outcome'], outcome_provenance=dict(source=str(manifest), sha256=files[str(manifest)],
             field='ground_truth_outcome', outcome_source=record.get('outcome_source')), total_frames=record['total_frames'], intervals=by_id[rid]))
    (output/'intervals.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in events))
    dump(output/'migration_summary.json', summary); dump(output/'migration_audit.json', dict(audit, excluded_source_events=excluded))
    dump(output/'provenance.json', dict(schema_version='tactile_canonical_v1', label_mapping=LEGACY_MAPPING,
         labels=LABELS, sources=files, outcome_source=str(manifest), outputs={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    validate_output(output, manifest)
    print(json.dumps(summary, indent=2))


def validate_output(output, manifest):
    records = {r['id']:r for r in read_jsonl(manifest)}
    provenance = json.loads((output/'provenance.json').read_text())
    for path, digest in provenance['sources'].items():
        if sha(path) != digest: raise ValueError(f'Input hash changed: {path}')
    for path, digest in provenance['outputs'].items():
        if sha(output/path) != digest: raise ValueError(f'Output hash changed: {path}')
    events = read_jsonl(output/'intervals.jsonl'); seen = set(); by_id=defaultdict(list)
    for e in events:
        validate_event(e, records[e['rollout_id']]); by_id[e['rollout_id']].append(e)
        identity = (e['rollout_id'],e['event_key'],e['start_frame'],e['end_frame'])
        if identity in seen: raise ValueError(f'Duplicate output: {identity}')
        seen.add(identity)
        if not e.get('provenance'): raise ValueError('Missing provenance')
        for p in e['provenance']:
            if (LEGACY_MAPPING[p['original_event_key']],p['original_start_frame'],p['original_end_frame']) != identity[1:]:
                raise ValueError('Mapping/boundary violation')
    for path in (output/'records').glob('*.json'):
        row=json.loads(path.read_text())
        if row['intervals'] != by_id[row['rollout_id']]: raise ValueError(f'Flat/per-rollout mismatch: {path}')
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=project_path('annotations/failure_annotations/v1/records'))
    parser.add_argument('--relabel', type=Path, default=project_path('outputs/sharpa_failure_relabel'))
    parser.add_argument('--legacy-backup', type=Path, default=project_path('outputs/usb_event_intervals/20261003_202352/original_annotations_backup_20261003_203655'))
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        events=validate_output(args.output, args.manifest); print(f'PASS: {len(events)} canonical intervals; source/output hashes unchanged')
    else: migrate(args)


if __name__ == '__main__': main()
