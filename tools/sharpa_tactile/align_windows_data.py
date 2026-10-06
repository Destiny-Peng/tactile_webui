"""Align-only fixed-length windows; hard Key labels and reproducible jitter."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from .common import ROOT, dump, sha

CLASSES = ('in_progress', 'success', 'failure')
STEPS = (1, 3, 5, 8, 12)
JITTERS = (0, 2, 5, 10)
WINDOW = 16


def configs():
    result = []
    for steps in [(s,) for s in STEPS] + [STEPS]:
        for jitter in JITTERS:
            result.append(dict(steps=list(steps), jitter=jitter, membership='span',
                               context='align_local', padding='edge', weight_mode='inverse_frequency', phase='primary'))
    # All combinations of the three unresolved choices at one fixed anchor.
    for membership in ('span', 'sampled'):
        for context in ('align_local', 'past_context'):
            for padding in ('edge', 'drop'):
                if (membership, context, padding) == ('span', 'align_local', 'edge'):
                    continue
                result.append(dict(steps=[3], jitter=5, membership=membership, context=context,
                                   padding=padding, weight_mode='inverse_frequency', phase='interpretation'))
    for weight in ('unweighted', 'sqrt_inverse'):
        result.append(dict(steps=[3], jitter=5, membership='span', context='align_local',
                           padding='edge', weight_mode=weight, phase='weight'))
    for c in result:
        steps = 'mix' if len(c['steps']) > 1 else str(c['steps'][0])
        c['id'] = f"step{steps}_n{c['jitter']}_{c['membership']}_{c['context']}_{c['padding']}_{c['weight_mode']}"
    assert len(result) == len({c['id'] for c in result})
    return result


def labels_for(frames, keys, outcomes, membership):
    frames = np.asarray(frames); keys = np.asarray(keys)
    if membership == 'span':
        contains = (frames[:, 0] <= keys) & (keys <= frames[:, -1])
    elif membership == 'sampled':
        contains = (frames == keys[:, None]).any(1)
    else:
        raise ValueError(membership)
    return np.where(contains, outcomes, 0).astype(np.int64)


def make_windows(events, config, split):
    indices = []; metadata = []
    for event in events:
        if event['split'] != split:
            continue
        frames = event['video_frames']; ticks = event['ticks']
        for step in (config['steps'] if split == 'train' else STEPS):
            lower = event['start_frame'] if config['context'] == 'align_local' else event['context_start_frame']
            allowed = np.flatnonzero((frames >= lower) & (frames <= event['sampling_end_frame']))
            if not len(allowed):
                continue
            # Never bridge an invalid synchronization gap or an excluded Insert segment.
            cuts = np.r_[0, np.flatnonzero((np.diff(frames[allowed]) != 1) | (np.diff(ticks[allowed]) != 1))+1, len(allowed)]
            for left, right in zip(cuts[:-1], cuts[1:]):
                segment = allowed[left:right]
                first_frame = int(frames[segment[0]])
                mapping = {int(frames[p]):int(p) for p in segment}
                for position in segment:
                    endpoint = int(frames[position])
                    if endpoint < event['start_frame']:
                        continue
                    desired = endpoint - np.arange(WINDOW-1, -1, -1)*step
                    if config['padding'] == 'drop' and desired[0] < first_frame:
                        continue
                    actual = np.maximum(desired, first_frame)
                    local = np.array([mapping[int(f)] for f in actual], dtype=np.int64)
                    indices.append(local + event['feature_offset'])
                    metadata.append(dict(event_id=event['event_id'], rollout_id=event['rollout_id'],
                        event_key=event['event_key'], event_index=event['event_index'], outcome=event['outcome'],
                        key=event['end_frame'], step=step, window_start_frame=int(actual[0]),
                        window_end_frame=endpoint, padded_frames=int((desired < first_frame).sum()),
                        sample_frames=actual.tolist(), sample_ticks=ticks[local].tolist(), split=split))
    index = np.asarray(indices, dtype=np.int64).reshape(-1, WINDOW)
    return index, metadata


def expanded_counts(metadata, config):
    frames = np.array([r['sample_frames'] for r in metadata], dtype=np.int64).reshape(-1, WINDOW)
    keys = np.array([r['key'] for r in metadata]); outcomes = np.array([r['outcome'] for r in metadata])
    counts = np.zeros(3, dtype=np.int64)
    for delta in range(-config['jitter'], config['jitter']+1):
        counts += np.bincount(labels_for(frames, keys+delta, outcomes, config['membership']), minlength=3)
    return counts


def load_events(output):
    manifest = json.loads((output/'align_manifest.json').read_text()); events=[]
    f6=[]; deform=[]; offset=0
    for row in manifest['events']:
        with np.load(output/row['feature_path']) as cache:
            event={**row, 'video_frames':cache['video_frames'].copy(), 'ticks':cache['ticks'].copy(), 'feature_offset':offset}
            f6.append(cache['f6'].copy()); deform.append(cache['deform'].copy())
            offset += len(cache['ticks']); events.append(event)
    return manifest, events, np.concatenate(f6), np.concatenate(deform)


def prepare(source, interval_source, output):
    output.mkdir(parents=True, exist_ok=True); (output/'features').mkdir(exist_ok=True)
    data=json.loads((source/'data_manifest.json').read_text()); split=json.loads((source/'split_manifest.json').read_text())
    old=json.loads((interval_source/'interval_manifest.json').read_text())
    assert data['status'] == old['status'] == 'complete'
    assert sha(source/'split_manifest.json') == sha(interval_source/'split_manifest.json')
    shutil.copyfile(source/'split_manifest.json', output/'split_manifest.json')
    membership={rid:s for s in ('train','val','test') for rid in split[s]}
    oldrows={(r['rollout_id'],r['event_index']):r for r in old['records']}
    selected=[e for e in data['source_intervals'] if int(e['event_key']) in (6,8)]
    rows=[]; audits=[]; source_hashes={}; maxspan=15*max(STEPS)
    for number, event in enumerate(selected):
        rid=event['rollout_id']; start=int(event['start_frame']); end=int(event['end_frame'])
        record=data['records'][rid]; original=data['signature']['episode_sources'][rid]
        assert sha(ROOT/record['synchronized_frames_path']) == original['frames_sha256']
        assert sha(ROOT/record['tactile_events_path']) == original['events_sha256']
        relatives=[e for e in data['source_intervals'] if e['rollout_id'] == rid]
        inserts=[e for e in relatives if int(e['event_key']) in (7,9)]
        # Neighbouring Align events cannot enter this event's context or shifted-Key suffix.
        previous=[int(e['end_frame']) for e in relatives if int(e['event_key']) in (6,8) and int(e['end_frame']) < start]
        following=[int(e['start_frame']) for e in relatives if int(e['event_key']) in (6,8) and int(e['start_frame']) > start]
        context_start=max(0, start-maxspan, max(previous)+1 if previous else 0)
        sampling_end=min(record['total_frames']-1, end+max(JITTERS), min(following)-1 if following else record['total_frames']-1)
        cache_path=source/'features'/(rid+'.npz'); source_hashes[str(cache_path.relative_to(ROOT))]=sha(cache_path)
        with np.load(cache_path) as cached:
            frames=cached['video_frames']; ticks=cached['ticks']
            allowed=(frames >= context_start)&(frames <= sampling_end)
            insert_mask=np.zeros(len(frames),bool)
            for insert in inserts:
                insert_mask |= (frames >= insert['start_frame'])&(frames <= insert['end_frame'])
            removed=int((allowed&insert_mask).sum()); allowed &= ~insert_mask
            # Past-context F6 uses a dense history; exclude any features whose encoder history sees Insert.
            clean_history=np.ones(len(frames),bool)
            for insert in inserts:
                clean_history &= ~((frames >= insert['start_frame'])&(frames-15 <= insert['end_frame']))
            allowed &= clean_history
            pos=np.flatnonzero(allowed); chosen_frames=frames[pos]; chosen_ticks=ticks[pos]
            f6=cached['f6'][pos].copy(); deform=cached['deform'][pos].copy()
        prior=oldrows[(rid,event['event_index'])]; prefix_path=interval_source/prior['feature_path']
        source_hashes[str(prefix_path.relative_to(ROOT))]=sha(prefix_path)
        # A separate align-local F6 cache, using existing in-interval edge-padded prefixes.
        f6_local=f6.copy()
        with np.load(prefix_path) as local:
            lookup={int(t):i for i,t in enumerate(local['ticks'])}
            for i,t in enumerate(chosen_ticks):
                if int(t) in lookup:
                    f6_local[i]=local['f6'][lookup[int(t)]]
        target=output/'features'/f'align_{number:04d}.npz'
        np.savez_compressed(target, f6=f6, f6_local=f6_local, deform=deform,
                            ticks=chosen_ticks, video_frames=chosen_frames)
        row=dict(event_id=f'align_{number:04d}', rollout_id=rid, event_index=event['event_index'], event_key=int(event['event_key']),
            outcome=1 if int(event['event_key']) == 8 else 2, start_frame=start, end_frame=end,
            context_start_frame=context_start, sampling_end_frame=sampling_end, split=membership[rid],
            feature_path=str(target.relative_to(output)), source_annotation=event['source_record'], feature_rows=len(pos))
        rows.append(row); audits.append(dict(event_id=row['event_id'], removed_insert_feature_rows=removed,
            local_rows=int(((chosen_frames >= start)&(chosen_frames <= end)).sum()),
            key_feature_available=bool((chosen_frames == end).any()), key_in_insert=any(e['start_frame']<=end<=e['end_frame'] for e in inserts)))
        print('ALIGN_CACHE',number+1,len(selected),row['event_id'],len(pos),flush=True)
    signature=dict(source=str(source.relative_to(ROOT)), interval_source=str(interval_source.relative_to(ROOT)),
        source_manifest_sha256=sha(source/'data_manifest.json'), interval_manifest_sha256=sha(interval_source/'interval_manifest.json'),
        split_sha256=sha(source/'split_manifest.json'), encoder_sha256={k:data['signature'][k+'_sha256'] for k in ('f6','deform')},
        annotation_sha256=sha(ROOT/data['intervals']), source_hashes=source_hashes,
        class_mapping=dict(enumerate(CLASSES)), window=WINDOW, steps=list(STEPS), jitters=list(JITTERS),
        time_unit='cam_high frame; sampled points require exact synchronized camera frame; endpoint stride 1',
        label='one hard class per window; shifted Key=start-independent original Align end + integer delta',
        insert='events 7/9 excluded; their annotated frames and dense F6 histories intersecting Insert excluded',
        f6_context='align_local uses original interval-only prefix; past_context uses dense causal rollout prefix',
        suffix='up to original Key+10, clipped before next Align, excluding Insert; fixed across configs',
        invalid='no windows crossing missing camera frames, tactile ticks or removed Insert segments',
        evaluation='original unshifted Key, common evaluation bank with all five steps, no subsampling or label augmentation')
    counts={s:{'intervals':sum(r['split']==s for r in rows), 'success':sum(r['split']==s and r['outcome']==1 for r in rows),
               'failure':sum(r['split']==s and r['outcome']==2 for r in rows)} for s in ('train','val','test')}
    assert len(rows)==len(selected)
    dump(output/'align_manifest.json',dict(status='complete',signature=signature,events=rows,audits=audits,split_counts=counts))
    generate_grid(output)


def generate_grid(output):
    manifest, events, _, _=load_events(output)
    plan=configs(); summaries=[]
    for config in plan:
        split_rows={}; valid=True
        for s in ('train','val','test'):
            index,meta=make_windows(events,config,s)
            counts=expanded_counts(meta,config if s=='train' else {**config,'jitter':0})
            split_rows[s]=dict(windows=len(meta), class_counts=counts.tolist(),
                events=len({r['event_id'] for r in meta}), padded_windows=sum(r['padded_frames']>0 for r in meta))
            valid &= bool((counts>0).all())
            directory=output/'datasets'/config['id']; directory.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(directory/(s+'.npz'),indices=index,
                frames=np.array([r['sample_frames'] for r in meta],dtype=np.int64).reshape(-1,WINDOW),
                keys=np.array([r['key'] for r in meta],dtype=np.int64), outcomes=np.array([r['outcome'] for r in meta],dtype=np.int64))
            dump(directory/(s+'_metadata.json'),meta)
        summaries.append(dict(config=config,splits=split_rows,eligible=valid,
                              reason=None if valid else 'at least one class absent: do not train a misleading three-class comparison'))
    dump(output/'dataset_grid.json',summaries)
    print('PREPARE_ALIGN_COMPLETE',json.dumps(manifest['split_counts']), 'eligible',sum(r['eligible'] for r in summaries),'/',len(plan),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--source',type=Path,required=True)
    p.add_argument('--interval-source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--grid-only', action='store_true')
    a=p.parse_args()
    if a.grid_only: generate_grid(a.output.resolve())
    else: prepare(a.source.resolve(),a.interval_source.resolve(),a.output.resolve())


if __name__=='__main__': main()
