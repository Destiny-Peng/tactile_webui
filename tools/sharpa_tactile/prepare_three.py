"""Cache complete rollouts with hard background/success/failure labels and unchanged splits."""
from pathlib import Path
import argparse
import datetime
import json
import shutil
import time
import numpy as np
import torch
from .common import ROOT, FINGERS, dump, sha, three_class_timeline
from .data import load_sources, episode_arrays, DeformStreams
from .models import FrozenEncoders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args(); args.output = args.output.resolve(); args.previous = args.previous.resolve(); torch.set_num_threads(2)
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    (out/'features').mkdir(exist_ok=True); (out/'frame_labels').mkdir(exist_ok=True)
    old = json.loads((args.previous/'data_manifest.json').read_text())
    intervals = ROOT/old['intervals']; manifest = ROOT/old['manifest']
    assert sha(intervals) == old['signature']['interval_sha256']
    assert sha(manifest) == old['signature']['manifest_sha256']
    split = json.loads((args.previous/'split_manifest.json').read_text())
    sources = load_sources(intervals, manifest)
    ids = [rid for name in ('train','val','test') for rid in split[name]]
    assert len(ids) == len(set(ids)) and set(ids) == set(sources)
    shutil.copyfile(args.previous/'split_manifest.json', out/'split_manifest.json')
    # Reuse the already executed sanity check, without rerunning reconstruction.
    for name in ('sanity.json','reconstruction_f6.npz','reconstruction_f6.png'):
        shutil.copyfile(args.previous/name, out/name)
    encoders = FrozenEncoders().to(args.device)
    assert sha(encoders.f6_path) == old['signature']['f6_sha256']
    assert sha(encoders.deform_path) == old['signature']['deform_sha256']
    signature = {**old['signature'], 'class_mapping':{'0':'background','1':'success (8/9)','2':'failure (6/7)'},
        'num_classes':3,'coverage':'complete usable rollouts, including post-interval suffix',
        'unlabelled':'background=0; no ignore or smoothing',
        'previous_output':str(args.previous.relative_to(ROOT)),
        'split_sha256':sha(args.previous/'split_manifest.json')}
    signature.pop('binary', None)
    existing = out/'data_manifest.json'
    if existing.exists() and json.loads(existing.read_text())['signature'] != signature:
        raise ValueError('Changed cache signature; use a new output directory')
    dump(existing, {'signature':signature,'status':'preparing'})
    audit = {}; started = time.monotonic()
    for i,(rid,(record,events)) in enumerate(sources.items()):
        # Audit source identities before reusing encoder features.
        src = old['signature']['episode_sources'][rid]
        assert sha(ROOT/record['synchronized_frames_path']) == src['frames_sha256']
        assert sha(ROOT/record['tactile_events_path']) == src['events_sha256']
        for finger in FINGERS:
            stat = (ROOT/record['tactile_stream_paths'][finger]['deform']).stat()
            assert stat.st_size == src['deform_streams'][finger]['size']
            assert stat.st_mtime_ns == src['deform_streams'][finger]['mtime_ns']
        arr = episode_arrays(record, events, signature['camera'], num_classes=3)
        audit[rid] = arr['audit']; ticks = arr['endpoints']
        np.save(out/'frame_labels'/(rid+'.npy'), three_class_timeline(events, record['total_frames']))
        target = out/'features'/(rid+'.npz')
        if target.exists():
            print(f'CACHE {i+1}/{len(sources)} reused {rid}', flush=True); continue
        f6 = np.empty((len(ticks),1280),np.float32); deform = np.empty((len(ticks),2560),np.float32)
        with np.load(args.previous/'features'/(rid+'.npz')) as cached:
            positions = np.searchsorted(ticks,cached['ticks'])
            assert np.array_equal(ticks[positions],cached['ticks'])
            f6[positions] = cached['f6']; deform[positions] = cached['deform']
        missing = np.ones(len(ticks),bool); missing[positions] = False
        missing_positions = np.flatnonzero(missing); streams = DeformStreams(record)
        for start in range(0,len(missing_positions),args.batch_size):
            pos = missing_positions[start:start+args.batch_size]; endpoints = ticks[pos]
            raw = np.stack([arr['f6'][t-15:t+1] for t in endpoints])
            images = streams.batch([arr['references'][t] for t in endpoints])
            f6[pos] = encoders.f6_features(torch.from_numpy(raw).to(args.device)).cpu().numpy()
            deform[pos] = encoders.deform_features(torch.from_numpy(images).to(args.device)).cpu().numpy()
        assert np.isfinite(f6).all() and np.isfinite(deform).all()
        assert set(np.unique(arr['labels'])).issubset({0,1,2})
        temp = target.with_suffix('.tmp.npz')
        np.savez_compressed(temp,f6=f6,deform=deform,labels=arr['labels'],video_frames=arr['video_frames'],
                            ticks=arr['ticks'],segment_starts=arr['segment_starts'])
        temp.replace(target)
        print(f'CACHE {i+1}/{len(sources)} rows={len(ticks)} new={len(missing_positions)} seconds={time.monotonic()-started:.1f} {rid}',flush=True)
    summary = {name:{'rollouts':len(split[name]), **{key:sum(audit[rid][key] for rid in split[name])
              for key in ('background_rows','success_rows','failure_rows','usable_feature_rows')}} for name in ('train','val','test')}
    dump(existing, {'signature':signature,'status':'complete','created_at':datetime.datetime.now().astimezone().isoformat(),
         'intervals':old['intervals'],'manifest':old['manifest'],'source_intervals':old['source_intervals'],
         'records':old['records'],'audit':audit,'split_counts':summary})
    print('PREPARE_THREE_COMPLETE', json.dumps(summary),flush=True)


if __name__ == '__main__': main()
