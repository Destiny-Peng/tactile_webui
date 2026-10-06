"""Annotated intervals become independent binary samples, with no background loss."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .common import ROOT, dump, sha
from .data import episode_arrays
from .models import FrozenEncoders


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1'); parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--f6-context',choices=['interval_only','past_rollout'],default='interval_only')
    args=parser.parse_args(); src=args.source.resolve(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); (out/'features').mkdir(exist_ok=True)
    data=json.loads((src/'data_manifest.json').read_text()); split=json.loads((src/'split_manifest.json').read_text())
    assert data['status']=='complete'
    membership={rid:name for name in ('train','val','test') for rid in split[name]}
    assert len(membership)==sum(len(split[n]) for n in ('train','val','test'))
    shutil.copyfile(src/'split_manifest.json',out/'split_manifest.json')
    for name in ('sanity.json','reconstruction_f6.npz','reconstruction_f6.png'):
        shutil.copyfile(src/name,out/name)
    encoders=FrozenEncoders().to(args.device)
    assert sha(encoders.f6_path)==data['signature']['f6_sha256']
    assert sha(encoders.deform_path)==data['signature']['deform_sha256']
    signature={'source':str(src.relative_to(ROOT)),'source_manifest_sha256':sha(src/'data_manifest.json'),
        'split_sha256':sha(src/'split_manifest.json'),'f6_sha256':sha(encoders.f6_path),'deform_sha256':sha(encoders.deform_path),
        'label_mapping':{'0':'success (8/9)','1':'failure (6/7)'},'unit':'annotated interval',
        'bounds':'closed [start_frame,end_frame]; original causal/observable fields are boundaries only',
        'f6_context':args.f6_context,'f6_padding':'repeat first valid in-interval tick on left' if args.f6_context=='interval_only' else 'original past rollout cache',
        'background':'no samples, no supervision','window':16,'class_weight_unit':'interval count',
        'encoder_feature':'continuous F6 1280D; deform pooled 2x2 2560D; float32 TF32 disabled'}
    manifest=out/'interval_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature']!=signature: raise ValueError('Changed settings; use new output')
    dump(manifest,{'signature':signature,'status':'preparing'})
    events_by_rollout={rid:[] for rid in membership}
    for e in data['source_intervals']:
        if int(e['event_key']) in (6,7,8,9): events_by_rollout[e['rollout_id']].append(e)
    rows=[]
    for rid,events in sorted(events_by_rollout.items()):
        record=data['records'][rid]
        origin=data['signature']['episode_sources'][rid]
        assert sha(ROOT/record['synchronized_frames_path'])==origin['frames_sha256']
        assert sha(ROOT/record['tactile_events_path'])==origin['events_sha256']
        arrays=episode_arrays(record,events,data['signature']['camera'],num_classes=3) if args.f6_context=='interval_only' else None
        with np.load(src/'features'/(rid+'.npz')) as cached:
            video=cached['video_frames']; ticks=cached['ticks']; f6=cached['f6']; deform=cached['deform']
            assert (np.diff(video)>=0).all()
            for e in events:
                index=len(rows); target=out/'features'/f'interval_{index:04d}.npz'
                selected=np.flatnonzero((video>=e['start_frame'])&(video<=e['end_frame']))
                if not len(selected): raise ValueError(f'Interval lacks valid synchronized features: {e}')
                selected_ticks=ticks[selected]; first=int(selected_ticks[0])
                prefix=np.flatnonzero(selected_ticks-first<15) if args.f6_context=='interval_only' else np.array([],dtype=int)
                label=int(int(e['event_key']) in (6,7))
                row={'interval_id':f'interval_{index:04d}','rollout_id':rid,'event_index':e['event_index'],'event_key':int(e['event_key']),
                    'start_frame':e['start_frame'],'end_frame':e['end_frame'],'label':label,'split':membership[rid],
                    'ticks':len(selected),'first_tick':first,'last_tick':int(selected_ticks[-1]),
                    'gaps':int((np.diff(selected_ticks)!=1).sum()),'f6_recomputed_prefix_ticks':len(prefix),
                    'feature_path':str(target.relative_to(out))}
                if not target.exists():
                    f6_selected=f6[selected].copy()
                    for start in range(0,len(prefix),args.batch_size):
                        pos=prefix[start:start+args.batch_size]
                        raw=np.stack([arrays['f6'][np.maximum(np.arange(int(selected_ticks[p])-15,int(selected_ticks[p])+1),first)] for p in pos])
                        f6_selected[pos]=encoders.f6_features(torch.from_numpy(raw).to(args.device)).cpu().numpy()
                    np.savez_compressed(target,f6=f6_selected,deform=deform[selected],label=np.array(label,dtype=np.int64),
                        ticks=selected_ticks,video_frames=video[selected])
                rows.append(row)
        print('INTERVAL_CACHE',rid,len(rows),flush=True)
    counts={name:{'intervals':sum(r['split']==name for r in rows),
        'success':sum(r['split']==name and r['label']==0 for r in rows),
        'failure':sum(r['split']==name and r['label']==1 for r in rows)} for name in ('train','val','test')}
    assert len(rows)==len(data['source_intervals'])
    dump(manifest,{'signature':signature,'status':'complete','created_at':datetime.datetime.now().astimezone().isoformat(),
        'records':rows,'split_counts':counts,'total_intervals':len(rows)})
    assert sha(src/'data_manifest.json')==signature['source_manifest_sha256']
    print('PREPARE_INTERVALS_COMPLETE',json.dumps(counts),flush=True)


if __name__=='__main__': main()
