from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch
from .common import ROOT, FINGERS, dump, sha
from .data import load_sources, episode_arrays, DeformStreams, rollout_split
from .models import FrozenEncoders


def reconstruction_once(encoders, arrays, streams, out, record):
    """Real-data F6 VQ-VAE reconstruction; no deform decoder per user instruction."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    candidates = arrays['endpoints'][arrays['labels'] >= 0]
    # Select a contact-rich window solely from this train rollout.
    t = int(max(candidates, key=lambda t: np.linalg.norm(arrays['f6'][t, :, :3])))
    raw = torch.from_numpy(arrays['f6'][t-15:t+1]).unsqueeze(0).to(encoders.min_f6.device)
    norm = encoders.normalize(raw)
    with torch.inference_mode():
        codes = encoders.vq.encode(norm)
        reconstructed = encoders.vq.decode_indices(codes)
    x, y = norm[0].cpu().numpy(), reconstructed[0].cpu().numpy()
    if not np.isfinite(y).all():
        raise ValueError('Nonfinite pretrained F6 reconstruction')
    stats = {'status': 'executed', 'rollout_id': record['id'], 'tick': t,
             'video_frame': int(arrays['video_frames'][np.where(arrays['endpoints']==t)[0][0]]),
             'normalization': 'Official frozen right-hand F6 min/max/mask',
             'normalized_mse': float(np.mean((x-y)**2)),
             'zero_prediction_mse': float(np.mean(x*x)),
             'codes': codes.cpu().tolist(), 'deform_reconstruction': 'omitted_by_user',
             'note': 'Reconstruction executed once on real training data; no arbitrary quality threshold.'}
    np.savez_compressed(out/'reconstruction_f6.npz', normalized_input=x, normalized_reconstruction=y,
                        raw_input=raw[0].cpu().numpy(), codes=codes.cpu().numpy())
    fig, axes = plt.subplots(5, 2, figsize=(12, 12), sharex=True)
    colors = ('#2563eb', '#16a34a', '#d97706')
    for j, finger in enumerate(FINGERS):
        for c in range(2):
            for k in range(3):
                d = c*3+k
                axes[j,c].plot(x[:,j,d],color=colors[k],label=('F' if c==0 else 'M')+str(k)+' input')
                axes[j,c].plot(y[:,j,d],color=colors[k],linestyle='--',label=('F' if c==0 else 'M')+str(k)+' recon')
            axes[j,c].set_title(finger + (' forces' if c==0 else ' moments'))
    axes[0,0].legend(fontsize=7,ncol=2);axes[0,1].legend(fontsize=7,ncol=2)
    fig.suptitle('Pretrained F6 reconstruction on a real training window (normalized values)')
    fig.tight_layout();fig.savefig(out/'reconstruction_f6.png',dpi=140);plt.close(fig)
    dump(out/'sanity.json',stats)
    print('SANITY',json.dumps(stats),flush=True)


def main():
    parser = argparse.ArgumentParser(description='Cache frozen tactile features and make shared rollout splits')
    parser.add_argument('--intervals',type=Path,default=ROOT/'outputs/usb_event_intervals/20261003_202352/intervals.jsonl')
    parser.add_argument('--manifest',type=Path,default=ROOT/'datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--camera',default='cam_high')
    args=parser.parse_args();torch.set_num_threads(2)
    args.output.mkdir(parents=True,exist_ok=True);cache=args.output/'features';cache.mkdir(exist_ok=True)
    sources=load_sources(args.intervals,args.manifest)
    records={rid:r for rid,(r,_) in sources.items()}
    arrays={rid:episode_arrays(r,events,args.camera) for rid,(r,events) in sources.items()}
    split=rollout_split(records,{rid:a['labels'] for rid,a in arrays.items()},args.seed)
    dump(args.output/'split_manifest.json',split)
    encoders=FrozenEncoders().to(args.device)
    signature={'interval_sha256':sha(args.intervals),'manifest_sha256':sha(args.manifest),
               'f6_sha256':sha(encoders.f6_path),'deform_sha256':sha(encoders.deform_path),
               'camera':args.camera,'window':16,'f6_hand':'right','f6_feature':'continuous pre-quantization, flatten 5x256',
               'deform_feature':'frozen T-Rex DeformEncoder -> AdaptiveAvgPool2d(2,2) -> flatten 5x512',
               'cache_dtype':'float32','encoder_precision':'float32; TF32 disabled','binary':{'0':'success (8/9)','1':'failure (6/7)'},
               'interval_bounds':'closed [causal,observable]','unlabelled':'loss ignored; past context retained',
               'causality':'F6 t-15:t; sync event receive_mono_ns <= tick_mono_ns; no future filling',
               'invalid_samples':'drop sample if any of 16 ticks invalid; reset sequence across gaps'}
    signature['episode_sources']={rid:{'frames_sha256':sha(ROOT/r['synchronized_frames_path']),
        'events_sha256':sha(ROOT/r['tactile_events_path']),
        'deform_streams':{finger:{'size':(ROOT/r['tactile_stream_paths'][finger]['deform']).stat().st_size,
        'mtime_ns':(ROOT/r['tactile_stream_paths'][finger]['deform']).stat().st_mtime_ns} for finger in FINGERS}}
        for rid,r in records.items()}
    old=args.output/'data_manifest.json'
    if old.exists() and json.loads(old.read_text())['signature']!=signature:
        raise ValueError('Feature cache settings changed; use a new output directory')
    dump(old,{'signature':signature,'status':'preparing'})
    rid=split['train'][0]
    if not (args.output/'sanity.json').exists():
        reconstruction_once(encoders,arrays[rid],DeformStreams(records[rid]),args.output,records[rid])
    audit={};started=time.monotonic()
    for i,(rid,(record,events)) in enumerate(sources.items()):
        arr=arrays[rid];target=cache/(rid+'.npz');audit[rid]=arr['audit']
        if target.exists():
            print(f'CACHE {i+1}/{len(sources)} reused {rid}',flush=True);continue
        streams=DeformStreams(record);endpoints=arr['endpoints']
        f6_features=[];deform_features=[]
        with torch.inference_mode():
            for start in range(0,len(endpoints),args.batch_size):
                ticks=endpoints[start:start+args.batch_size]
                raw=np.stack([arr['f6'][t-15:t+1] for t in ticks])
                images=streams.batch([arr['references'][t] for t in ticks])
                f6=encoders.f6_features(torch.from_numpy(raw).to(args.device)).cpu().numpy()
                deform=encoders.deform_features(torch.from_numpy(images).to(args.device)).cpu().numpy()
                if not np.isfinite(f6).all() or not np.isfinite(deform).all():
                    raise ValueError('Nonfinite encoded features')
                f6_features.append(f6);deform_features.append(deform)
        temp=target.with_suffix('.tmp.npz')
        np.savez_compressed(temp,f6=np.concatenate(f6_features),deform=np.concatenate(deform_features),
                            labels=arr['labels'],video_frames=arr['video_frames'],ticks=arr['ticks'],
                            segment_starts=arr['segment_starts'])
        temp.replace(target)
        print(f'CACHE {i+1}/{len(sources)} rows={len(endpoints)} seconds={time.monotonic()-started:.1f} {rid}',flush=True)
    assert all(not p.requires_grad for p in encoders.parameters())
    summary={name:{'rollouts':len(split[name]),'failure_rows':sum(audit[r]['failure_rows'] for r in split[name]),
                   'success_rows':sum(audit[r]['success_rows'] for r in split[name])} for name in ('train','val','test')}
    dump(args.output/'data_manifest.json',{'signature':signature,'created_at':datetime.datetime.now().astimezone().isoformat(),
         'intervals':str(args.intervals.relative_to(ROOT)),'manifest':str(args.manifest.relative_to(ROOT)),
         'source_intervals':[e for _,events in sources.values() for e in events],
         'records':records,'audit':audit,'split_counts':summary,'status':'complete'})
    print('PREPARE_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
