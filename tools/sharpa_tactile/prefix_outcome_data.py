"""Outcome-only SHARPA multimodal dataset and frozen causal feature caches."""
import argparse
import collections
import datetime
import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np

from .common import ROOT, FINGERS, dump, sha

OLD_ROOT=ROOT.parent/'LF3R'
OLD_MANIFEST=OLD_ROOT/'datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl'
MANIFEST=ROOT/'datasets/raw_failrecovery/manifest.jsonl'
DEFORM=ROOT/'checkpoints/T-Rex/encoders/sharpa_wave_deform_encoder.pth'
DINO=ROOT/'checkpoints/dinov2-small'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def database(record):
    return Path(record['data_root'])/record['source_database_path']


def open_database(record):
    c=sqlite3.connect(database(record).resolve().as_uri()+'?mode=ro',uri=True)
    c.execute('PRAGMA query_only=ON')
    return c


def causal_alignment(record, camera="cam_high"):
    # All timestamps compared in receiver monotonic domain, never mixing
    # SDK/device/controller clocks. Stored snapshots are causal sample-and-hold.
    camera_source={'cam_high':'camera:realsense_color','cam_wrist':'camera:wrist_right'}[camera]
    sources=['tactile:right:'+f for f in FINGERS]+[camera_source,'robot:right_arm','hand_state']
    held={};rows=[];omitted=0;snapshot_future=0;stale=0
    with open_database(record) as conn:
        camera_count=conn.execute('SELECT MAX(frame_index)+1 FROM camera_frames WHERE source=?',(camera_source,)).fetchone()[0]
        if camera_count is None:raise ValueError(f'Missing camera {camera_source}: {record["id"]}')
        for index,elapsed,tick,raw in conn.execute('SELECT frame_index,elapsed_s,tick_mono_ns,snapshot_json FROM synchronized_frames ORDER BY frame_index'):
            snap=json.loads(raw)
            for key in sources:
                item=snap.get(key)
                if not item:continue
                receive=item.get('receive_mono_ns')
                if receive is None or int(receive)>tick:
                    snapshot_future+=1;continue
                if not item.get('valid',False):continue
                if key.startswith('tactile:'):
                    if item.get('payload',{}).get('deform_shape')!=[240,240] or item.get('event_id') is None:continue
                elif key==camera_source:
                    if item.get('frame_index') is None:continue
                elif key=='robot:right_arm':
                    p=item.get('payload',{})
                    values=p.get('actual_tcp_pose_base_m_rotvec_rad',[])+p.get('actual_q_rad',[])
                    if len(values)!=12 or not np.isfinite(values).all():continue
                elif key=='hand_state':
                    values=item.get('payload',{}).get('hands',{}).get('right',{}).get('angles',[])
                    if len(values)!=22 or not np.isfinite(values).all():continue
                if key not in held or receive>=held[key]['receive_mono_ns']:held[key]=item
            if not all(key in held for key in sources):omitted+=1;continue
            state=held['robot:right_arm']['payload']
            pose=state['actual_tcp_pose_base_m_rotvec_rad']+state['actual_q_rad']+held['hand_state']['payload']['hands']['right']['angles']
            video=int(held[camera_source]['frame_index'])
            if not 0<=video<camera_count:raise ValueError('Camera reference outside video')
            receive=[int(held[key]['receive_mono_ns']) for key in sources]
            assert max(receive)<=tick
            stale+=sum(bool(held[key].get('stale')) for key in sources)
            rows.append((index,elapsed,tick,video,pose,[held[key]['event_id'] for key in sources[:5]],receive))
    if not rows:raise ValueError(f'No complete causal prefix: {record["id"]}')
    arrays=dict(ticks=np.array([r[0] for r in rows],np.int64),
                time_s=np.array([r[1] for r in rows],np.float64),
                tick_mono_ns=np.array([r[2] for r in rows],np.int64),
                video_frames=np.array([r[3] for r in rows],np.int64),
                pose=np.array([r[4] for r in rows],np.float32),
                tactile_event_ids=np.array([r[5] for r in rows],np.int64),
                receive_mono_ns=np.array([r[6] for r in rows],np.int64))
    assert np.all(np.diff(arrays['ticks'])>0) and np.all(np.diff(arrays['tick_mono_ns'])>0)
    age=(arrays['tick_mono_ns'][:,None]-arrays['receive_mono_ns'])/1e6
    audit=dict(usable_ticks=len(rows),omitted_startup_ticks=omitted,
               rejected_future_snapshot_references=snapshot_future,held_stale_source_ticks=stale,
               maximum_age_ms_by_source=dict(zip(sources,age.max(0).tolist())),
               median_age_ms_by_source=dict(zip(sources,np.median(age,axis=0).tolist())),
               causal_violations=int((age<0).sum()))
    return arrays,audit


def prepare(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    (out/'alignment').mkdir()
    old={Path(r['source_manifest_path']).parent.name:r for r in read_rows(OLD_MANIFEST)}
    source=read_rows(MANIFEST);records=[];excluded=[]
    manifest_sha=sha(MANIFEST);old_sha=sha(OLD_MANIFEST)
    for raw in source:
        episode=Path(raw['source_database_path']).parent.name
        prior=old.get(episode);outcome=raw['ground_truth_outcome']
        provenance=dict(source=str(MANIFEST.relative_to(ROOT)),sha256=manifest_sha,field='ground_truth_outcome')
        if 'discard' in str(raw.get('review',{}).get('notes','')).lower():
            excluded.append(dict(rollout_id=raw['id'],reason='source review explicitly Discard'));continue
        if outcome=='unknown' and prior and prior['ground_truth_outcome'] in ('success','failure'):
            outcome=prior['ground_truth_outcome']
            provenance=dict(source=str(OLD_MANIFEST),sha256=old_sha,source_rollout_id=prior['id'],
                            field='ground_truth_outcome',outcome_source=prior.get('outcome_source'))
        if outcome not in ('success','failure'):
            excluded.append(dict(rollout_id=raw['id'],reason='no existing final outcome in current or prior manifest'));continue
        record=dict(raw,ground_truth_outcome=outcome,label=int(outcome=='failure'),gt_provenance=provenance,
                    prior_rollout_id=prior['id'] if prior else None)
        arrays,audit=causal_alignment(record)
        path=out/'alignment'/(record['id']+'.npz');np.savez_compressed(path,**arrays)
        record.update(alignment_path=str(path.relative_to(out)),alignment_audit=audit)
        records.append(record)
        print('ALIGN',len(records),record['id'],len(arrays['ticks']),flush=True)
    # Fixed rollout split, stratified by task and final outcome. No annotations.
    rng=np.random.default_rng(42);groups=collections.defaultdict(list)
    for r in records:groups[(r['task_key'],r['label'])].append(r['id'])
    membership={};split={k:[] for k in ('train','val','test')}
    for key,ids in sorted(groups.items()):
        ids=np.array(sorted(ids));rng.shuffle(ids);hold=max(1,round(len(ids)*.15)) if len(ids)>=3 else 0
        hold=min(hold,(len(ids)-1)//2)
        for phase,part in (('test',ids[:hold]),('val',ids[hold:2*hold]),('train',ids[2*hold:])):
            for rid in part:membership[str(rid)]=phase;split[phase].append(str(rid))
    for r in records:r['split']=membership[r['id']]
    for phase in split:assert {r['label'] for r in records if r['split']==phase}=={0,1}
    dump(out/'dataset_manifest.json',dict(records=records,excluded=excluded,source_manifest_sha256=manifest_sha,
             prior_manifest_sha256=old_sha,gt_uses_annotations=False))
    dump(out/'split_manifest.json',dict(seed=42,unit='rollout',stratification='task x final outcome',**split))
    (out/'input_manifest_snapshot.jsonl').write_bytes(MANIFEST.read_bytes())
    (out/'prior_manifest_snapshot.jsonl').write_bytes(OLD_MANIFEST.read_bytes())
    dump(out/'data_protocol.json',dict(created_at=datetime.datetime.now().astimezone().isoformat(),
         gt='success0/failure1 from existing final outcomes; label constant across full rollout; no interval annotations',
         unknown='explicit discard or absent final outcome excluded with full ID/reason list; no pseudo labels',
         timeline='all recorder synchronized ticks after first complete causal observation; no F6 window warmup',
         alignment='latest valid snapshot at or before receiver tick_mono_ns, causal hold across missing/stale readings; no future interpolation',
         pose='34D measured right TCP xyz+rotvec6, right arm actual joint6, measured right hand angles22; no operator/action input',
         rgb='cam_high realsense_color only; frozen DINOv2 ViT-S/14 CLS384, official resize256/center224/ImageNet normalization',
         tactile='frozen existing DeformEncoder, float32 grayscale0..255, each finger AdaptiveAvgPool2x2=512, concatenate2560',
         dimensions='cached tactile2560/RGB384 -> trainable Linear128 each; pose normalize+MLP128; all modalities128D before common fusion',
         encoders={'deform_sha256':sha(DEFORM),'dino_model_sha256':sha(DINO/'model.safetensors')},
         references=['https://github.com/facebookresearch/dinov2','https://huggingface.co/facebook/dinov2-small']))
    print('PREPARE_COMPLETE',len(records),collections.Counter(r['ground_truth_outcome']for r in records),flush=True)


def reusable_deform(record,a):
    old_id=record.get('prior_rollout_id')
    if not old_id:return {}
    candidates=[ROOT/'outputs/sharpa_tactile_three_class/20261004_160624/features'/(old_id+'.npz'),
                ROOT/'outputs/sharpa_failure_relabel/20261006_165000/features'/(old_id+'.npz')]
    existing=next((p for p in candidates if p.is_file()),None)
    if not existing:return {}
    signature_path=existing.parent.parent/('data_manifest.json' if existing==candidates[0] else 'protocol.json')
    signature=json.loads(signature_path.read_text())
    expected=signature.get('signature',{}).get('deform_sha256') or signature.get('encoder_sha256',{}).get('deform')
    # Hash once per worker rather than rereading the checkpoint for each rollout.
    if not hasattr(reusable_deform,'encoder_sha'):reusable_deform.encoder_sha=sha(DEFORM)
    if expected!=reusable_deform.encoder_sha:return {}
    old_records={r['id']:r for r in read_rows(OLD_MANIFEST)}
    prior=old_records[old_id]
    old_sync=OLD_ROOT/prior['synchronized_frames_path']
    if not old_sync.is_file():return {}
    old_rows=read_rows(old_sync)
    with np.load(existing) as z:
        tick_to_index={int(t):i for i,t in enumerate(z['ticks'])}
        bank=z['deform']  # Decompress the array once, not once per cached tick.
        features={}
        for j,t in enumerate(a['ticks']):
            i=tick_to_index.get(int(t))
            if i is None:continue
            row=old_rows[int(t)]
            if int(row['tick_mono_ns'])!=int(a['tick_mono_ns'][j]):continue
            event_ids=[row.get('tactile',{}).get(f,{}).get('event_id') for f in FINGERS]
            if event_ids!=a['tactile_event_ids'][j].tolist():continue
            features[j]=bank[i].copy()
    return features


def extract(args):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    records=json.loads((args.output/'dataset_manifest.json').read_text())['records']
    cache=args.output/'features'/args.modality;cache.mkdir(parents=True,exist_ok=True)
    if args.modality=='tactile':
        spec=importlib.util.spec_from_file_location('frozen_deform',ROOT/'repos/T-Rex/qwen_vla/DeformAE.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        encoder=module.DeformEncoder();encoder.load_state_dict(torch.load(DEFORM,map_location='cpu',weights_only=True),strict=True)
        encoder.eval().requires_grad_(False).to(args.device)
    else:
        import cv2
        from transformers import AutoImageProcessor,Dinov2Model
        processor=AutoImageProcessor.from_pretrained(DINO,local_files_only=True,use_fast=False)
        encoder=Dinov2Model.from_pretrained(DINO,local_files_only=True).eval().requires_grad_(False).to(args.device)
    records=records[args.shard_index::args.shards]
    for number,r in enumerate(records,1):
        target=cache/(r['id']+'.npz')
        if target.exists():
            with np.load(target) as z:
                assert z['features'].shape[1]==(2560 if args.modality=='tactile' else 384)
            print('CACHED',args.modality,r['id'],flush=True);continue
        with np.load(args.output/r['alignment_path']) as z:a={k:z[k] for k in z.files}
        if args.modality=='tactile':
            reused=reusable_deform(r,a)
            features=np.empty((len(a['ticks']),2560),np.float32)
            for j,value in reused.items():features[j]=value
            missing=[j for j in range(len(features)) if j not in reused]
            # Compare one reused tick on the first eligible rollout, in the same
            # extraction pass, to validate the reused encoder/preprocessing scale.
            check_index=next(iter(reused)) if reused and number==1 else None
            work=missing+([check_index] if check_index is not None else [])
            with open_database(r) as conn,torch.inference_mode():
                for start in range(0,len(work),8):
                    ii=work[start:start+8];images=[]
                    for j in ii:
                        for event in a['tactile_event_ids'][j]:
                            row=conn.execute('SELECT deform_blob,deform_shape_json FROM tactile_frames WHERE event_id=?',(int(event),)).fetchone()
                            if row is None or json.loads(row[1])!=[240,240] or len(row[0])!=57600:
                                raise ValueError(f'Invalid deform {r["id"]}/{event}')
                            images.append(np.frombuffer(row[0],np.uint8).reshape(1,240,240))
                    x=torch.from_numpy(np.stack(images).astype(np.float32)).to(args.device)
                    value=F.adaptive_avg_pool2d(encoder(x),(2,2)).reshape(len(ii),2560).cpu().numpy()
                    for j,v in zip(ii,value):
                        if j==check_index:
                            if not np.allclose(v,features[j],rtol=2e-4,atol=2e-3):raise ValueError('Reuse encoder feature mismatch')
                        else:features[j]=v
            audit=dict(reused_ticks=len(reused),fresh_ticks=len(missing),reuse_verification='mono timestamps and all5 event IDs agree with original cached source',reuse_numeric_check=check_index is not None)
        else:
            import cv2
            desired=np.unique(a['video_frames']);lookup={};cap=cv2.VideoCapture(str(Path(r['data_root'])/r['camera_video_paths'][r.get('rgb_camera_key','cam_high')]))
            if not cap.isOpened():raise ValueError(f'Cannot read video {r["id"]}')
            want=set(desired.tolist());images=[];indices=[];frame=0
            def flush():
                if not images:return
                with torch.inference_mode():
                    value=encoder(**processor(images=images,return_tensors='pt').to(args.device)).last_hidden_state[:,0].cpu().numpy()
                lookup.update(zip(indices,value));images.clear();indices.clear()
            while frame<=desired[-1]:
                ok,bgr=cap.read()
                if not ok:raise ValueError(f'Video truncated {r["id"]}/{frame}')
                if frame in want:
                    images.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));indices.append(frame)
                    if len(images)==64:flush()
                frame+=1
            flush();cap.release()
            features=np.stack([lookup[int(i)] for i in a['video_frames']]).astype(np.float32)
            audit=dict(encoded_unique_frames=len(desired),causal_held_frames=len(features)-len(desired),camera=r.get('rgb_camera_key','cam_high'))
        assert np.isfinite(features).all()
        temp=target.with_suffix('.tmp.npz');np.savez_compressed(temp,features=features,ticks=a['ticks']);temp.replace(target)
        dump(target.with_suffix('.json'),dict(rollout_id=r['id'],modality=args.modality,rows=len(features),**audit))
        print('FEATURES',args.modality,number,len(records),r['id'],len(features),audit,flush=True)
    print('FEATURE_COMPLETE',args.modality,args.shard_index,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','extract'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--modality',choices=('tactile','rgb'),default='tactile')
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    args=p.parse_args();args.output=args.output.resolve();globals()[args.action](args)
