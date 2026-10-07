"""Refresh live 3/4 labels, retrain both frame heads, audit frame3 on success."""
from .common import project_path, relative_path
from .common import ANNOTATION_ROOT
import argparse
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from .common import ROOT,dump,read_jsonl,sha
from .probe_metrics import metrics,write_csv
from .failure_relabel import EVENTS,KINDS,SEEDS,load_samples,train


def prepare(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    backup=out/'original_annotations_backup';backup.mkdir()
    manifest_path=project_path('datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl')
    raw_records=read_jsonl(manifest_path)
    records={r['id']:r for r in raw_records if 'usb' in r['task_key']}
    events=[];snapshots=[];eligible=[];excluded=[]
    old_events=json.loads((args.source/'source_intervals.json').read_text())
    old_by_id=defaultdict(list)
    for e in old_events:old_by_id[e['rollout_id']].append(e)
    from .data import load_sources
    interval_path=getattr(args, 'intervals', ANNOTATION_ROOT/'intervals.jsonl')
    sources=load_sources(interval_path, manifest_path, label_keys=(3,4))
    shutil.copyfile(interval_path, out/'canonical_intervals.jsonl')
    snapshots.append(dict(source=str(interval_path), backup='canonical_intervals.jsonl', sha256=sha(interval_path)))
    for rid,(r,annotation_events) in sorted(sources.items()):
        if rid not in records or r['ground_truth_outcome']!='failure':continue
        selected=[e for e in annotation_events if e['event_key'] in (3,4)]
        if selected:eligible.append(rid);events.extend(selected)
    prior=json.loads((args.source/'split_manifest.json').read_text())
    split={k:sorted(set(prior[k])&set(eligible)) for k in ('train','val')}
    newly_eligible=sorted(set(eligible)-set(split['train'])-set(split['val']))
    split['train']=sorted(split['train']+newly_eligible)
    assert not set(split['train'])&set(split['val']) and set(split['train']+split['val'])==set(eligible)
    membership={rid:phase for phase,ids in split.items() for rid in ids}
    dump(out/'split_manifest.json',dict(seed=42,unit='rollout',policy='preserve previous membership;newly eligible rollout added to train',
         source_split_sha256=sha(args.source/'split_manifest.json'),newly_eligible=newly_eligible,**split))
    dump(out/'source_intervals.json',events)
    changed=[]
    for rid in sorted(set(old_by_id)|set(eligible)):
        previous=[{k:e[k] for k in ('event_index','event_key','start_frame','end_frame')} for e in old_by_id[rid]]
        current=[{k:e[k] for k in ('event_index','event_key','start_frame','end_frame')} for e in events if e['rollout_id']==rid]
        if previous!=current:changed.append(dict(rollout_id=rid,previous=previous,current=current))
    dump(out/'annotation_snapshot.json',dict(records=snapshots,excluded=excluded,changed_rollouts=changed,
         event_mapping=EVENTS,source_manifest_sha256=sha(manifest_path)))
    print('LATEST_LABELS',len(eligible),'failure rollouts',len(events),'intervals',
          {k:sum(e['event_key']==k for e in events) for k in (3,4)},'split',len(split['train']),len(split['val']),flush=True)
    frame_rows=[];missing=[]
    for rid in eligible:
        candidates=(args.source/'features'/(rid+'.npz'),args.cache/'features'/(rid+'.npz'))
        path=next((p for p in candidates if p.exists()),None)
        if path is None:
            missing.append(rid);path=out/'features'/(rid+'.npz')
        frame_rows.append(dict(group='frame',rollout_id=rid,sample_id=rid,split=membership[rid],
            task_key=records[rid]['task_key'],feature_path=str(relative_path(path)),feature_origin='reused' if rid not in missing else 'new'))
    if missing:extract_missing(args,records,missing)
    dump(out/'dataset_manifest.json',dict(status='complete',feature_source='.',labels_from_snapshot=True,
         records=frame_rows,excluded=excluded,new_feature_rollouts=missing))
    success_records=[]
    for rid,r in sorted(records.items()):
        if r['ground_truth_outcome']!='success':continue
        path=args.cache/'features'/(rid+'.npz');assert path.exists(),f'Success feature cache missing:{rid}'
        success_records.append(dict(rollout_id=rid,sample_id=rid,task_key=r['task_key'],
            feature_path=str(relative_path(path)),ground_truth=0))
    assert len(success_records)==92 and not set(eligible)&{r['rollout_id'] for r in success_records}
    dump(out/'success_manifest.json',dict(unit='rollout',GT='all0 by user success-audit rule',records=success_records))
    protocol=json.loads((args.source/'protocol.json').read_text())
    protocol.update(groups={'frame_3':'current camera frame within union of current3 intervals=1,all else0 including4',
                            'frame_4':'current camera frame within union of current4 intervals=1,all else0 including3'},
        event_mapping={'frame_3':'inside3=1,outside3=0','frame_4':'inside4=1,outside4=0'},
        source=str(relative_path(args.source)),baseline_frame3=str(relative_path(args.frame3_baseline)),
        label_source='canonical merged intervals snapshot; cached labels never used',
        sampling='full valid failure rollout,dense step0,noGaussian/noKey/noaugmentation',
        split='preserve old train/val membership;new eligible assigned train;no test',
        normalization='train20 rollouts only mean/std,std floor.01;success never used',
        interval_context='not applicable: only frame classification in this run',
        loss='train-frame-count balanced BCEWithLogits w0=N/(2N0),w1=N/(2N1)',
        success='frame3 checkpoints only;all92success rolloutsGT0;fixed.5,any validtick>=.5 ->rolloutFP;no threshold tuning',
        success_selection='not used for training,normalization,checkpoint selection or threshold selection',
        created_at=datetime.datetime.now().astimezone().isoformat())
    dump(out/'protocol.json',protocol)
    distribution=[]
    for group in ('frame_3','frame_4'):
        for phase,ss in load_samples(out,group).items():
            y=np.concatenate([s['y'] for s in ss]);n0,n1=int((y==0).sum()),int(y.sum());assert n0>0 and n1>0
            distribution.append(dict(group=group,split=phase,rollouts=len(ss),frames=len(y),negative=n0,positive=n1,
                positive_fraction=n1/len(y),negative_weight=len(y)/(2*n0),positive_weight=len(y)/(2*n1)))
    write_csv(out/'prepared_distribution.csv',distribution)
    print('DATA_READY',distribution,'success',len(success_records),flush=True)


def extract_missing(args,records,missing):
    """Exact existing encoder preprocessing; bypass unrelated VLA package imports."""
    import importlib.util
    import torch
    from torch import nn
    from tactile_vqvae.models.tactile_vqvae import TactileVQVAE,TactileVQVAEConfig
    from .data import episode_arrays,DeformStreams
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    spec=importlib.util.spec_from_file_location('lf3r_frozen_deform',project_path('repos/T-Rex/qwen_vla/DeformAE.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fpath=project_path('checkpoints/T-Rex/encoders/f6_tactile_vqvae.pt');dpath=project_path('checkpoints/T-Rex/encoders/sharpa_wave_deform_encoder.pth')
    signature=json.loads((args.source/'protocol.json').read_text())['encoder_sha256']
    assert sha(fpath)==signature['f6'] and sha(dpath)==signature['deform']
    blob=torch.load(fpath,map_location='cpu',weights_only=True);vq=TactileVQVAE(TactileVQVAEConfig.from_dict(blob['config']))
    vq.load_state_dict(blob['model_state'],strict=True);deform=module.DeformEncoder()
    deform.load_state_dict(torch.load(dpath,map_location='cpu',weights_only=True),strict=True)
    vq.requires_grad_(False).eval().to(args.device);deform.requires_grad_(False).eval().to(args.device)
    pool=nn.AdaptiveAvgPool2d((2,2))
    stats={k:torch.as_tensor(blob['stats']['tacf6_'+k])[30:60].reshape(5,6).to(args.device) for k in ('min','max','mask')}
    (args.output/'features').mkdir(exist_ok=True)
    with torch.inference_mode():
        for rid in missing:
            a=episode_arrays(records[rid],[],num_classes=3);streams=DeformStreams(records[rid])
            f6=np.empty((len(a['ticks']),1280),np.float32);df=np.empty((len(a['ticks']),2560),np.float32)
            for first in range(0,len(f6),8):
                ticks=a['ticks'][first:first+8];raw=torch.from_numpy(np.stack([a['f6'][t-15:t+1] for t in ticks])).to(args.device)
                normalized=(2*(raw-stats['min'])/(stats['max']-stats['min']+1e-8)-1).clamp(-1,1)
                normalized=torch.where(stats['mask'],normalized,raw)
                f6[first:first+len(ticks)]=vq.encoder(normalized).flatten(1).cpu().numpy()
                images=torch.from_numpy(streams.batch([a['references'][int(t)] for t in ticks])).to(args.device)
                df[first:first+len(ticks)]=pool(deform(images.flatten(0,1))).reshape(len(ticks),2560).cpu().numpy()
            np.savez_compressed(args.output/'features'/(rid+'.npz'),f6=f6,deform=df,ticks=a['ticks'],video_frames=a['video_frames'])
            print('NEW_FROZEN_CACHE',rid,len(f6),flush=True)
    del vq,deform;torch.cuda.empty_cache()


def success_stats(samples,p,offsets,kind,seed):
    rows=[]
    for i,s in enumerate(samples):
        a,b=offsets[i:i+2];local=p[a:b];assert len(local)==len(s['frames'])
        hits=np.flatnonzero(local>=.5);first=int(hits[0]) if len(hits) else None
        rows.append(dict(input=kind,seed=seed,rollout_id=s['rollout_id'],task_key=s['task_key'],frames=len(local),
            fp_frames=len(hits),frame_fpr=len(hits)/len(local),rollout_fp=bool(len(hits)),
            max_probability=float(local.max()),mean_probability=float(local.mean()),
            first_alarm_camera_frame=int(s['frames'][first]) if first is not None else None,
            first_alarm_seconds=float(s['frames'][first]/30) if first is not None else None))
    summary={}
    for task in ('all','usb_panel','usb_socket'):
        rr=[r for r in rows if task=='all' or r['task_key']==task]
        assert rr
        frames=sum(r['frames'] for r in rr);fp_frames=sum(r['fp_frames'] for r in rr);fp_rollouts=sum(r['rollout_fp'] for r in rr)
        summary[task]=dict(rollouts=len(rr),fp_rollouts=int(fp_rollouts),rollout_fpr=fp_rollouts/len(rr),
            frames=frames,fp_frames=fp_frames,frame_fpr=fp_frames/frames)
    return summary,rows


def fit(args):
    records=json.loads((args.output/'success_manifest.json').read_text())['records'];ss=[]
    print('LOADING_SUCCESS_FROZEN_CACHE',len(records),flush=True)
    for i,r in enumerate(records):
        with np.load(project_path(r['feature_path'])) as z:
            s=dict(r,frames=z['video_frames'].copy(),f6=z['f6'].copy(),deform=z['deform'].copy())
        s['y']=np.zeros(len(s['frames']),np.float32);ss.append(s)
        if (i+1)%20==0:print('SUCCESS_CACHE',i+1,len(records),flush=True)
    offsets=np.r_[0,np.cumsum([len(s['frames']) for s in ss])]
    np.savez_compressed(args.output/'success_coordinates.npz',video_frames=np.concatenate([s['frames'] for s in ss]),offsets=offsets)
    def after_run(model,infer,kind,group,seed,directory,result):
        if group!='frame_3':return
        p,offsets=infer(model,ss,kind)
        summary,rows=success_stats(ss,p,offsets,kind,seed)
        np.savez_compressed(directory/'success_predictions.npz',probabilities=p,offsets=offsets)
        write_csv(directory/'success_rollouts.csv',rows);result['success_frame3']=summary
        print('SUCCESS_FRAME3',kind,seed,'FP',summary['all']['fp_rollouts'],'/',len(ss),
              'frameFPR',round(summary['all']['frame_fpr'],4),flush=True)
    args.groups=('frame_3','frame_4');args.after_run=after_run
    train(args)
    dump(args.output/'gpu_complete.json',dict(status='complete',training_runs=30,success_checkpoint_evaluations=15,
         encoders_frozen=True,completed_at=datetime.datetime.now().astimezone().isoformat()))
    print('GPU_WORK_COMPLETE',flush=True)


def aggregate(results,group,kind):
    rr=[r for r in results if r['group']==group and r['input']==kind]
    assert len(rr)==5 and {r['seed'] for r in rr}==set(SEEDS)
    row=dict(group=group,input=kind,seeds=5)
    for key in ('accuracy','balanced_accuracy','macro_f1','precision','recall','fpr','pr_auc','roc_auc'):
        values=[r['val'][key] for r in rr];row[key+'_mean']=float(np.mean(values));row[key+'_std']=float(np.std(values,ddof=1))
    return row


def report(args):
    out=args.output;results=json.loads((out/'results.json').read_text());assert len(results)==30
    val_summary=[aggregate(results,g,k) for g in ('frame_3','frame_4') for k in KINDS]
    write_csv(out/'summary.csv',val_summary)
    success_records=json.loads((out/'success_manifest.json').read_text())['records']
    with np.load(out/'success_coordinates.npz') as z:frames=z['video_frames'].copy();offsets=z['offsets'].copy()
    ss=[dict(r,frames=frames[offsets[i]:offsets[i+1]]) for i,r in enumerate(success_records)]
    events=[];success_runs=[];success_summary=[];normal=[]
    for r in results:
        directory=out/'runs'/r['group']/r['input']/f"seed_{r['seed']}"
        with np.load(directory/'val_predictions.npz') as z:replay=metrics(z['labels'],z['probabilities'])
        assert replay['confusion_matrix']==r['val']['confusion_matrix']
        for key in ('balanced_accuracy','macro_f1','pr_auc','roc_auc'):assert np.isclose(replay[key],r['val'][key])
        if r['group']!='frame_3':continue
        with np.load(directory/'success_predictions.npz') as z:
            assert np.array_equal(z['offsets'],offsets);summary,rows=success_stats(ss,z['probabilities'],offsets,r['input'],r['seed'])
        assert summary==r['success_frame3'];events.extend(rows)
        for task,v in summary.items():success_runs.append(dict(input=r['input'],seed=r['seed'],task=task,**v))
    write_csv(out/'success_rollout_predictions.csv',events);write_csv(out/'success_frame3_runs.csv',success_runs)
    for kind in KINDS:
        for task in ('all','usb_panel','usb_socket'):
            rr=[r for r in success_runs if r['input']==kind and r['task']==task]
            row=dict(input=kind,task=task,seeds=5,rollouts=rr[0]['rollouts'],frames=rr[0]['frames'])
            for key in ('fp_rollouts','rollout_fpr','fp_frames','frame_fpr'):
                values=[r[key] for r in rr];row[key+'_mean']=float(np.mean(values));row[key+'_std']=float(np.std(values,ddof=1))
            row['fp_rollouts_min']=min(r['fp_rollouts'] for r in rr);row['fp_rollouts_max']=max(r['fp_rollouts'] for r in rr)
            success_summary.append(row)
    write_csv(out/'success_frame3_summary.csv',success_summary)
    for group in ('frame_3','frame_4'):
        for kind in KINDS:
            rr=[r for r in results if r['group']==group and r['input']==kind]
            cm=np.mean([r['val']['confusion_matrix'] for r in rr],axis=0);nr=cm/cm.sum(1,keepdims=True)
            for gt in (0,1):
                for pred in (0,1):normal.append(dict(group=group,input=kind,gt=gt,prediction=pred,mean_count=cm[gt,pred],row_normalized=nr[gt,pred]))
    write_csv(out/'confusion_normalized.csv',normal)
    # Re-score previous predictions with refreshed GT to separate relabeling from retraining.
    rescored=[]
    for group,baseline in (('frame_3',args.frame3_baseline),('frame_4',args.source)):
        current=load_samples(out,group,only_split='val')['val'];y=np.concatenate([s['y'] for s in current])
        for kind in KINDS:
            for seed in SEEDS:
                directory=baseline/'runs'/group/kind/f'seed_{seed}'
                previous=json.loads((directory/'val_samples.json').read_text())
                assert [s['rollout_id'] for s in current]==[s['rollout_id'] for s in previous]
                with np.load(directory/'val_predictions.npz') as z:
                    assert np.array_equal(z['offsets'],np.r_[0,np.cumsum([len(s['frames']) for s in current])])
                    rescore=metrics(y,z['probabilities'])
                rescored.append(dict(group=group,input=kind,seed=seed,val=rescore))
    dump(out/'previous_models_on_updated_GT.json',rescored)
    write_csv(out/'previous_models_on_updated_GT.csv',[aggregate(rescored,g,k) for g in ('frame_3','frame_4') for k in KINDS])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(11,7))
    for i,group in enumerate(('frame_3','frame_4')):
        for j,kind in enumerate(KINDS):
            cm=np.mean([r['val']['confusion_matrix'] for r in results if r['group']==group and r['input']==kind],axis=0)
            v=cm/cm.sum(1,keepdims=True);ax=axes[i,j];ax.imshow(v,vmin=0,vmax=1,cmap='Blues')
            for a in (0,1):
                for b in (0,1):ax.text(b,a,f'{100*v[a,b]:.1f}%\n({cm[a,b]:.1f})',ha='center',va='center',color='white' if v[a,b]>.5 else 'black')
            names=['outside','event'+group[-1]];ax.set_xticks([0,1],names);ax.set_yticks([0,1],names)
            ax.set_title(group+' / '+kind);ax.set_xlabel('Prediction');ax.set_ylabel('GT')
    fig.tight_layout();fig.savefig(out/'confusion_normalized.png',dpi=180);fig.savefig(out/'confusion_normalized.pdf');plt.close(fig)
    for group in ('frame_3','frame_4'):
        samples=load_samples(out,group,only_split='val')['val'];fig,axes=plt.subplots(len(samples),3,figsize=(15,3*len(samples)),squeeze=False)
        for j,kind in enumerate(KINDS):
            values=[]
            for seed in SEEDS:
                with np.load(out/'runs'/group/kind/f'seed_{seed}'/'val_predictions.npz') as z:
                    values.append(z['probabilities'].copy());cuts=z['offsets'].copy()
            values=np.stack(values)
            for i,s in enumerate(samples):
                local=values[:,cuts[i]:cuts[i+1]];mean=local.mean(0);sd=local.std(0,ddof=1);ax=axes[i,j];x=s['frames']/30
                ax.plot(x,mean);ax.fill_between(x,np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),alpha=.15)
                ax.step(x,s['y'],where='post',color='black',alpha=.5);ax.axhline(.5,color='gray',ls='--');ax.set_ylim(-.03,1.03)
                ax.set_title(s['rollout_id'].split('mixedfail_')[-1]+' / '+kind);ax.set_xlabel('Camera time(s)')
        fig.suptitle(group+' refreshedGT;mean ± seed SD');fig.tight_layout()
        fig.savefig(out/(group+'_val_curves.png'),dpi=180);fig.savefig(out/(group+'_val_curves.pdf'));plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j,metric in enumerate(('rollout_fpr','frame_fpr')):
        for i,task in enumerate(('all','usb_panel','usb_socket')):
            rows=[next(r for r in success_summary if r['task']==task and r['input']==k) for k in KINDS]
            axes[j].bar(np.arange(3)+(i-1)*.25,[100*r[metric+'_mean'] for r in rows],width=.25,
                yerr=[100*r[metric+'_std'] for r in rows],capsize=3,label=task)
        axes[j].set_xticks(np.arange(3),list(KINDS));axes[j].set_ylim(0,105);axes[j].set_ylabel('%')
        axes[j].set_title('Success frame3 '+metric);axes[j].legend()
    fig.tight_layout();fig.savefig(out/'success_frame3_fpr.png',dpi=180);fig.savefig(out/'success_frame3_fpr.pdf');plt.close(fig)
    distribution=list(csv.DictReader((out/'prepared_distribution.csv').open()))
    snapshot=json.loads((out/'annotation_snapshot.json').read_text());split=json.loads((out/'split_manifest.json').read_text())
    lines=['# 补标后重跑：frame3 only / frame4 only，以及success上的frame3误报','',
        f"使用本轮实时标注的新快照：{len(split['train'])+len(split['val'])}条failure rollout，原train/val成员不变，新有3/4的rollout加入train，现{len(split['train'])}train/{len(split['val'])}val，无test。{len(snapshot['changed_rollouts'])}条rollout的3/4标注相对原始实验变化；canonical merged intervals备份到canonical_intervals.jsonl。旧实验保留。",'',
        '## GT、输入、loss和输出','',
        'Frame3 only：当前last frame处于任意最新3 interval的闭区间`[causal_onset_frame,observable_onset_frame]`内→1，其他→0包括4。Frame4 only：当前帧处于任意最新4 interval内→1，其他→0包括3。区间并集，不填补间隔，不用Gaussian/Key扩展；两组GT都从本轮快照重新生成，绝不复用缓存中的旧标签。背景有效帧全部参与。','',
        'F6连续past16rawtick→冻结T-Rex finger encoder1280D→Linear128；Deform当前tick→冻结每指encoder→pool2×2得到512D×5=2560D→Linear128；Fusion concat256。单层单向GRU128→Linear1 sigmoid，输出当前帧属于对应正例区间的score，≥0.5判1。GRU每rollout重置，仅使用当前及历史，无效tick省略。仅复用冻结特征，不训练encoder。','',
        'Balanced BCEWithLogits，训练帧数权重`w0=N/(2N0),w1=N/(2N1)`；每组按新GT重算。train-only标准化，std下限0.01。AdamW lr0.001/wd0.0001，batch8/max30epoch/patience8/clip1，seeds42–46。各run仅按failure val BA选best epoch，平手最早；success不参与loss、归一化、epoch选择或阈值选择。两组×三模态×五seeds=30次新训练。','',
        'BA=两类recall平均，Macro F1=两类F1平均；P/R/FPR针对各组positive1。AP是tie-aware average precision，ROC-AUC是ROC梯形积分。表为5seeds均值±样本SD；同一val也用于epoch选择，不是独立test。','',
        '## Failure train/val分布','', '| Group | Split | Rollouts | Negative | Positive |','|---|---|---:|---:|---:|']
    for r in distribution:lines.append(f"| {r['group']} | {r['split']} | {r['rollouts']} | {r['negative']} | {r['positive']} |")
    lines+=['','## Failure validation（%，mean ± SD）','', '| Group | Input | BA | Macro F1 | Precision | Recall | FPR | AP | ROC-AUC |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in val_summary:
        values=[f"{100*r[k+'_mean']:.2f} ± {100*r[k+'_std']:.2f}" for k in ('balanced_accuracy','macro_f1','precision','recall','fpr','pr_auc','roc_auc')]
        lines.append('| '+r['group']+' | '+r['input']+' | '+' | '.join(values)+' |')
    lines+=['','## Success推理：仅frame3 checkpoint','',
        '使用上述frame3模型的15个best checkpoint，在92条success上推理（usb_panel47，usb_socket45）。按用户success评价规则，全段GT一律0；即使success轨迹中出现局部异常，本轮任一valid tick检出都算该rollout FP。固定threshold0.5，无平滑、debounce、连续帧要求或按success调阈值。','',
        '`frame FPR = 检出为1的有效tick数 / 全部有效tick数`；`rollout FPR = 存在至少一个p≥0.5的success rollout数 / success rollout总数`。先逐seed计算，再求mean±SD；不把5seeds平均后的曲线再阈值化。所有success没有positive GT，故不报告其BA、recall、AP/ROC。','',
        'Failure训练数据均为usb_socket，因此同时分任务报告success FPR，区分同任务与usb_panel的跨任务推理。Success并未混入failure train/val，也不改变之前无test设置；这里是额外的纯负例误报检查。','',
        '| Input | Success cohort | N | FP rollouts mean ± SD | Rollout FPR % | Frame FPR % |','|---|---|---:|---:|---:|---:|']
    for r in success_summary:lines.append(f"| {r['input']} | {r['task']} | {r['rollouts']} | {r['fp_rollouts_mean']:.2f} ± {r['fp_rollouts_std']:.2f} | {100*r['rollout_fpr_mean']:.2f} ± {100*r['rollout_fpr_std']:.2f} | {100*r['frame_fpr_mean']:.2f} ± {100*r['frame_fpr_std']:.2f} |")
    lines+=['','Success每条rollout每个seed的FP判定、检出帧数、max/mean score及首次检出时间见success_rollout_predictions.csv；逐seed总数见success_frame3_runs.csv；汇总见success_frame3_summary.csv。原始每tick概率和offsets保存在各frame3 run的success_predictions.npz，success_manifest.json与success_coordinates.npz提供映射。只对frame3做success推理，没有对frame4做这一额外检查。','',
        '## 标注变更与旧模型重评分','',
        'annotation_snapshot.json列出每条changed rollout的前后event/边界。previous_models_on_updated_GT.csv将上次保存的val概率直接按新GT重评分，使用完全相同的4条val rollout；没有重新训练旧模型。它用于区分仅纠正GT导致的指标变化与本轮重新训练的变化；不可直接将旧GT下的原分数与本轮新GT分数视为同一任务比较。','',
        '![Confusion normalized](confusion_normalized.png)','',
        '矩阵GT为行，prediction为列，按GT行归一化；括号为5seeds平均原始计数。','',
        '![Frame3 refreshed validation](frame_3_val_curves.png)','',
        '![Frame4 refreshed validation](frame_4_val_curves.png)','',
        '验证曲线彩线/阴影为5seeds mean±SD，黑线为新hard GT，横轴为相机时间。','',
        '![Success frame3 false positives](success_frame3_fpr.png)','']
    (out/'README.md').write_text('\n'.join(lines))
    dump(out/'verification.json',dict(status='PASS',training_runs=30,seeds=list(SEEDS),latest_annotation_GT=True,
        old_val_membership_preserved=True,train_val_disjoint=True,success_disjoint=True,success_rollouts=92,
        success_only_frame3=True,success_any_detection_FP=True,no_success_selected_threshold=True,
        all_val_and_success_metrics_replayed=True,old_val_predictions_rescored_on_new_GT=True))
    for file in (Path(__file__),Path(__file__).with_name('failure_relabel.py')):shutil.copyfile(file,out/file.name)
    dump(out/'completion.json',dict(status='complete',completed_at=datetime.datetime.now().astimezone().isoformat(),
         code_sha256=sha(Path(__file__)),training_code_sha256=sha(Path(__file__).with_name('failure_relabel.py'))))
    publish(args)
    print('REPORT_COMPLETE',len(results),'new runs,success92×15',flush=True)


def publish(args):
    out=args.output;weekly=project_path('WeeklySummary/10.5');destination=weekly/'failure_relabel'/out.name
    destination.mkdir(parents=True,exist_ok=False)
    files=['README.md','protocol.json','split_manifest.json','annotation_snapshot.json','source_intervals.json',
        'dataset_manifest.json','success_manifest.json','prepared_distribution.csv','label_distribution.csv',
        'results.json','summary.csv','success_frame3_summary.csv','success_frame3_runs.csv','success_rollout_predictions.csv',
        'previous_models_on_updated_GT.csv','previous_models_on_updated_GT.json','confusion_normalized.csv',
        'confusion_normalized.png','confusion_normalized.pdf','frame_3_val_curves.png','frame_3_val_curves.pdf',
        'frame_4_val_curves.png','frame_4_val_curves.pdf','success_frame3_fpr.png','success_frame3_fpr.pdf',
        'verification.json','completion.json','gpu_complete.json']
    checks={}
    for name in files:
        shutil.copyfile(out/name,destination/name);assert sha(out/name)==sha(destination/name);checks[name]=sha(out/name)
    dump(destination/'copy_sha256.json',checks)
    text=(out/'README.md').read_text()
    for name in ('confusion_normalized.png','frame_3_val_curves.png','frame_4_val_curves.png','success_frame3_fpr.png'):
        text=text.replace(']('+name+')','](failure_relabel/'+out.name+'/'+name+')')
    (weekly/'10.5failure_relabel_updated.md').write_text(text+'\n完整周报副本：[README](failure_relabel/'+out.name+'/README.md)。\n')
    link='failure_relabel/'+out.name+'/README.md'
    section='\n\n## 补标后重新训练及success误报检查\n\n本轮读取新的3/4标注，20train/4val，无test；frame3/frame4三模态各5seeds，共30次。只有frame3在92success上推理，任意valid tick检出即rollout FP，固定0.5阈值；分USB任务报告FPR。旧实验保留，旧val概率也按新GT重评分。[新完整报告]('+link+')。\n'
    for name in ('10.5failure_relabel.md','10.5.md'):
        with (weekly/name).open('a') as f:f.write(section)
    now=datetime.datetime.now().astimezone().isoformat();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    env=project_path('environment_reports', 'SHARPA_FAILURE_RELABEL_REFRESH_'+out.name+'.md')
    commands='source ./project_env.sh\n'+'\n'.join(
        f'PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -u -m sharpa_tactile.failure_relabel_refresh {action} '
        f'--source {relative_path(args.source)} --cache {relative_path(args.cache)} '
        f'--frame3-baseline {relative_path(args.frame3_baseline)} --output {relative_path(out)} --device {args.device}'
        for action in ('run','report'))
    env.write_text(f'# Refreshed3/4 frame training and success frame3 audit\n\nCOMPLETE:{now}. HEAD:`{commit}`.\n\n30frozen-feature GRU runs,24failure rollouts20train4val,no test.15frame3 checkpoints on92success;fixed.5 any detection=rolloutFP. Success never used in selection/normalization/loss. Project-local ProcVLM venv;GPU{args.device};GPU work sequential in one process.\n\n```bash\n{commands}\n```\n\nCLI run exits only after all predictions/checkpoints/results and gpu_complete.json are saved and CUDA synchronized;os._exit0 bypasses slow unrelated Torch exit hooks.\n\nMetrics replay andweeklycopySHA PASS. Report:`{relative_path(destination)}/README.md`.\n')
    for name in ('SETUP_STATUS.md','SYSTEM_INFO.txt'):
        with (project_path(name)).open('a') as f:f.write(f'\nFailure-label refresh COMPLETE({now}):30frozen-featureGRUruns {args.device},frame3/frame4×3modal×5seeds;24failure20train4val,no test. Frame3-only success92×15inference,any.5hit=rolloutFP;no success training/selection. Allreplay/copySHA PASS. Report:{relative_path(env)}. HEAD:{commit}.\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','report'))
    parser.add_argument('--source',type=Path,default=project_path('outputs/sharpa_failure_relabel/20261006_165000'))
    parser.add_argument('--cache',type=Path,default=project_path('outputs/sharpa_tactile_three_class/20261004_160624'))
    parser.add_argument('--frame3-baseline',type=Path,default=project_path('outputs/sharpa_failure_relabel/20261006_185200_frame3'))
    parser.add_argument('--intervals',type=Path,default=ANNOTATION_ROOT/'intervals.jsonl')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    for name in ('source','cache','frame3_baseline','output'):setattr(args,name,getattr(args,name).resolve())
    if args.action=='run':
        prepare(args);fit(args)
        import torch
        torch.cuda.synchronize(args.device)
        sys.stdout.flush();sys.stderr.flush();os._exit(0)
    else:report(args)


if __name__=='__main__':main()
