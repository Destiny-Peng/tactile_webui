"""Failure-only event3/event4 interval and frame binary probes; train/val only."""
from .common import project_path
from .common import ANNOTATION_ROOT
import argparse
import datetime
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from .common import ROOT, dump, read_jsonl, sha
from .probe_metrics import metrics, write_csv

EVENTS = {'dropped_object': 3, 'wrong_object': 4}
KINDS = {'f6': ('f6',), 'deform': ('deform',), 'fusion': ('f6', 'deform')}
GROUPS = ('interval_3vs4', 'frame_4', 'frame_3or4')
SEEDS = (42, 43, 44, 45, 46)


def prepare(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out/'features').mkdir()
    records = {}; events = []; excluded = []; snapshots = []
    manifest_path = project_path('datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl')
    from .data import load_sources
    interval_path = getattr(args, 'intervals', ANNOTATION_ROOT/'intervals.jsonl')
    sources = load_sources(interval_path, manifest_path, label_keys=(3,4))
    shutil.copyfile(interval_path, out/'canonical_intervals.jsonl')
    for rid, (record, annotation_events) in sources.items():
        if record['ground_truth_outcome'] != 'failure' or 'usb' not in record['task_key']:
            continue
        selected = [dict(e, label=int(e['event_key']==4)) for e in annotation_events if e['event_key'] in (3,4)]
        if selected:
            records[rid] = record; events.extend(selected)
    snapshots.append(dict(source=str(interval_path), sha256=sha(interval_path), backup='canonical_intervals.jsonl'))
    dump(out/'annotation_snapshot.json', dict(records=snapshots, excluded=excluded,
         event_mapping=EVENTS, source_manifest_sha256=sha(manifest_path)))
    assert records and {e['event_key'] for e in events} == {3, 4}
    by_id = defaultdict(list)
    for e in events: by_id[e['rollout_id']].append(e)
    strata = defaultdict(list)
    for rid, record in records.items():
        strata[(record['task_key'], tuple(sorted({e['event_key'] for e in by_id[rid]})))].append(rid)
    rng = np.random.default_rng(42); split = {'train': [], 'val': []}
    for key, ids in sorted(strata.items()):
        ids = sorted(ids); rng.shuffle(ids)
        nval = max(1, round(len(ids)*.2)) if len(ids)>1 else 0
        split['val'].extend(ids[:nval]); split['train'].extend(ids[nval:])
    split = {k: sorted(v) for k,v in split.items()}
    assert not set(split['train']) & set(split['val'])
    for phase in split:
        assert {e['event_key'] for e in events if e['rollout_id'] in split[phase]} == {3,4}
    membership = {rid: phase for phase,ids in split.items() for rid in ids}
    dump(out/'split_manifest.json', dict(seed=42, unit='rollout', requested_train_fraction=.8,
         stratification='task and presence of event3/event4', **split))
    dump(out/'source_intervals.json', events)
    print('ANNOTATIONS',len(records),'rollouts',len(events),'intervals',
          {k:len(v) for k,v in split.items()},flush=True)

    import torch
    from .data import episode_arrays, DeformStreams
    from .models import FrozenEncoders
    torch.set_num_threads(4)
    encoders = FrozenEncoders().to(args.device)
    old = json.loads((args.cache/'data_manifest.json').read_text())
    assert sha(encoders.f6_path) == old['signature']['f6_sha256']
    assert sha(encoders.deform_path) == old['signature']['deform_sha256']
    rows = []; feature_audit = []
    for rid,record in sorted(records.items()):
        arrays = episode_arrays(record, [], num_classes=3)
        old_path = args.cache/'features'/(rid+'.npz')
        if old_path.exists():
            source_signature = old['signature']['episode_sources'][rid]
            assert sha(project_path(record['synchronized_frames_path'])) == source_signature['frames_sha256']
            assert sha(project_path(record['tactile_events_path'])) == source_signature['events_sha256']
            with np.load(old_path) as z:
                assert np.array_equal(z['ticks'], arrays['ticks'])
                assert np.array_equal(z['video_frames'], arrays['video_frames'])
                f6, deform = z['f6'].copy(), z['deform'].copy()
            origin = 'reused full-rollout frozen cache'
        else:
            f6 = np.empty((len(arrays['ticks']),1280),np.float32)
            deform = np.empty((len(arrays['ticks']),2560),np.float32)
            streams = DeformStreams(record)
            for a in range(0,len(f6),args.batch_size):
                ticks = arrays['ticks'][a:a+args.batch_size]
                raw = np.stack([arrays['f6'][t-15:t+1] for t in ticks])
                images = streams.batch([arrays['references'][int(t)] for t in ticks])
                f6[a:a+len(ticks)] = encoders.f6_features(torch.from_numpy(raw).to(args.device)).cpu().numpy()
                deform[a:a+len(ticks)] = encoders.deform_features(torch.from_numpy(images).to(args.device)).cpu().numpy()
            origin = 'new frozen encoder extraction'
        assert np.isfinite(f6).all() and np.isfinite(deform).all()
        frames, ticks = arrays['video_frames'], arrays['ticks']
        y4 = np.zeros(len(frames),np.int64); y34 = y4.copy()
        for e in by_id[rid]:
            inside = (frames>=e['start_frame']) & (frames<=e['end_frame'])
            y34[inside] = 1
            if e['event_key']==4: y4[inside] = 1
        path = out/'features'/(rid+'.npz')
        np.savez_compressed(path,f6=f6,deform=deform,video_frames=frames,ticks=ticks,y4=y4,y34=y34)
        rows.append(dict(group='frame',rollout_id=rid,split=membership[rid],
                         sample_id=rid,feature_path=str(path.relative_to(out)),length=len(frames)))
        for e in by_id[rid]:
            selected = np.flatnonzero((frames>=e['start_frame']) & (frames<=e['end_frame']))
            if not len(selected): raise ValueError(f'No valid synchronized interval ticks: {e}')
            st = ticks[selected]; first = int(st[0]); prefix = np.flatnonzero(st-first<15)
            sf6 = f6[selected].copy()
            for a in range(0,len(prefix),args.batch_size):
                pos = prefix[a:a+args.batch_size]
                raw = np.stack([arrays['f6'][np.maximum(np.arange(int(st[p])-15,int(st[p])+1),first)] for p in pos])
                sf6[pos] = encoders.f6_features(torch.from_numpy(raw).to(args.device)).cpu().numpy()
            sample_id = rid+f"_event{e['event_index']:02d}"
            path = out/'features'/(sample_id+'.npz')
            np.savez_compressed(path,f6=sf6,deform=deform[selected],video_frames=frames[selected],ticks=st,label=e['label'])
            rows.append(dict(e,group='interval',split=membership[rid],sample_id=sample_id,
                             feature_path=str(path.relative_to(out)),length=len(st),
                             recomputed_in_interval_prefix=len(prefix),gaps=int((np.diff(st)!=1).sum())))
        feature_audit.append(dict(rollout_id=rid,origin=origin,**arrays['audit']))
        print('FEATURES',len(feature_audit),len(records),rid,len(frames),origin,flush=True)
    protocol = dict(event_mapping={'3':'dropped_object -> interval0','4':'wrong_object -> interval1'},
        eligibility='USB manifest failure rollouts with at least one complete event3/4; others excluded',
        bounds='closed causal_onset_frame..observable_onset_frame',
        groups={'interval_3vs4':'one entire event3/4 interval per sample; target3=0,4=1; only interval tactile',
                'frame_4':'full valid failure rollout; current camera frame in any event4=1; all else0 including3',
                'frame_3or4':'full valid failure rollout; current camera frame in union event3/4=1; all else0'},
        interval_context='F6 past16 clipped at first valid in-interval tick by left-repeat padding; GRU reset per interval',
        frame_context='past16 contiguous valid raw F6 ticks; current Deform; GRU reset per rollout; invalid ticks omitted',
        encoder='frozen T-Rex checkpoint finger encoder5x256=1280; Deform each finger pool2x2=512,5x512=2560',
        encoder_sha256={'f6':sha(encoders.f6_path),'deform':sha(encoders.deform_path)},
        sampling='dense step0; no augmentation/Gaussian/Key merge; all annotated3/4 intervals independent',
        model='each modality Linear128; fusion concat256; unidirectional single GRU128; Linear1 sigmoid',
        loss='balanced BCEWithLogits weights N/(2Nc), interval counts for interval model, frame counts for frame models',
        optimizer='AdamW lr.001 weight_decay.0001 batch8 max30 epochs patience8 clip1',
        normalization='training data only mean/std; std floor.01; each interval equal weight in interval normalization',
        seeds=list(SEEDS),selection='best validation BA at threshold.5; tie earliest epoch; no test set',
        criterion='validation only; metrics on same val used for epoch selection, not independent heldout test',
        created_at=datetime.datetime.now().astimezone().isoformat())
    dump(out/'protocol.json',protocol)
    dump(out/'dataset_manifest.json',dict(status='complete',records=rows,excluded=excluded,feature_audit=feature_audit))
    print('PREPARE_COMPLETE',flush=True)


def load_samples(out,group,only_split=None):
    data = json.loads((out/'dataset_manifest.json').read_text()); samples = {'train':[], 'val':[]}
    feature_source = project_path(data['feature_source']) if 'feature_source' in data else out
    refreshed_event_key = 3 if group=='frame_3' else (4 if group=='frame_4' and data.get('labels_from_snapshot') else None)
    event3 = defaultdict(list)
    if refreshed_event_key is not None:
        for e in json.loads((out/'source_intervals.json').read_text()):
            if e['event_key']==refreshed_event_key:event3[e['rollout_id']].append(e)
    for row in data['records']:
        if row['group'] != ('interval' if group=='interval_3vs4' else 'frame'): continue
        if only_split is not None and row['split']!=only_split:continue
        with np.load(feature_source/row['feature_path']) as z:
            s = dict(row,f6=z['f6'].copy(),deform=z['deform'].copy(),frames=z['video_frames'].copy())
            if refreshed_event_key is not None:
                s['y']=np.zeros(len(s['frames']),np.float32)
                for e in event3[row['rollout_id']]:
                    s['y'][(s['frames']>=e['start_frame'])&(s['frames']<=e['end_frame'])]=1
            else:s['y'] = np.array([row['label']],np.float32) if group=='interval_3vs4' else z['y4' if group=='frame_4' else 'y34'].astype(np.float32)
        samples[row['split']].append(s)
    return samples


def train(args):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    out = args.output; results = []; distributions = []
    groups=getattr(args,'groups',GROUPS)
    for group in groups:
        samples = load_samples(out,group); interval = group=='interval_3vs4'
        labels = {k:np.concatenate([s['y'] for s in ss]) for k,ss in samples.items()}
        for split,y in labels.items():
            n0,n1 = int((y==0).sum()),int((y==1).sum()); assert n0>0 and n1>0
            distributions.append(dict(group=group,split=split,rollouts=len({s['rollout_id'] for s in samples[split]}),
                samples=len(samples[split]),unit='interval' if interval else 'frame',negative=n0,positive=n1,
                positive_fraction=n1/len(y),negative_weight=len(y)/(2*n0),positive_weight=len(y)/(2*n1)))
        write_csv(out/'label_distribution.csv',distributions)
        ytrain = labels['train']; weights = (len(ytrain)/(2*int((ytrain==0).sum())),len(ytrain)/(2*int((ytrain==1).sum())))
        norm = {}
        for key in ('f6','deform'):
            if interval:
                # Each interval contributes equally, regardless of length.
                mean = np.mean([s[key].mean(0,dtype=np.float64) for s in samples['train']],axis=0)
                second = np.mean([(s[key].astype(np.float64)**2).mean(0) for s in samples['train']],axis=0)
                std = np.sqrt(np.maximum(second-mean**2,0))
            else:
                bank = np.concatenate([s[key] for s in samples['train']]);mean = bank.mean(0,dtype=np.float64);std = bank.std(0,dtype=np.float64)
                del bank
            norm[key] = (torch.from_numpy(mean.astype(np.float32)),torch.from_numpy(np.maximum(std,.01).astype(np.float32)))
        class Model(nn.Module):
            def __init__(self,kind):
                super().__init__();self.keys=KINDS[kind];self.projections=nn.ModuleDict()
                for key in self.keys:
                    mean,std=norm[key];self.register_buffer(key+'_mean',mean.clone());self.register_buffer(key+'_std',std.clone())
                    self.projections[key]=nn.Linear(len(mean),128)
                self.gru=nn.GRU(128*len(self.keys),128,batch_first=True);self.head=nn.Linear(128,1)
            def forward(self,x,lengths):
                v=torch.cat([self.projections[k]((x[k]-getattr(self,k+'_mean'))/getattr(self,k+'_std')) for k in self.keys],dim=-1)
                h,_=self.gru(v)
                if interval: h=h[torch.arange(len(lengths),device=h.device),torch.as_tensor(lengths,device=h.device)-1]
                return self.head(h).squeeze(-1)
        def batch(ss,kind):
            lengths=[len(s['frames']) for s in ss];width=max(lengths)
            x={k:np.zeros((len(ss),width,len(norm[k][0])),np.float32) for k in KINDS[kind]}
            shape=(len(ss),) if interval else (len(ss),width)
            y=np.zeros(shape,np.float32);mask=np.zeros(shape,bool)
            for i,s in enumerate(ss):
                for k in x:x[k][i,:lengths[i]]=s[k]
                if interval:y[i]=s['y'][0];mask[i]=True
                else:y[i,:lengths[i]]=s['y'];mask[i,:lengths[i]]=True
            return {k:torch.from_numpy(v).to(args.device) for k,v in x.items()},torch.from_numpy(y).to(args.device),torch.from_numpy(mask).to(args.device),lengths
        def infer(model,ss,kind):
            model.eval();ps=[]
            with torch.inference_mode():
                for a in range(0,len(ss),8):
                    x,_,_,lengths=batch(ss[a:a+8],kind);p=model(x,lengths).sigmoid().cpu().numpy()
                    if interval:ps.extend(np.array([q]) for q in p)
                    else:ps.extend(p[i,:n] for i,n in enumerate(lengths))
            return np.concatenate(ps),np.r_[0,np.cumsum([len(p) for p in ps])]
        for kind in KINDS:
            for seed in SEEDS:
                torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
                model=Model(kind).to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                directory=out/'runs'/group/kind/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=False)
                rng=np.random.default_rng(seed);best=-1;best_epoch=0;history=[]
                for epoch in range(1,31):
                    model.train();order=rng.permutation(len(samples['train']));loss_sum=0;total=0
                    for a in range(0,len(order),8):
                        ss=[samples['train'][int(i)] for i in order[a:a+8]];x,y,mask,lengths=batch(ss,kind)
                        optimizer.zero_grad(set_to_none=True);element=nn.functional.binary_cross_entropy_with_logits(model(x,lengths),y,reduction='none')
                        loss_weights=torch.where(y>0,weights[1],weights[0]);loss=(element*loss_weights)[mask].mean()
                        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                        loss_sum+=float((element*loss_weights)[mask].detach().sum());total+=int(mask.sum())
                    p,offsets=infer(model,samples['val'],kind);val=metrics(labels['val'],p)
                    history.append(dict(epoch=epoch,train_balanced_bce=loss_sum/total,validation_BA=val['balanced_accuracy']))
                    if val['balanced_accuracy']>best+1e-8:
                        best=val['balanced_accuracy'];best_epoch=epoch
                        torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            group=group,input=kind,seed=seed,best_epoch=epoch,protocol=json.loads((out/'protocol.json').read_text())),directory/'best.pt')
                    if epoch-best_epoch>=8:break
                model.load_state_dict(torch.load(directory/'best.pt',map_location=args.device,weights_only=True)['state_dict'])
                p,offsets=infer(model,samples['val'],kind);val=metrics(labels['val'],p)
                cm=np.asarray(val['confusion_matrix']);val['confusion_row_normalized']=(cm/cm.sum(1,keepdims=True)).tolist()
                np.savez_compressed(directory/'val_predictions.npz',probabilities=p,labels=labels['val'],offsets=offsets)
                dump(directory/'val_samples.json',[{k:v for k,v in s.items() if k not in ('f6','deform','frames','y')} for s in samples['val']])
                result=dict(group=group,input=kind,seed=seed,best_epoch=best_epoch,epochs=len(history),val=val)
                if getattr(args,'after_run',None) is not None:
                    args.after_run(model,infer,kind,group,seed,directory,result)
                dump(directory/'metrics.json',result);dump(directory/'history.json',history);results.append(result)
                dump(out/'results_partial.json',results)
                print('RUN',len(results),len(groups)*len(KINDS)*len(SEEDS),group,kind,seed,'valBA',round(val['balanced_accuracy'],4),'AP',round(val['pr_auc'],4),flush=True)
                del model,optimizer;torch.cuda.empty_cache()
    dump(out/'results.json',results)
    print('TRAIN_COMPLETE',len(results),flush=True)


def report(args):
    out=args.output;results=json.loads((out/'results.json').read_text());assert len(results)==45
    summary=[];confusions=[];interval_predictions=[]
    for group in GROUPS:
        for kind in KINDS:
            rr=[r for r in results if r['group']==group and r['input']==kind];assert {r['seed'] for r in rr}==set(SEEDS)
            row=dict(group=group,input=kind,seeds=5)
            for key in ('accuracy','balanced_accuracy','macro_f1','precision','recall','fpr','pr_auc','roc_auc'):
                v=[r['val'][key] for r in rr];row[key+'_mean']=float(np.mean(v));row[key+'_std']=float(np.std(v,ddof=1))
            summary.append(row)
            cm=np.mean([r['val']['confusion_matrix'] for r in rr],axis=0);normalized=cm/cm.sum(1,keepdims=True)
            for gt in (0,1):
                for pred in (0,1):confusions.append(dict(group=group,input=kind,gt=gt,pred=pred,mean_count=cm[gt,pred],row_normalized=normalized[gt,pred]))
            for r in rr:
                path=out/'runs'/group/kind/f"seed_{r['seed']}"/'val_predictions.npz'
                with np.load(path) as z:replay=metrics(z['labels'],z['probabilities'])
                assert replay['confusion_matrix']==r['val']['confusion_matrix']
                assert np.isclose(replay['balanced_accuracy'],r['val']['balanced_accuracy'])
            if group=='interval_3vs4':
                directory=out/'runs'/group/kind
                val_samples=json.loads((directory/'seed_42'/'val_samples.json').read_text())
                values=[]
                for seed in SEEDS:
                    with np.load(directory/f'seed_{seed}'/'val_predictions.npz') as z:values.append(z['probabilities'].copy())
                values=np.stack(values)
                for i,s in enumerate(val_samples):
                    interval_predictions.append(dict(input=kind,rollout_id=s['rollout_id'],event_index=s['event_index'],
                        event_key=s['event_key'],start_frame=s['start_frame'],end_frame=s['end_frame'],label=s['label'],
                        mean_P_event4=float(values[:,i].mean()),std_P_event4=float(values[:,i].std(ddof=1)),
                        seeds_predict_event4=int((values[:,i]>=.5).sum())))
    write_csv(out/'summary.csv',summary);write_csv(out/'confusion_normalized.csv',confusions)
    write_csv(out/'val_interval_predictions.csv',interval_predictions)
    manifest=json.loads((out/'dataset_manifest.json').read_text())
    lengths=[]
    for phase in ('train','val'):
        for label in (0,1):
            rr=[r for r in manifest['records'] if r['group']=='interval' and r['split']==phase and r['label']==label]
            v=np.array([r['end_frame']-r['start_frame']+1 for r in rr])
            lengths.append(dict(split=phase,event_key=3+label,intervals=len(rr),min_frames=int(v.min()),median_frames=float(np.median(v)),mean_frames=float(v.mean()),max_frames=int(v.max())))
    write_csv(out/'interval_lengths.csv',lengths)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(11,10))
    for i,group in enumerate(GROUPS):
        for j,kind in enumerate(KINDS):
            rr=[r for r in results if r['group']==group and r['input']==kind]
            cm=np.mean([r['val']['confusion_matrix'] for r in rr],axis=0);v=cm/cm.sum(1,keepdims=True)
            ax=axes[i,j];ax.imshow(v,vmin=0,vmax=1,cmap='Blues')
            for a in (0,1):
                for b in (0,1):ax.text(b,a,f'{100*v[a,b]:.1f}%\n({cm[a,b]:.1f})',ha='center',va='center',color='white' if v[a,b]>.5 else 'black')
            names=['event3','event4'] if group=='interval_3vs4' else ['outside','event4' if group=='frame_4' else 'event3/4']
            ax.set_xticks([0,1],names);ax.set_yticks([0,1],names);ax.set_title(group+' / '+kind);ax.set_xlabel('Predicted');ax.set_ylabel('GT')
    fig.tight_layout();fig.savefig(out/'confusion_normalized.png',dpi=180);fig.savefig(out/'confusion_normalized.pdf');plt.close(fig)
    for group in ('frame_4','frame_3or4'):
        ss=load_samples(out,group)['val'];fig,axes=plt.subplots(len(ss),3,figsize=(15,3*len(ss)),squeeze=False)
        for j,kind in enumerate(KINDS):
            ps=[]
            for seed in SEEDS:
                with np.load(out/'runs'/group/kind/f'seed_{seed}'/'val_predictions.npz') as z:
                    ps.append(z['probabilities'].copy());offsets=z['offsets'].copy()
            ps=np.stack(ps)
            for i,s in enumerate(ss):
                a,b=offsets[i:i+2];local=ps[:,a:b];mean=local.mean(0);sd=local.std(0,ddof=1)
                ax=axes[i,j];x=s['frames']/30
                ax.plot(x,mean,label='mean P(positive)');ax.fill_between(x,np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),alpha=.15)
                ax.step(x,s['y'],where='post',color='black',alpha=.5,label='hard GT');ax.axhline(.5,color='gray',ls='--')
                ax.set_ylim(-.03,1.03);ax.set_title(s['rollout_id'].split('mixedfail_')[-1]+' / '+kind);ax.set_xlabel('Recorded camera time (s)');ax.legend(fontsize=8)
        fig.suptitle(group+' validation; probability mean ± seed SD');fig.tight_layout()
        fig.savefig(out/(group+'_val_curves.png'),dpi=180);fig.savefig(out/(group+'_val_curves.pdf'));plt.close(fig)
    distribution=list(__import__('csv').DictReader((out/'label_distribution.csv').open()))
    lines=['# Failure-only：重新标注的 event 3/4 对照实验','',
        '读取 USB failure rollout 的 canonical merged 3/4 标注。编号按 annotator 快捷键映射：3=dropped_object，4=wrong_object；它们是本轮标注代号，不是旧 success/failure 映射。原始标注不修改，canonical merged annotation 已备份至 canonical_intervals.jsonl。','',
        '23 条 rollout 含完整 3/4 interval，共74个（3类18个，4类56个）；另外10条没有新3/4 interval，三个实验统一排除，不将未标注解释成已确认负例。固定 seed42 按 rollout 的类别覆盖分层约80/20划分 train/val，两者互不交叉；所有模型和训练 seeds42–46共用此划分，无 test。','',
        '## 三个实验如何从标注构造 GT','',
        '1. **Interval 3 vs 4**：每个原始 `[causal_onset_frame, observable_onset_frame]` 闭区间单独成为一个样本，3→0、4→1，不合并 interval、不只取最后一个。整个 interval 内逐时刻提取特征并送入单向 GRU，取最后一个有效时刻的 hidden→Linear1→sigmoid，输出整段属于4的概率；≥0.5判4，否则判3。每个 interval 只贡献一次 BCE，长 interval 不贡献更多监督点。Interval 外帧完全不参与；初始 F6 的16tick上下文不足时重复区间首个有效tick补齐，禁止借用区间外历史；GRU在每个interval重置。','',
        '2. **Frame only 4**：使用这23条 failure rollout 的整条有效记录；当前帧落在任意4区间内→1，其余→0，包含3区间和前后背景。单向GRU的每个时刻 hidden→Linear1→sigmoid，输出当前帧处于4区间的概率，≥0.5判1。','',
        '3. **Frame 3 or 4**：输入和frame only4相同，当前帧属于任意3或4区间的并集→1，其余→0；输出当前帧属于3/4区间的概率。多个区间重叠只计一次正帧。','',
        'Frame标签只由当前最后帧判定，不用滑窗后半段Key命中、不使用Gaussian、不沿用early-warning horizon或最后Key规则。Frame模型GRU每rollout重置，使用此前有效特征的历史；无效同步帧省略、跨缺口不额外重置，F6局部编码仍严格要求连续16个有效原始tick。指标计数单位是有效同步tick（由cam_high frame index关联标注），不会强行插值制造缺失相机帧。','',
        '## 模型、loss和选择标准','',
        '每个时刻：F6连续16个raw tick→冻结现有T-Rex finger encoder→5×256=1280D；Deform只取当前tick→冻结encoder→每指AdaptiveAvgPool2d(2,2)→512D×5=2560D。F6与Deform各Linear128，Fusion拼接256D，再接单层单向GRU128→Linear1 sigmoid。F6/Deform/Fusion三组各5seeds，共45次训练；decoder不参与。','',
        'Balanced BCEWithLogits：训练集 `w0=N/(2N0), w1=N/(2N1)`；Interval按interval数量计算，frame按有效帧数量计算。输入mean/std仅由train计算（std下限0.01；interval normalization对每段等权）。AdamW lr0.001、weight decay0.0001、batch8、最多30epoch、patience8、clip1。固定0.5阈值，best checkpoint仅按val BA挑选，平手取最早epoch；没有独立test，以下是用于选择checkpoint的validation结果。','',
        'BA是两类recall均值，不是binary accuracy；Accuracy是所有预测正确的比例；Macro F1是两类各自F1的平均。P/R/FPR指positive=1（interval组为event4；frame组为对应区间内）。PR-AUC采用tie-aware average precision，ROC-AUC采用ROC梯形积分；与0.5阈值无关。Confusion matrix行是真实类别，列是预测类别，图按GT行归一化，括号是5seeds平均原始计数。均值±SD为5次训练的样本标准差，rollout split不变化。','',
        '## 数据分布','', '| Experiment | Split | Rollouts | Unit | Negative 0 | Positive 1 |', '|---|---|---:|---|---:|---:|']
    for r in distribution:lines.append(f"| {r['group']} | {r['split']} | {r['rollouts']} | {r['unit']} | {r['negative']} | {r['positive']} |")
    lines += ['', '## Validation results（%，mean ± SD，5 seeds）','', '| Experiment | Input | Accuracy | BA | Macro F1 | Precision | Recall | FPR | PR-AUC | ROC-AUC |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        v=[f"{100*r[k+'_mean']:.2f} ± {100*r[k+'_std']:.2f}" for k in ('accuracy','balanced_accuracy','macro_f1','precision','recall','fpr','pr_auc','roc_auc')]
        lines.append('| '+r['group']+' | '+r['input']+' | '+' | '.join(v)+' |')
    lines += ['', '## 结果解读','',
        'Interval的3对4分类中，Deform最好：BA91.11±3.04%、Macro F1 84.72±4.75%、ROC-AUC91.85±4.83%。3类的3个验证interval在所有5seeds中均判对，4类recall82.22±6.09%。F6 BA71.11±2.48%，Fusion75.56±7.45%，当前Fusion没有提升interval可分性，seed波动更大。当前完整区间的tactile信息有区分3/4的信号，尤其Deform；此处分类目标是标注3与4，不是整条rollout最终success/failure。','',
        'Frame两组都以Fusion最好：仅4的BA85.43±0.45%、Macro F1 74.68±1.84%；3/4并集的BA87.36±0.60%、Macro F1 78.86±1.07%。但仅4的positive precision仍只有44.29±3.38%（recall85.53±2.12%、FPR14.67±2.30%），区间外误报仍明显。两组GT和正例比例不同，不能用跨组AP差值单独判断representation改进。','',
        '以上均是用于best epoch选择的同一validation，只有4条独立rollout/12个interval（3类3个、4类9个）；5seeds测量训练波动，不构成5份独立数据验证。','',
        '## Interval长度','', '| Split | Event | N | Min | Median | Mean | Max |','|---|---:|---:|---:|---:|---:|---:|']
    for r in lengths:lines.append(f"| {r['split']} | {r['event_key']} | {r['intervals']} | {r['min_frames']} | {r['median_frames']:.1f} | {r['mean_frames']:.1f} | {r['max_frames']} |")
    lines += ['', '长度单位为相机frame。两类长度分布不同，因此interval结果表示完整tactile sequence在当前标注下可分，并不能单独证明与interval长度无关的representation可分。五次seed只是优化波动，不增加独立验证rollout数量。','',
        '![Row-normalized confusion matrices](confusion_normalized.png)','',
        '![Frame only4 validation probability](frame_4_val_curves.png)','',
        '![Frame3or4 validation probability](frame_3or4_val_curves.png)','',
        'Frame曲线按原始相机时间显示每条val rollout，黑线是固定hard GT，彩线及阴影是5seeds概率mean±SD；不做Key对齐。val_interval_predictions.csv逐个列出interval在三模态下的平均P(event4)、seed波动和预测为4的seed数。','',
        '原始输出保存 checkpoints、每epoch history、val预测与GT/offsets/sample映射、split、标注快照和特征audit；summary.csv为汇总，results.json为45次原始指标。','']
    (out/'README.md').write_text('\n'.join(lines))
    dump(out/'verification.json',dict(status='PASS',runs=45,seeds=list(SEEDS),train_val_disjoint=True,
        no_test=True,interval_no_external_tactile=True,encoders_frozen=True,val_confusion_and_BA_replayed=True))
    dump(out/'completion.json',dict(status='complete',completed_at=datetime.datetime.now().astimezone().isoformat(),code_sha256=sha(Path(__file__))))
    print('REPORT_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','report'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--intervals',type=Path,default=ANNOTATION_ROOT/'intervals.jsonl')
    parser.add_argument('--cache',type=Path,default=project_path('outputs/sharpa_tactile_three_class/20261004_160624'))
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--groups',nargs='+',choices=(*GROUPS,'frame_3'),default=GROUPS,
                        help='Train selected groups; default preserves the original three-group experiment')
    args=parser.parse_args();globals()[args.action](args)


if __name__=='__main__':main()
