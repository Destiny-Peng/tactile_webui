"""Fit causal thresholds to frozen Deform early-warning curves; never fit test."""
from .common import project_path, relative_path
import argparse
import csv
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from .common import ROOT, dump, sha
from .early_warning import HORIZONS, SEEDS, target, write_csv
from .calibrate_early_warning import evaluate, mean_std, event_scores, select_threshold


def coordinates(feature_source):
    # Read coordinates only: no encoder, tactile features or live relabeling.
    manifest = json.loads((feature_source/'dataset_manifest.json').read_text())
    samples = {split: [] for split in ('val', 'test')}
    for row in manifest['records']:
        if row['group'] != 'merged_align' or row['split'] not in samples:
            continue
        with np.load(feature_source/row['feature_path']) as z:
            samples[row['split']].append(dict(row, frames=z['video_frames'].copy()))
    assert all(len(ss) == 17 and sum(s['outcome'] == 2 for s in ss) == 3 for ss in samples.values())
    assert not ({s['rollout_id'] for s in samples['val']} & {s['rollout_id'] for s in samples['test']})
    return samples


def history_features(p, frames):
    """Past risk only, plus elapsed time; excludes current risk from threshold input."""
    n = len(p)
    previous = np.r_[.5, p[:-1]]
    prior_mean = np.r_[.5, np.cumsum(p[:-1])/np.arange(1, n)]
    prior_max = np.r_[.5, np.maximum.accumulate(p[:-1])]
    ema = np.empty(n); ema[0] = .5
    for i in range(1, n):
        ema[i] = .9*ema[i-1]+.1*p[i-1]
    change = np.r_[0., np.diff(previous)]
    elapsed = (frames-frames[0])/30
    return np.stack([previous, prior_mean, prior_max, ema, change,
                     np.log1p(elapsed)/np.log(121), elapsed/60], axis=1).astype(np.float32)


def operating_point(samples, margins, offsets, h):
    scores=[]; labels=[]
    for i, s in enumerate(samples):
        _, mask=target(s['frames'],s['key'],s['outcome']==2,h)
        scores.append(float(margins[offsets[i]:offsets[i+1]][mask].max()))
        labels.append(s['outcome']==2)
    scores=np.asarray(scores,np.float64); labels=np.asarray(labels)
    allowed=int(np.floor(.15*(~labels).sum()+1e-12))
    candidates=np.r_[np.unique(scores),np.nextafter(scores.max(),np.inf)]
    choices=[]
    for v in candidates:
        hits=scores>=v;fp=int((hits&~labels).sum());tp=int((hits&labels).sum())
        if fp<=allowed:choices.append((tp,float(v),fp))
    tp,v,fp=max(choices,key=lambda x:(x[0],x[1]))
    return dict(offset=v,validation_allowed_fp=allowed,validation_selected_fp=fp,
                validation_selected_tp=tp)


def train(args):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    samples=coordinates(args.features)
    # All held-out curves are archived before training, with hashes.
    arrays={};provenance=[]
    for h in HORIZONS:
        for seed in SEEDS:
            arrays[h,seed]={}
            for split in ('val','test'):
                path=args.source/'runs'/f'H{h}'/f'seed_{seed}'/(split+'_predictions.npz')
                with np.load(path) as z:p=z['probabilities'].copy();offsets=z['offsets'].copy()
                assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
                assert np.array_equal(offsets,np.r_[0,np.cumsum([len(s['frames']) for s in samples[split]])])
                arrays[h,seed][split]=(p,offsets)
                provenance.append(dict(path=str(relative_path(path)),sha256=sha(path)))
    for split,ss in samples.items():
        offsets=np.r_[0,np.cumsum([len(s['frames']) for s in ss])]
        np.savez_compressed(out/(split+'_coordinates.npz'),frames=np.concatenate([s['frames'] for s in ss]),offsets=offsets)
    dump(out/'samples.json',{split:[{k:v for k,v in s.items() if k!='frames'} for s in ss] for split,ss in samples.items()})
    dump(out/'score_provenance.json',provenance)
    # Stratified 3-fold OOF: each validation failure appears in exactly one fold.
    fold=np.empty(17,np.int64)
    for failure in (False,True):
        ids=[i for i,s in enumerate(samples['val']) if (s['outcome']==2)==failure]
        for j,i in enumerate(ids):fold[i]=j%3
    dump(out/'folds.json',[dict(rollout_id=s['rollout_id'],fold=int(fold[i]),failure=s['outcome']==2) for i,s in enumerate(samples['val'])])
    dump(out/'protocol.json',dict(source=str(relative_path(args.source)),features=str(relative_path(args.features)),
        modality='Deform only; original pretrained encoder and GRU heads frozen, original cached risk reused',
        horizons=list(HORIZONS),seeds=list(SEEDS),labels='original early-warning snapshot; no live3/4 relabeling',
        threshold='tau(t)=sigmoid(a(history_before_t,elapsed_t)+calibrated_offset); risk>=tau',
        inputs=['previous risk','past mean risk','past max risk','past EMA(.9)','previous risk difference','log1p elapsed seconds/log121','elapsed seconds/60'],
        current_risk_excluded_from_threshold_input=True,key_or_future_or_length_input=False,
        mlp='7->Linear16->ReLU->Linear1',gru='7->causal GRU16->Linear1',
        loss='balanced BCEWithLogits(logit(risk)-a(history),y), fold-train valid frame counts',
        training='AdamW lr.003 wd.001, 20 fixed epochs, full calibration-rollout batch, clip1, seed42-46; no early stopping',
        calibration='3-fold rollout-stratified OOF on original val17; event FPR<=15% maximize recall, tie higher offset; then refit all val17, freeze offset for test17',
        fixed_baseline='same cached risk, constant threshold selected on all original val at event FPR<=15%',
        caution='base checkpoints were epoch-selected on this original val; OOF is only independent of threshold-head fitting, not fully independent of base checkpoint selection',
        test='17 original untouched test rollouts, 14success/3failure; no test selection; H0 detection, H>0 early prediction'))

    class Head(nn.Module):
        def __init__(self,kind):
            super().__init__();self.kind=kind
            self.body=nn.GRU(7,16,batch_first=True) if kind=='gru' else nn.Sequential(nn.Linear(7,16),nn.ReLU())
            self.output=nn.Linear(16,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        def forward(self,x):
            hidden=self.body(x)[0] if self.kind=='gru' else self.body(x)
            return self.output(hidden).squeeze(-1)

    def batch(ss,p,offsets,h):
        length=max(len(s['frames']) for s in ss);count=len(ss)
        x=np.zeros((count,length,7),np.float32);logits=np.zeros((count,length),np.float32)
        y=np.zeros((count,length),np.float32);mask=np.zeros((count,length),bool);sizes=[]
        for i,s in enumerate(ss):
            local=p[offsets[i]:offsets[i+1]].astype(np.float64);n=len(local);sizes.append(n)
            x[i,:n]=history_features(local,s['frames']);clipped=np.clip(local,1e-6,1-1e-6)
            logits[i,:n]=np.log(clipped/(1-clipped))
            yy,valid=target(s['frames'],s['key'],s['outcome']==2,h);y[i,:n]=yy;mask[i,:n]=valid
        return [torch.as_tensor(a,device=args.device) for a in (x,logits,y,mask)],sizes

    def fit(kind,seed,b,ids):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        model=Head(kind).to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
        x,logit,y,mask=[v[ids] for v in b];pos=y[mask].sum();neg=mask.sum()-pos
        assert pos>0 and neg>0
        w1=mask.sum()/(2*pos);w0=mask.sum()/(2*neg);weight=torch.where(y==1,w1,w0)
        history=[]
        for epoch in range(20):
            model.train();optimizer.zero_grad();margin=logit-model(x)
            loss=(nn.functional.binary_cross_entropy_with_logits(margin,y,reduction='none')*weight)[mask].mean()
            assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();history.append(float(loss.detach()))
        return model,history

    def infer(model,b,sizes):
        model.eval()
        with torch.no_grad():a=model(b[0]).cpu().numpy();margin=(b[1]-model(b[0])).cpu().numpy()
        return np.concatenate([a[i,:n] for i,n in enumerate(sizes)]),np.concatenate([margin[i,:n] for i,n in enumerate(sizes)])

    results=[];events=[];count=0
    for h in HORIZONS:
        for seed in SEEDS:
            data={split:batch(samples[split],*arrays[h,seed][split],h) for split in ('val','test')}
            # Fixed threshold baseline uses raw risk with no clipping.
            vp,vo=arrays[h,seed]['val']
            ey,es=event_scores(samples['val'],vp,vo,h);chosen=select_threshold(ey,es,.15)
            baseline=dict(offset=chosen['threshold'],validation_allowed_fp=chosen['allowed_success_fp'],
                          validation_selected_fp=chosen['selected_success_fp'],validation_selected_tp=chosen['selected_failure_tp'])
            result=dict(horizon_frames=h,seed=seed,model='fixed',**baseline)
            for split in ('val','test'):
                f,e,rows=evaluate(samples[split],*arrays[h,seed][split],h,seed,split,baseline['offset'],.15)
                result[split]=f;result[split+'_event']=e
                events.extend([dict(model='fixed',**r) for r in rows])
            results.append(result)
            for kind in ('mlp','gru'):
                directory=out/'runs'/kind/f'H{h}'/f'seed_{seed}';directory.mkdir(parents=True)
                b,sizes=data['val'];oof=np.empty(len(vp),np.float64);fold_history=[]
                for k in range(3):
                    ids=np.flatnonzero(fold!=k).tolist();model,hist=fit(kind,seed+k*100,b,ids)
                    _,margin=infer(model,b,sizes)
                    for i in np.flatnonzero(fold==k):oof[vo[i]:vo[i+1]]=margin[vo[i]:vo[i+1]]
                    fold_history.append(hist);del model
                selection=operating_point(samples['val'],oof,vo,h)
                model,hist=fit(kind,seed,b,list(range(17)))
                torch.save(dict(state_dict=model.state_dict(),kind=kind,offset=selection['offset'],protocol='past-risk7 causal threshold',seed=seed,horizon_frames=h),directory/'best.pt')
                result=dict(horizon_frames=h,seed=seed,model=kind,**selection)
                result['oof_frame'],result['oof_event'],_=evaluate(samples['val'],oof,vo,h,seed,'val_oof',selection['offset'],.15)
                np.savez_compressed(directory/'oof_predictions.npz',margins=oof,offsets=vo)
                for split in ('val','test'):
                    bb,ss=data[split];a,margin=infer(model,bb,ss);p,offsets=arrays[h,seed][split]
                    assert np.isfinite(margin).all()
                    f,e,rows=evaluate(samples[split],margin,offsets,h,seed,split,selection['offset'],.15)
                    result[split]=f;result[split+'_event']=e
                    events.extend([dict(model=kind,**r) for r in rows])
                    tau=1/(1+np.exp(-np.clip(a.astype(np.float64)+selection['offset'],-700,700)))
                    np.savez_compressed(directory/(split+'_predictions.npz'),risk=p,threshold=tau,threshold_logit=a.astype(np.float64)+selection['offset'],margins=margin,offsets=offsets)
                dump(directory/'metrics.json',result);dump(directory/'history.json',dict(folds=fold_history,refit=hist))
                results.append(result);dump(out/'results.json',results)
                count+=1;print('DYNAMIC',count,50,kind,h,seed,'testFPR',round(result['test_event']['fpr'],4),'correctFirst',round(result['test_event']['correct_first_alarm_rate'],4),flush=True)
                del model
    write_csv(out/'event_predictions.csv',events)
    dump(out/'gpu_complete.json',dict(status='complete',threshold_heads=50,fit_models_including_OOF=200))
    torch.cuda.synchronize(args.device);sys.stdout.flush();sys.stderr.flush();os._exit(0)


def report(args):
    out=args.output;results=json.loads((out/'results.json').read_text());assert len(results)==75
    # Exactly reproduce the earlier fixed-threshold calibration, including >1
    # alarm-disabled sentinel when no validation failure can be recalled.
    coords=coordinates(args.features);fixed_events=[]
    for r in results:
        if r['model']!='fixed':continue
        h,seed=r['horizon_frames'],r['seed'];cache={}
        for split in ('val','test'):
            with np.load(args.source/'runs'/f'H{h}'/f'seed_{seed}'/(split+'_predictions.npz')) as z:
                cache[split]=(z['probabilities'].copy(),z['offsets'].copy())
        y,scores=event_scores(coords['val'],*cache['val'],h);selected=select_threshold(y,scores,.15)
        r['offset']=selected['threshold'];r['validation_selected_fp']=selected['selected_success_fp'];r['validation_selected_tp']=selected['selected_failure_tp']
        for split in ('val','test'):
            f,e,events=evaluate(coords[split],*cache[split],h,seed,split,r['offset'],.15)
            r[split]=f;r[split+'_event']=e;fixed_events.extend([dict(model='fixed',**v) for v in events])
    dump(out/'results.json',results)
    with (out/'event_predictions.csv').open() as f:
        original_events=[r for r in csv.DictReader(f) if r['model']!='fixed']
    write_csv(out/'event_predictions.csv',fixed_events+original_events)
    rows=[];confusion=[]
    for h in HORIZONS:
        for kind in ('fixed','mlp','gru'):
            rr=[r for r in results if r['horizon_frames']==h and r['model']==kind];assert len(rr)==5
            row=dict(horizon_frames=h,model=kind,seeds=5)
            for phase in ('test','test_event'):
                for key in ('balanced_accuracy','macro_f1','precision','recall','fpr','pr_auc','roc_auc','positive_band_detection_rate','correct_first_alarm_rate','undecided_rate','median_effective_early_warning_lead_seconds'):
                    if key not in rr[0][phase]:continue
                    mean,sd,n=mean_std([r[phase][key] for r in rr]);row[phase+'_'+key+'_mean']=mean;row[phase+'_'+key+'_std']=sd;row[phase+'_'+key+'_valid_seeds']=n
            rows.append(row)
            for phase in ('test','test_event'):
                cm=np.mean([r[phase]['confusion_matrix'] for r in rr],axis=0);norm=cm/cm.sum(1,keepdims=True)
                for a in (0,1):
                    for b in (0,1):confusion.append(dict(horizon_frames=h,model=kind,phase=phase,gt=a,pred=b,mean_count=cm[a,b],row_normalized=norm[a,b]))
    write_csv(out/'summary.csv',rows);write_csv(out/'confusion_normalized.csv',confusion)
    samples=json.loads((out/'samples.json').read_text())
    for split,ss in samples.items():
        with np.load(out/(split+'_coordinates.npz')) as z:
            frames=z['frames'].copy();offsets=z['offsets'].copy()
        for i,s in enumerate(ss):s['frames']=frames[offsets[i]:offsets[i+1]]
    # Replay all archived predictions; check fold-only calibration constraint.
    for r in results:
        if r['model']=='fixed':continue
        directory=out/'runs'/r['model']/f"H{r['horizon_frames']}"/f"seed_{r['seed']}"
        with np.load(directory/'oof_predictions.npz') as z:
            selected=operating_point(samples['val'],z['margins'],z['offsets'],r['horizon_frames'])
        assert selected['offset']==r['offset'] and selected['validation_selected_fp']<=2
        for split in ('val','test'):
            with np.load(directory/(split+'_predictions.npz')) as z:
                f,e,_=evaluate(samples[split],z['margins'],z['offsets'],r['horizon_frames'],r['seed'],split,r['offset'],.15)
            assert f['confusion_matrix']==r[split]['confusion_matrix'] and e['confusion_matrix']==r[split+'_event']['confusion_matrix']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for h in HORIZONS:
        for split in ('val','test'):
            ss=samples[split];fig,axes=plt.subplots(3,3,figsize=(14,9),squeeze=False)
            failures=[i for i,s in enumerate(ss) if s['outcome']==2]
            for i,index in enumerate(failures):
                s=ss[index]
                for j,kind in enumerate(('fixed','mlp','gru')):
                    risks=[];thresholds=[]
                    for seed in SEEDS:
                        r=next(r for r in results if r['horizon_frames']==h and r['seed']==seed and r['model']==kind)
                        if kind=='fixed':
                            with np.load(args.source/'runs'/f'H{h}'/f'seed_{seed}'/(split+'_predictions.npz')) as z:
                                a,b=z['offsets'][index:index+2];risk=z['probabilities'][a:b].copy()
                            threshold=np.full(len(risk),r['offset'])
                        else:
                            with np.load(out/'runs'/kind/f'H{h}'/f'seed_{seed}'/(split+'_predictions.npz')) as z:
                                a,b=z['offsets'][index:index+2];risk=z['risk'][a:b].copy();threshold=z['threshold'][a:b].copy()
                        risks.append(risk);thresholds.append(threshold)
                    ax=axes[i,j];x=(s['frames']-s['key'])/30
                    for values,label,color in ((risks,'base risk','C0'),(thresholds,'threshold','C1')):
                        values=np.stack(values);mean=values.mean(0);sd=values.std(0,ddof=1)
                        ax.plot(x,mean,label=label,color=color);ax.fill_between(x,np.maximum(0,mean-sd),np.minimum(1,mean+sd),alpha=.15,color=color)
                    if h>0:ax.axvspan(-h/30,0,color='green',alpha=.08)
                    ax.axvline(0,color='black',ls=':');ax.set_ylim(-.03,1.03);ax.set_title(kind+' / '+s['rollout_id'].split('mixedfail_')[-1]);ax.set_xlabel('Seconds relative to Key');ax.legend()
            fig.suptitle(f'{split} H={h}: frozen risk vs causal threshold, 5-seed mean ± SD');fig.tight_layout();fig.savefig(out/f'{split}_H{h}_threshold.png',dpi=160);plt.close(fig)
    lines=['# Early-warning：固定阈值 vs MLP/GRU 时变阈值','',
        '复用原Deform early-warning的H={0,8,15,30,45}、seeds42–46及原17val/17test。冻结原encoder和risk模型，训练50个threshold head（MLP/GRU各25）；不改变旧标注或原模型。本轮不是最新3/4 frame任务。','',
        '## 在线输入与阈值','',
        '当前risk为p(t)。阈值模型读取t之前的risk：previous、past mean/max、EMA(.9)、previous change，加当前距rollout第一个有效tick的elapsed seconds及log elapsed；总7D。MLP=7→16→ReLU→1；GRU=7→单向GRU16→1。模型输出a(t)，τ(t)=sigmoid(a(t)+offset)，p(t)≥τ(t)报警。当前risk用于比较，不能进入阈值网络输入。GRU逐rollout重置，禁止future risk、Key-relative time、最终outcome或总长度输入；Key仅用于GT和事后绘图。','',
        '训练用margin=logit(clip(p,1e-6,1−1e-6))−a(t)，对margin做balanced BCEWithLogits。H>0：最后Align failure end前[anchor−H,anchor)为1、此前为0、anchor后mask；success全0。H0 failure anchor后为1。权重按threshold训练fold有效帧N/(2Nc)重算。AdamW lr.003/wd.001，clip1，固定20epoch，完整rollout batch；无early stopping或test选epoch。MLP与GRU用相同7D信息，后者另保留历史状态。','',
        '## Calibration与split','',
        '原val17（14success/3failure）做3fold rollout-stratified交叉拟合：每fold持出1failure及4或5success；OOF每帧都由没见过该rollout的threshold head产生。在OOF每rollout最大margin上选offset，使event FPR≤15%（最多2条success FP），最大failure recall，平手更高offset。之后用全部val17重训同结构，固定OOF offset到原test17。固定阈值baseline在同一原val的max raw risk上按相同15%规则校准。','',
        '**限制：原base checkpoint曾用这17val做epoch selection，OOF仅隔离threshold head的拟合，并不能消除base选择对val的影响。OOF模型与全val refit模型的score尺度也可能漂移，因此不保证refit val/test FPR满足15%。** test仅用于最终评价，无test阈值/模型/H选择。5seeds不增加3条test failure的独立样本数。','',
        '## Test结果（%，5seed mean±样本SD）','',
        '| H frames | Decision | Frame BA | Macro F1 | Event precision | Recall | FPR | 正带检出 | 正确首报 | 有效lead秒 |','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(row,key,scale=100):
        m=row[key+'_mean'];s=row[key+'_std']
        return '—' if m is None else f'{m*scale:.2f} ± {s*scale:.2f}' if s is not None else f'{m*scale:.2f}'
    for row in rows:
        keys=('test_balanced_accuracy','test_macro_f1','test_event_precision','test_event_recall','test_event_fpr','test_event_positive_band_detection_rate','test_event_correct_first_alarm_rate')
        lines.append('| '+str(row['horizon_frames'])+' | '+row['model']+' | '+' | '.join(fmt(row,key) for key in keys)+' | '+fmt(row,'test_event_median_effective_early_warning_lead_seconds',1)+' |')
    lines+=['','BA为两类recall平均，Macro F1为两类F1平均。Event alarm=有效范围内至少一次检出，success检出即FP；positive-band detection与正确首次报警以全部failure rollout为分母。Lead仅对H>0且首次报警位于目标正带的failure计算；漏报/错误早报不填0。H0不是预警。事件precision/recall与frame指标分开。AP/ROC对margin计算，sigmoid(margin)只可视为决策score，不声称校准概率。','',
        '所有逐seed数值见summary.csv/results.json；逐rollout首报、lead及误报见event_predictions.csv。每个动态run保留OOF margin、refit val/test的原risk、τ(t)、margin及offset映射。val曲线为refit后的in-sample诊断，最终比较以test为准。图中Key-relative横轴只用于事后展示。','',
        '## Test概率曲线与时变阈值','']
    for h in HORIZONS:lines+= [f'![Test H{h}](test_H{h}_threshold.png)','']
    protocol=json.loads((out/'protocol.json').read_text())
    if protocol.get('training_regime')=='nested_loss_earlystop':
        lines=[line.replace('固定20epoch，完整rollout batch；无early stopping或test选epoch。', '内部训练max300/patience50，按rollout-held-out balanced BCE选择最佳epoch；完整rollout batch，无test选epoch。').replace('## Calibration与split','## Nested loss-based epoch selection、calibration与split') for line in lines]
        lines=[line.replace('OOF每帧都由没见过该rollout的threshold head产生。', '每个outer训练子集内部再固定持出1failure+3success用于loss早停，内部train/val与outer holdout均rollout分离；按最低内部val BCE选择epoch，用该epoch数在全部outer train重拟合后才产生OOF。OOF每帧的rollout既未参与该head梯度训练，也未参与epoch选择。').replace('之后用全部val17重训同结构，固定OOF offset到原test17。', '最终重拟合全部val17，训练轮数取三个内部loss-selected最佳epoch的中位数（不超过300），固定OOF offset到原test17。重拟合过程使用已选轮数，不会把自身训练loss冒充validation loss。') for line in lines]
        lines+=['','本轮较旧20-update实验同时增加了优化预算与内部loss-selected epoch流程；不能将全部差异单独归因于epoch数。内部val loss沿用其训练子集计算的类别权重，验证集不另拟合权重。','']
    (out/'README.md').write_text('\n'.join(lines))
    dump(out/'verification.json',dict(status='PASS',models=50,fold_models=150,inner_loss_models=150 if protocol.get('training_regime')=='nested_loss_earlystop' else 0,runs=75,OOF_rollout_disjoint=True,
        no_key_or_future_inputs=True,test_not_used_for_fitting_or_selection=True,prediction_metrics_replayed=True))
    shutil.copyfile(Path(__file__),out/'dynamic_threshold.py')
    dump(out/'completion.json',dict(status='complete',completed_at=datetime.datetime.now().astimezone().isoformat(),code_sha256=sha(Path(__file__))))
    destination=project_path('WeeklySummary/10.5/early_warning_dynamic_threshold', out.name);destination.mkdir(parents=True,exist_ok=False)
    checks={}
    for path in out.iterdir():
        if path.is_file():shutil.copyfile(path,destination/path.name);assert sha(path)==sha(destination/path.name);checks[path.name]=sha(path)
    dump(destination/'copy_sha256.json',checks)
    text=(out/'README.md').read_text()
    for h in HORIZONS:text=text.replace(f'](test_H{h}_threshold.png)',f'](early_warning_dynamic_threshold/{out.name}/test_H{h}_threshold.png)')
    (project_path('WeeklySummary/10.5/10.5early_warning_dynamic_threshold.md')).write_text(text)
    link=f'early_warning_dynamic_threshold/{out.name}/README.md'
    for filename in ('10.5criticalreward.md','10.5.md'):
        with (project_path('WeeklySummary/10.5', filename)).open('a') as f:f.write(f'\n\n## Learned online threshold (Deform)\n\nMLP/causal GRU时变阈值，5H×5seeds，原val rollout交叉拟合校准，固定原test评价；原risk frozen。[报告]({link})。\n')
    print('REPORT_COMPLETE',50,'learned thresholds',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('run','report'))
    p.add_argument('--source',type=Path,default=project_path('outputs/sharpa_early_warning/20261006_001500'))
    p.add_argument('--features',type=Path,default=project_path('outputs/sharpa_gaussian_online_data/20261005_182000'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:1')
    args=p.parse_args()
    for key in ('source','features','output'):setattr(args,key,getattr(args,key).resolve())
    train(args) if args.action=='run' else report(args)


if __name__=='__main__':main()
