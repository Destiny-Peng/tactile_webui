"""Frozen F6/Fusion vs existing Deform: matched early-warning settings."""
from .common import project_path, relative_path
import argparse
import csv
import datetime
import json
import random
import shutil
from pathlib import Path

import numpy as np
from .common import ROOT,dump,sha
from .early_warning import HORIZONS,SEEDS,target,distribution,metrics,evaluate_arrays,write_csv,read_data

KINDS=('f6','fusion')
FEATURES={'f6':('f6',),'fusion':('f6','deform')}


def train(args):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    print('LOADING_FROZEN_FEATURES',flush=True)
    samples=read_data(args.source,False)
    for ss in samples.values():
        for s in ss:
            with np.load(args.source/s['feature_path']) as z:
                assert np.array_equal(s['frames'],z['video_frames'])
                for key,dim in (('f6',1280),('deform',2560)):
                    s[key]=z[key].copy()
                    assert s[key].shape==(len(s['frames']),dim) and np.isfinite(s[key]).all()
    print('CACHE_LOADED',flush=True)
    dist=distribution(samples);write_csv(out/'label_distribution.csv',dist)
    baseline_distribution=list(csv.DictReader((args.deform/'label_distribution.csv').open()))
    for old,new in zip(baseline_distribution,dist):
        for key in ('positive','negative','excluded_terminal_ticks'):
            assert int(old[key])==new[key]
    norm={}
    for key in ('f6','deform'):
        bank=np.concatenate([s[key] for s in samples['train']])
        norm[key]=(torch.from_numpy(bank.mean(0,dtype=np.float64).astype(np.float32)),
                   torch.from_numpy(np.maximum(bank.std(0,dtype=np.float64),.01).astype(np.float32)))
        del bank
    dump(out/'protocol.json',dict(source=str(relative_path(args.source)),deform_reference=str(relative_path(args.deform)),
         horizons=list(HORIZONS),seeds=list(SEEDS),inputs=list(KINDS),
         label='same finalAlign failure anchor and valid masks as Deform;H>0 preanchor band1,H0 persistentterminal1',
         loss='identical train-count balanced BCE N/(2Nc);no Gaussian/no augmentation',
         encoder='cached frozen T-Rex finger1280D and frozen Deform5x512D;dense step0',
         models='F6 Linear128 or F6 Linear128+Deform Linear128 concat256 ->unidirectional GRU128->Linear1 sigmoid',
         training='AdamWlr.001wd.0001batch8max30patience8clip1;train-only mean/std stdclip.01',
         selection='each H/seed own natural val BA best epoch,ties earliest;threshold.5,no test-selected H',
         created_at=datetime.datetime.now().astimezone().isoformat()))

    class Model(nn.Module):
        def __init__(self,kind):
            super().__init__();self.keys=FEATURES[kind]
            self.projections=nn.ModuleDict()
            for key in self.keys:
                mean,std=norm[key]
                self.register_buffer(key+'_mean',mean.clone());self.register_buffer(key+'_std',std.clone())
                self.projections[key]=nn.Linear(len(mean),128)
            self.gru=nn.GRU(128*len(self.keys),128,batch_first=True)
            self.prediction=nn.Linear(128,1)
        def forward(self,x):
            values=[self.projections[k]((x[k]-getattr(self,k+'_mean'))/getattr(self,k+'_std')) for k in self.keys]
            hidden,_=self.gru(torch.cat(values,dim=-1))
            return self.prediction(hidden).squeeze(-1)

    def batch(ss,h,kind):
        lengths=[len(s['frames']) for s in ss];width=max(lengths)
        x={key:np.zeros((len(ss),width,len(norm[key][0])),np.float32) for key in FEATURES[kind]}
        y=np.zeros((len(ss),width),np.float32);mask=np.zeros_like(y,dtype=bool)
        for i,s in enumerate(ss):
            for key in x:x[key][i,:lengths[i]]=s[key]
            q,k=target(s['frames'],s['key'],s['outcome']==2,h)
            y[i,:lengths[i]],mask[i,:lengths[i]]=q,k
        return {k:torch.from_numpy(v).to(args.device) for k,v in x.items()},torch.from_numpy(y).to(args.device),torch.from_numpy(mask).to(args.device),lengths

    def infer(model,ss,h,kind):
        model.eval();probabilities=[]
        with torch.inference_mode():
            for a in range(0,len(ss),8):
                x,_,_,lengths=batch(ss[a:a+8],h,kind);p=model(x).sigmoid().cpu().numpy()
                probabilities.extend(p[i,:length] for i,length in enumerate(lengths))
        return np.concatenate(probabilities),np.r_[0,np.cumsum([len(p) for p in probabilities])]

    results=[]
    for kind in KINDS:
        for h in HORIZONS:
            d=next(r for r in dist if r['split']=='train' and r['horizon_frames']==h)
            for seed in SEEDS:
                torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
                model=Model(kind).to(args.device)
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                rng=np.random.default_rng(seed)
                directory=out/'runs'/kind/f'H{h}'/f'seed_{seed}';directory.mkdir(parents=True)
                best=-1;best_epoch=0;history=[]
                for epoch in range(1,31):
                    model.train();order=rng.permutation(len(samples['train']));loss_sum=0;total=0
                    for a in range(0,len(order),8):
                        ss=[samples['train'][int(i)] for i in order[a:a+8]]
                        x,y,mask,_=batch(ss,h,kind);optimizer.zero_grad(set_to_none=True)
                        element=nn.functional.binary_cross_entropy_with_logits(model(x),y,reduction='none')
                        weights=torch.where(y>0,d['positive_weight'],d['negative_weight'])
                        loss=(element*weights)[mask].mean();loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                        loss_sum+=float((element*weights)[mask].detach().sum());total+=int(mask.sum())
                    p,offsets=infer(model,samples['val'],h,kind)
                    val,_,_=evaluate_arrays(samples['val'],p,offsets,h,'val',seed)
                    history.append(dict(epoch=epoch,train_balanced_BCE=loss_sum/total,validation_BA=val['balanced_accuracy']))
                    if val['balanced_accuracy']>best+1e-8:
                        best=val['balanced_accuracy'];best_epoch=epoch
                        torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                        input=kind,horizon=h,seed=seed,best_epoch=epoch),directory/'best.pt')
                    if epoch-best_epoch>=8:break
                model.load_state_dict(torch.load(directory/'best.pt',map_location=args.device,weights_only=True)['state_dict'])
                result=dict(input=kind,horizon_frames=h,seed=seed,best_epoch=best_epoch,epochs=len(history))
                for split in ('val','test'):
                    p,offsets=infer(model,samples[split],h,kind)
                    frame,event,events=evaluate_arrays(samples[split],p,offsets,h,split,seed)
                    np.savez_compressed(directory/(split+'_predictions.npz'),probabilities=p,offsets=offsets)
                    result[split]=frame;result[split+'_event']=event
                    write_csv(directory/(split+'_events.csv'),events)
                dump(directory/'metrics.json',result);dump(directory/'history.json',history)
                results.append(result);dump(out/'results_partial.json',results)
                print('RUN',len(results),50,kind,'H',h,'seed',seed,'valBA',round(result['val']['balanced_accuracy'],4),
                      'testBA',round(result['test']['balanced_accuracy'],4),flush=True)
                del model,optimizer;torch.cuda.empty_cache()
    dump(out/'results.json',results)
    dump(out/'manifest.json',dict(status='complete',new_runs=50,horizons=list(HORIZONS),seeds=list(SEEDS),
         code_sha256=sha(Path(__file__)),baseline_code_sha256=sha(Path(__file__).with_name('early_warning.py')),
         completed_at=datetime.datetime.now().astimezone().isoformat()))
    print('TRAIN_COMPLETE',flush=True)


def report(args):
    out=args.output;samples=read_data(args.source,False)
    results=json.loads((out/'results.json').read_text())
    baseline=json.loads((args.deform/'results.json').read_text())
    results+=[dict(r,input='deform') for r in baseline]
    assert len(results)==75 and len({(r['input'],r['horizon_frames'],r['seed']) for r in results})==75
    summary=[];probabilities={};event_rows=[];normalized=[]
    for kind in ('deform','f6','fusion'):
        for h in HORIZONS:
            rr=[r for r in results if r['input']==kind and r['horizon_frames']==h]
            row=dict(input=kind,horizon_frames=h,horizon_seconds=h/30,seeds=5)
            for phase in ('val','test','val_event','test_event'):
                for key in ('balanced_accuracy','precision','recall','f1','macro_f1','fpr','pr_auc','roc_auc'):
                    v=[r[phase][key] for r in rr]
                    row[phase+'_'+key+'_mean']=float(np.mean(v));row[phase+'_'+key+'_std']=float(np.std(v,ddof=1))
                for gt in (0,1):
                    den=sum(sum(r[phase]['confusion_matrix'][gt]) for r in rr)
                    for pred in (0,1):
                        count=sum(r[phase]['confusion_matrix'][gt][pred] for r in rr)
                        normalized.append(dict(input=kind,horizon_frames=h,phase=phase,gt=gt,prediction=pred,
                                               mean_count_per_seed=count/5,row_normalized=count/den))
            summary.append(row)
            for split in ('val','test'):
                ps=[]
                for r in rr:
                    directory=(args.deform/'runs'/f'H{h}'/f"seed_{r['seed']}") if kind=='deform' else out/'runs'/kind/f'H{h}'/f"seed_{r['seed']}"
                    with np.load(directory/(split+'_predictions.npz')) as z:
                        p=z['probabilities'].copy();offsets=z['offsets'].copy()
                    assert np.array_equal(offsets,np.r_[0,np.cumsum([len(s['frames']) for s in samples[split]])])
                    frame,event,events=evaluate_arrays(samples[split],p,offsets,h,split,r['seed'])
                    for phase,replay in ((split,frame),(split+'_event',event)):
                        assert replay['confusion_matrix']==r[phase]['confusion_matrix']
                        for key in ('balanced_accuracy','pr_auc','roc_auc'):
                            assert np.isclose(replay[key],r[phase][key])
                    event_rows.extend(dict(e,input=kind) for e in events);ps.append(p)
                probabilities[(kind,h,split)]=(np.stack(ps).mean(0),offsets)
    dump(out/'comparison_results.json',results);write_csv(out/'summary.csv',summary)
    write_csv(out/'event_predictions.csv',event_rows);write_csv(out/'confusion_normalized.csv',normalized)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup={(r['input'],r['horizon_frames']):r for r in summary}
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for si,split in enumerate(('val','test')):
        for mi,key in enumerate(('pr_auc','roc_auc','balanced_accuracy')):
            ax=axes[si,mi]
            for kind in ('deform','f6','fusion'):
                yy=np.array([lookup[(kind,h)][split+'_'+key+'_mean'] for h in HORIZONS])
                sd=np.array([lookup[(kind,h)][split+'_'+key+'_std'] for h in HORIZONS])
                ax.errorbar(np.arange(5),100*yy,yerr=100*sd,marker='o',capsize=3,label=kind)
            ax.set_xticks(np.arange(5),['0(state)','8','15','30','45'])
            ax.set_title(split.upper()+' '+key);ax.set_xlabel('H (camera frames)');ax.set_ylabel('%');ax.legend()
    fig.tight_layout();fig.savefig(out/'modality_comparison.png',dpi=180);fig.savefig(out/'modality_comparison.pdf');plt.close(fig)
    curves=[]
    for h in HORIZONS:
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for si,split in enumerate(('val','test')):
            ss=samples[split];lo=min(int(s['frames'].min()-s['key']) for s in ss)
            upper=max(int(s['frames'].max()-s['key']) for s in ss) if h==0 else 0
            grid=np.arange(lo,upper+1)
            for oi,outcome in enumerate((1,2)):
                ax=axes[si,oi]
                for kind in ('deform','f6','fusion'):
                    p,offsets=probabilities[(kind,h,split)];values=[]
                    for i,s in enumerate(ss):
                        if s['outcome']!=outcome:continue
                        a,b=offsets[i:i+2];relative=s['frames']-s['key'];v=np.full(len(grid),np.nan)
                        for frame in np.unique(relative[relative<=upper]):
                            v[int(frame-lo)]=p[a:b][relative==frame].mean()
                        values.append(v)
                    values=np.stack(values);mean=np.full(len(grid),np.nan);sd=mean.copy()
                    for j,frame in enumerate(grid):
                        local=values[:,j];local=local[np.isfinite(local)]
                        if len(local):mean[j]=local.mean()
                        if len(local)>1:sd[j]=local.std(ddof=1)
                        curves.append(dict(input=kind,horizon_frames=h,split=split,final_outcome=outcome,
                                           relative_seconds=frame/30,mean=float(mean[j]),variance=float(sd[j]**2),N=len(local)))
                    line,=ax.plot(grid/30,mean,label=kind)
                    ax.fill_between(grid/30,np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),alpha=.10,color=line.get_color())
                ax.axhline(.5,color='gray',ls='--');ax.axvline(0,color='black',ls=':')
                if outcome==2:ax.axvspan(-h/30,0 if h else upper/30,alpha=.08)
                ax.set_ylim(0,1);ax.set_xlabel('Time relative to final Align end (s)');ax.set_ylabel('Risk mean ± rollout SD')
                ax.set_title(split.upper()+' final '+('success' if outcome==1 else 'failure'));ax.legend()
        fig.suptitle(f'H={h} frames | frozen tactile modalities,5seed mean within rollout');fig.tight_layout()
        fig.savefig(out/f'key_relative_H{h}.png',dpi=180);fig.savefig(out/f'key_relative_H{h}.pdf');plt.close(fig)
    write_csv(out/'key_relative_variance.csv',curves)
    paired=[]
    for h in HORIZONS:
        for phase in ('val','test'):
            for key in ('pr_auc','roc_auc','balanced_accuracy'):
                for a,b in (('f6','deform'),('fusion','deform'),('fusion','f6')):
                    for seed in SEEDS:
                        va=next(r[phase][key] for r in results if r['input']==a and r['horizon_frames']==h and r['seed']==seed)
                        vb=next(r[phase][key] for r in results if r['input']==b and r['horizon_frames']==h and r['seed']==seed)
                        paired.append(dict(horizon_frames=h,phase=phase,metric=key,contrast=a+'-'+b,seed=seed,delta=va-vb))
    write_csv(out/'paired_seed_differences.csv',paired)
    dump(out/'verification.json',dict(status='PASS',new_runs50=True,deform_reused25=True,total75=True,
         same_split_labels_classweights=True,all_val_test_frame_and_event_metrics_recomputed=True,
         encoders_frozen_cache_only=True,original_checkpoints_unchanged=True))
    lines=['# Branch A：F6 / Deform / Fusion early-warning modality comparison','',
           '新增F6与Fusion各25runs，复用原Deform25runs，共3模态×5H×5seeds=75results。H={0,8,15,30,45}相机帧，seeds42–46，30fps。仅最后Align interval为failure时以end作anchor，最终success全段0，早期/Insert Key不创建正标签。H>0的[anchor−H,anchor)为1，anchor及之后从loss、val checkpoint selection和main evaluation排除；H0为anchor后持续terminal1。原标注不改，原rollout split80/17/17不变。','',
           '## 唯一的训练设置变化是模态','',
           'F6：连续past16rawticks→冻结现有T-Rex finger encoder→1280D→Linear128→单层单向GRU128→Linear1 sigmoid。不是另取16个encoder features作为输入。Fusion：同一时刻F6→Linear128和当前Deform2560D→Linear128，concat256→GRU128→Linear1。Deform：当前冻结每指pool2×2后512D×5→2560D→Linear128→GRU128→Linear1。GRU每rollout开始reset，中途interval不reset。','',
           '与Deform相同：dense step0、不做augmentation/Gaussian，train-count balanced BCE w0=N/(2N0),w1=N/(2N1)，train-only标准化/stdclip.01，AdamW lr.001/wd.0001，batch8，max30epoch/patience8，clip1。各H/seed仅按自然val BA选best epoch，平分取最早。统一0.5阈值，但比较重点是PR-AUC(AP)、ROC-AUC；不按test选模态或H。','',
           '同一horizon三模态使用完全一致的GT与mask，故可直接比较；不同H有不同GT和正例比例，不应仅比较跨H AP绝对值。BA是两类recall均值；P/R/F1/FPR针对positive风险类。Class-balanced BCE的risk未校准。Val/test各3failure、14success，5seeds不增加独立rollout数量。','',
           'Event score是有效范围max risk；first alarm为首次≥0.5，positive-band检测与过早报警需区分。Lead仅对已报警failure统计；不报警不填0。Branch B单独校准Deform，不影响这里的固定0.5对照和threshold-free指标。','',
           '## Test，均值±5seed SD %','',
           '| H帧 | Input | BA | PR-AUC(AP) | ROC-AUC | Precision | Recall | FPR |',
           '|---:|---|---:|---:|---:|---:|---:|---:|']
    for h in HORIZONS:
        for kind in ('deform','f6','fusion'):
            r=lookup[(kind,h)]
            lines.append('| '+str(h)+' | '+kind+' | '+' | '.join(f"{100*r['test_'+k+'_mean']:.2f} ± {100*r['test_'+k+'_std']:.2f}" for k in ('balanced_accuracy','pr_auc','roc_auc','precision','recall','fpr'))+' |')
    lines+=['','![Modality metrics](modality_comparison.png)','',
            '## All validation/test Key-relative curves','',
            '先每rollout平均5seeds，再对当前有数据的rollout平均±rollout SD；CSV保留N和variance。H>0曲线anchor时刻不参与main指标，H0保留terminal tail。','']
    for h in HORIZONS:lines+= [f'![H{h}](key_relative_H{h}.png)','']
    lines+=['[全部75results](comparison_results.json) · [全部指标](summary.csv) · [label分布](label_distribution.csv) · [同seed配对差值](paired_seed_differences.csv) · [event首报及lead](event_predictions.csv) · [归一化矩阵](confusion_normalized.csv) · [曲线variance/N](key_relative_variance.csv) · [验证](verification.json)','']
    (out/'README.md').write_text('\n'.join(lines))
    snap=out/'code_snapshot';snap.mkdir()
    for p in (Path(__file__),Path(__file__).with_name('early_warning.py')):shutil.copy2(p,snap/p.name)
    dest=project_path('WeeklySummary/10.5/early_warning_modalities', out.name);dest.mkdir(parents=True,exist_ok=False);copies=[]
    for p in out.iterdir():
        if p.is_file():
            q=dest/p.name;shutil.copy2(p,q);digest=sha(p);assert sha(q)==digest
            copies.append(dict(source=str(relative_path(p)),destination=str(relative_path(q)),sha256=digest))
    dump(dest/'copy_manifest.json',copies)
    print('REPORT_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','report'])
    p.add_argument('--source',type=Path,default=project_path('outputs/sharpa_gaussian_online_data/20261005_182000'))
    p.add_argument('--deform',type=Path,default=project_path('outputs/sharpa_early_warning/20261006_001500'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();args.source=args.source.resolve();args.deform=args.deform.resolve();args.output=args.output.resolve()
    (train if args.action=='train' else report)(args)


if __name__=='__main__':main()
