"""Complete every F6 temporal sampling setting with seeds42..46, reuse seed42."""
from .common import project_path, relative_path
import argparse
import csv
import datetime
import json
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import ROOT,dump,sha
from .merged_online import MergedProbe,load,save
from .gaussian_online import read_samples,batch,evaluate
from .train_intervals import normalization
from .align_windows import metrics


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    data=project_path('outputs/sharpa_gaussian_online_data/20261005_182000');dm=json.loads((data/'dataset_manifest.json').read_text());samples=read_samples(data,dm,'original_align');norm=normalization(samples['train'])
    sources=[project_path('outputs/sharpa_f6_temporal_sampling', tag) for tag in ('20261005_195800','20261005_202500')]
    banks={mode:{step:[] for step in range(1,9)} for mode in ('feature','raw')}
    for step in range(1,9):
        source=sources[0 if step<=3 else 1]
        for sample in samples['train']:
            z=load(source/'train_features'/f"{sample['interaction_id']}_step{step}.npz")
            for mode in ('feature','raw'):
                s=dict(sample);s['f6']=z[mode];s['positions']=z['positions'];s['labels']=z['labels'];banks[mode][step].append(s)
    for mode in banks:banks[mode][0]=samples['train']
    configs=[('baseline',0)]+[(mode,step) for mode in ('feature','raw') for step in range(1,9)]+[(mode,'multi') for mode in ('feature','raw')]
    dump(out/'protocol.json',dict(seeds=list(range(42,47)),sigma=8,configs=[dict(mode=m,step=s) for m,s in configs],unique_configs=19,total_runs=95,new_runs=76,reused_seed42=19,validation_test_step=0,multi_steps=[0,1,2,3],settings='same original_align split, F6 encoder frozen; shared dense train normalization; sigma8 originaltime; same AdamW .001/.0001 batch8 max30 patience8 clip1 plain softCE',source_banks=[str(relative_path(p)) for p in sources]))
    results=[];all_predictions={};torch.backends.cudnn.allow_tf32=True
    for seed in range(42,47):
        for mode,step in configs:
            identity=mode+'_step'+str(step);directory=out/'runs'/identity/f'seed_{seed}';directory.mkdir(parents=True);config=dict(input_kind='f6',head_kind='gru')
            if seed==42:
                src=sources[1 if isinstance(step,int) and step>=4 else 0]/'runs'/identity
                old=json.loads((src/'metrics.json').read_text());model=MergedProbe(**config);blob=torch.load(src/'best.pt',map_location=args.device,weights_only=True);model.load_state_dict(blob['state_dict']);model.to(args.device)
                shutil.copy2(src/'best.pt',directory/'best.pt');shutil.copy2(src/'history.json',directory/'history.json');val=old['validation'];test=old['test'];best_epoch=old['best_epoch'];epochs=old['epochs'];p=load(src/'test_predictions.npz')['probabilities'];vp=load(src/'val_predictions.npz')['probabilities']
            else:
                torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);model=MergedProbe(**config)
                for k,v in norm.items():getattr(model,k).copy_(v)
                model.to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);aug_rng=np.random.default_rng(seed+10000);cycles=np.stack([aug_rng.permutation(4) for _ in samples['train']]);best=-1;best_epoch=0;history=[]
                for epoch in range(1,31):
                    model.train();order=rng.permutation(len(samples['train']));total=0;loss_sum=0
                    for start in range(0,len(order),8):
                        ids=order[start:start+8]
                        if mode=='baseline':chosen=[samples['train'][int(i)] for i in ids]
                        else:chosen=[banks[mode][int(cycles[int(i),(epoch-1)%4]) if step=='multi' else step][int(i)] for i in ids]
                        x,q,mask,_=batch(chosen,'f6',8,args.device);optimizer.zero_grad(set_to_none=True);logits,_=model(**x);element=-(q*logits.log_softmax(-1)).sum(-1);loss=element[mask].mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();loss_sum+=float(element[mask].detach().sum());total+=int(mask.sum())
                    val,_=evaluate(model,samples['val'],8,args.device);score=val['balanced_accuracy'];history.append(dict(epoch=epoch,train_soft_ce=loss_sum/total,train_ticks=total,val_balanced_accuracy=score,val_macro_f1=val['macro_f1']))
                    if score>best+1e-8:
                        best=score;best_epoch=epoch;torch.save(dict(model_config=config,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},sigma=8,seed=seed,sampling_mode=mode,step=step,best_epoch=epoch),directory/'best.pt')
                    if epoch-best_epoch>=8:break
                blob=torch.load(directory/'best.pt',map_location=args.device,weights_only=True);model.load_state_dict(blob['state_dict']);val,vp=evaluate(model,samples['val'],8,args.device);test,p=evaluate(model,samples['test'],8,args.device);epochs=len(history);dump(directory/'history.json',history)
                del optimizer
            for split,prob in (('val',vp),('test',p)):
                ss=samples[split];save(directory/(split+'_predictions.npz'),dict(probabilities=prob,labels=np.concatenate([s['labels'] for s in ss]),offsets=np.r_[0,np.cumsum([len(s['labels']) for s in ss])]))
                all_predictions[(identity,seed,split)]=prob
            r=dict(setting=identity,mode=mode,step=step,seed=seed,sigma=8,best_epoch=best_epoch,epochs=epochs,validation=val,test=test,reused=seed==42);results.append(r);dump(directory/'metrics.json',r);dump(out/'results_partial.json',results);print('RUN',len(results),95,identity,seed,'testBA',round(test['balanced_accuracy'],4),flush=True);del model;torch.cuda.empty_cache()
    dump(out/'results.json',results);per_seed=[]
    for r in results:
        m=r['test'];per_seed.append(dict(setting=r['setting'],mode=r['mode'],step=r['step'],seed=r['seed'],val_BA=r['validation']['balanced_accuracy'],BA=m['balanced_accuracy'],macro_F1=m['macro_f1'],failure_precision=m['per_class']['failure']['precision'],failure_recall=m['per_class']['failure']['recall'],failure_FPR=m['failure_false_positive_rate']))
        stored=load(out/'runs'/r['setting']/f"seed_{r['seed']}"/'test_predictions.npz');recomputed=metrics(stored['labels'],stored['probabilities']);assert recomputed['confusion_matrix']==m['confusion_matrix']
    write_csv(out/'per_seed.csv',per_seed);summary=[];paired=[]
    for mode,step in configs:
        identity=mode+'_step'+str(step);rr=[r for r in per_seed if r['setting']==identity];entry=dict(setting=identity,mode=mode,step=step,seeds=5)
        for key in ('val_BA','BA','macro_F1','failure_precision','failure_recall','failure_FPR'):
            values=[r[key] for r in rr];entry[key+'_mean']=float(np.mean(values));entry[key+'_std']=float(np.std(values,ddof=1))
        deltas=[r['BA']-next(x['BA'] for x in per_seed if x['setting']=='baseline_step0' and x['seed']==r['seed']) for r in rr];entry['paired_delta_BA_mean']=float(np.mean(deltas));entry['paired_delta_BA_std']=float(np.std(deltas,ddof=1));summary.append(entry)
        for r,d in zip(rr,deltas):paired.append(dict(setting=identity,seed=r['seed'],delta_BA_vs_same_seed_baseline=d))
    write_csv(out/'summary.csv',summary);write_csv(out/'paired_seed_differences.csv',paired)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for ax,key in zip(axes.ravel(),('BA','macro_F1','failure_precision','failure_recall','failure_FPR','val_BA')):
        baseline=summary[0]
        for mode,color in (('feature','tab:blue'),('raw','tab:orange')):
            rr=[baseline]+[r for r in summary if r['mode']==mode and isinstance(r['step'],int)];x=[0]+[r['step'] for r in rr[1:]];y=np.array([100*r[key+'_mean'] for r in rr]);sd=np.array([100*r[key+'_std'] for r in rr]);ax.errorbar(x,y,yerr=sd,color=color,marker='o',capsize=3,label=mode);multi=next(r for r in summary if r['mode']==mode and r['step']=='multi');ax.errorbar([9 if mode=='feature' else 9.25],[100*multi[key+'_mean']],yerr=[100*multi[key+'_std']],color=color,marker='s',capsize=4)
        ax.set_title(key+' mean ± seed SD');ax.set_xlabel('Train step; M=multi0..3');ax.set_xticks(range(10),[str(i) for i in range(9)]+['M']);ax.set_ylabel('%');ax.grid(alpha=.15)
    axes[0,0].legend();fig.suptitle('F6 GRU sigma8 | seeds42–46 | dense step0 validation/test');fig.tight_layout();fig.savefig(out/'step_comparison_five_seeds.png',dpi=170);fig.savefig(out/'step_comparison_five_seeds.pdf');plt.close(fig)
    curve_rows=[]
    for split in ('val','test'):
        ss=samples[split];offsets=np.r_[0,np.cumsum([len(s['labels']) for s in ss])];lo=min(int(s['video_frames'].min()-s['key']) for s in ss);hi=max(int(s['video_frames'].max()-s['key']) for s in ss);grid=np.arange(lo,hi+1)
        for mode in ('feature','raw'):
            identities=['baseline_step0']+[mode+'_step'+str(i) for i in range(1,9)]+[mode+'_stepmulti'];fig,axes=plt.subplots(2,10,figsize=(30,7),sharex=True,sharey=True)
            for col,identity in enumerate(identities):
                prob=np.stack([all_predictions[(identity,seed,split)] for seed in range(42,47)]).mean(0)
                for row,outcome in enumerate((1,2)):
                    vv=[]
                    for i,s in enumerate(ss):
                        if s['outcome']!=outcome:continue
                        a,b=offsets[i:i+2];relative=s['video_frames']-s['key'];table=np.full((len(grid),3),np.nan)
                        for frame in np.unique(relative):table[int(frame-lo)]=prob[a:b][relative==frame].mean(0)
                        vv.append(table)
                    values=np.stack(vv);n=np.isfinite(values[:,:,0]).sum(0);mean=np.full((len(grid),3),np.nan);sd=mean.copy()
                    for j,count in enumerate(n):
                        local=values[:,j];local=local[np.isfinite(local[:,0])]
                        if count:mean[j]=local.mean(0)
                        if count>1:sd[j]=local.std(0,ddof=1)
                        for k,label in enumerate(('progress','success','failure')):curve_rows.append(dict(split=split,setting=identity,gt_outcome='success' if outcome==1 else 'failure',relative_seconds=float(grid[j]/30),probability=label,mean=float(mean[j,k]),variance=float(sd[j,k]**2),std=float(sd[j,k]),interaction_count=int(count)))
                    ax=axes[row,col]
                    for k,(label,color) in enumerate(zip(('progress','success','failure'),('gray','tab:blue','tab:orange'))):ax.plot(grid/30,mean[:,k],color=color,label=label);ax.fill_between(grid/30,np.clip(mean[:,k]-sd[:,k],0,1),np.clip(mean[:,k]+sd[:,k],0,1),color=color,alpha=.08)
                    ax.axvline(0,color='red',linestyle=':',linewidth=1);ax.set_xlim(lo/30,hi/30);ax.set_ylim(0,1);ax.set_title(identity+'\n'+('Success GT' if outcome==1 else 'Failure GT'),fontsize=9);ax.set_xlabel('Seconds relative to Key',fontsize=8)
            axes[0,0].legend(fontsize=7);fig.suptitle(split.upper()+' '+mode+' | seed-averaged interaction probabilities | mean ± interaction SD');fig.tight_layout();fig.savefig(out/f'{mode}_{split}_key_relative_curves.png',dpi=160);fig.savefig(out/f'{mode}_{split}_key_relative_curves.pdf');plt.close(fig)
    write_csv(out/'key_relative_mean_variance.csv',curve_rows);dump(out/'verification.json',dict(status='PASS',runs=95,new_runs=76,seeds=list(range(42,47)),same_dense_val_test=True,all_metrics_recomputed=True,step0_shared=True,minimal_checks=True))
    lines=['# F6 temporal sampling：全部step0–8与multi，5 seeds','', '固定original_align、Gaussianσ8、F6+GRU、frozen encoder、原rollout split、optimizer、同一dense-train normalization和训练设置。seeds42–46；step=stride−1，Feature/Raw step0共享。两类各8个单step及multi0–3，共19唯一配置×5=95runs；复用19个seed42结果，新增76次训练。val/test均dense step0，epoch按val三类硬GT BA选择，没有test选参数。','',
    'Feature-step先连续16raw ticks冻结编码，再稀疏feature；Raw-step先取16个稀疏raw ticks编码，再GRU。相同step的训练endpoint及无效点过滤一致；Gaussian保留原position/key_index，σ不随stride重编号。Raw历史跨度15×(step+1)/30秒，step8为4.5秒；短interaction左补首tick，因此高step同时增加padding。multi仍只轮换0/1/2/3，每epoch每interaction一个step；不扩展到0–8。不同stride改变监督tick数量，训练配置相同不意味着监督tick数相同。','',
    '下表是同一test2210有效tick/24Align/17rollout的三类frame-wise指标，非双阈值event结果。均值±样本标准差(ddof1)来自5seeds。paired_delta_BA与每个相同seed的dense baseline配对；seed SD不是rollout置信区间。','', '| Train setting | Val BA % | Test BA % | Macro F1 % | Fail precision % | Fail recall % | Fail FPR % | Paired ΔBA pp |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append('| '+r['setting']+' | '+' | '.join(f"{100*r[k+'_mean']:.2f} ± {100*r[k+'_std']:.2f}" for k in ('val_BA','BA','macro_F1','failure_precision','failure_recall','failure_FPR'))+f" | {100*r['paired_delta_BA_mean']:.2f} ± {100*r['paired_delta_BA_std']:.2f} |")
    lines+=['','![Five-seed comparison](step_comparison_five_seeds.png)','', '## Key-relative curves','', '每interaction先平均5seeds，再按GT outcome对当时有数据的interaction等权平均。阴影为interaction SD，不是seed SD；逐时间N/variance在CSV。重复相机帧先合并，缺失不补值；各模式图上排Success GT，下排Failure GT，三条概率progress/success/failure，Key为0秒。','']
    for mode in ('feature','raw'):
        for split in ('val','test'):lines.extend([f'![{mode} {split}]({mode}_{split}_key_relative_curves.png)',''])
    lines+=['[95次完整指标](results.json) · [per-seed CSV](per_seed.csv) · [5seed summary](summary.csv) · [配对差异](paired_seed_differences.csv) · [曲线/N/variance](key_relative_mean_variance.csv) · [验证](verification.json)','']
    (out/'README.md').write_text('\n'.join(lines));dump(out/'manifest.json',dict(status='complete',created_at=datetime.datetime.now().astimezone().isoformat(),source_data=str(relative_path(data)),runs=95,new_runs=76,code_sha256=sha(Path(__file__))))
    snapshot=out/'code_snapshot';snapshot.mkdir();shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    target=project_path('WeeklySummary/10.5/f6_temporal_sampling', out.name);target.mkdir(parents=True,exist_ok=False);copies=[]
    for p in out.iterdir():
        if p.is_file():q=target/p.name;shutil.copy2(p,q);copies.append(dict(source=str(relative_path(p)),destination=str(relative_path(q)),sha256=sha(p)))
    dump(target/'copy_manifest.json',copies)
    with (project_path('WeeklySummary/10.5/10.5.md')).open('a') as f:f.write('\n\n## 15. F6 temporal sampling：step0–8与multi全部5seeds\n\n'+ '\n'.join(lines[2:lines.index('## Key-relative curves')]).replace('](step_comparison_five_seeds.png)',f'](f6_temporal_sampling/{out.name}/step_comparison_five_seeds.png)')+f'\n\n[完整5seed报告与曲线](f6_temporal_sampling/{out.name}/README.md)。\n')
    print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
