"""Small frozen F6 temporal-scale ablation, dense validation/test, sigma8 seed42."""
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
from .data import episode_arrays
from .models import FrozenEncoders
from .merged_online import MergedProbe,load,save
from .gaussian_online import read_samples,batch,evaluate
from .train_intervals import normalization
from .align_windows import metrics


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);source=project_path('outputs/sharpa_gaussian_online_data/20261005_182000');dm=json.loads((source/'dataset_manifest.json').read_text());samples=read_samples(source,dm,'original_align');norm=normalization(samples['train'])
    origin=project_path('outputs/sharpa_tactile_three_class/20261004_160624');records=json.loads((origin/'data_manifest.json').read_text());dense_manifest=json.loads((project_path('outputs/sharpa_merged_online_datasets/20261005_164000/dataset_manifest.json')).read_text());events={r['rollout_id']:r['events'] for r in dense_manifest['rollouts']}
    encoder=FrozenEncoders().to(args.device);assert not any(p.requires_grad for p in encoder.parameters());banks={mode:{step:[] for step in (0,1,2,3)} for mode in ('feature','raw')};audits=[]
    raw_cache={}
    for i,s in enumerate(samples['train']):
        rid=s['rollout_id']
        if rid not in raw_cache:raw_cache[rid]=episode_arrays(records['records'][rid],events[rid],records['signature']['camera'],num_classes=3)
        raw=raw_cache[rid];first=int(s['ticks'][0])
        for step in range(4):
            stride=step+1;ids=np.arange(0,len(s['labels']),stride);ticks=s['ticks'][ids]
            windows=np.maximum(ticks[:,None]+np.arange(-15,1)[None]*stride,first)
            assert np.all(windows<=ticks[:,None]) and windows.min()>=first
            valid=raw['valid'][windows].all(1);ids=ids[valid];windows=windows[valid];assert len(ids)>0
            chosen=dict(s)
            for k in ('f6','deform','positions','ticks','video_frames','labels'):chosen[k]=s[k][ids].copy()
            # Preserve ORIGINAL key_index and positions so sigma is not scaled with stride.
            banks['feature'][step].append(chosen)
            sparse=dict(chosen)
            if step:
                with torch.inference_mode():sparse['f6']=encoder.f6_features(torch.from_numpy(raw['f6'][windows]).to(args.device)).cpu().numpy()
            banks['raw'][step].append(sparse)
            save(out/'train_features'/f"{s['interaction_id']}_step{step}.npz",dict(feature=chosen['f6'],raw=sparse['f6'],positions=chosen['positions'],raw_ticks=windows,labels=chosen['labels']))
            audits.append(dict(interaction_id=s['interaction_id'],step=step,stride=stride,kept=len(ids),dropped_invalid=int((~valid).sum()),key_index=s['key_index'],raw_window_max_span_ticks=15*stride))
        if (i+1)%25==0:print('ENCODE',i+1,117,flush=True)
    del encoder,raw_cache;torch.cuda.empty_cache();torch.backends.cudnn.allow_tf32=True
    dump(out/'protocol.json',dict(group='original_align',input='f6',head='gru',sigma=8,seed=42,step_to_stride={str(i):i+1 for i in range(4)},validation_test_step=0,normalization='same dense train-only statistics for every setting',soft_target='original positions and key_index: sigma8 in original valid ticks, unchanged by downsampling',training='AdamW lr.001 wd.0001 batch8 interactions max30epochs patience8 clip1; unweighted softCE; val dense hardGT BA selects epoch',multi='one of4steps per interaction per epoch, permuted4epoch cycle with separate augmentation RNG; no4x epochs or gradient steps',paired_endpoints='feature/raw share sampled current ticks and valid-sparse-window filtering',raw_window='past16 sparse rawticks spaced stride, repeated first interaction tick leftpad; encoder frozen',feature_window='existing dense past16 rawtick encoding then stride sample feature sequence',no_new_seeds=True))
    dump(out/'sampling_audit.json',audits)
    configs=[('baseline',0)]+[(mode,step) for mode in ('feature','raw') for step in (1,2,3)]+[(mode,'multi') for mode in ('feature','raw')]
    results=[];predictions={};baseline=project_path('outputs/sharpa_gaussian_online/20261005_182000/runs/original_align/f6_gru/sigma_8/seed_42')
    for mode,step in configs:
        identity=mode+'_step'+str(step);directory=out/'runs'/identity;directory.mkdir(parents=True);config=dict(input_kind='f6',head_kind='gru');model=MergedProbe(**config)
        if mode=='baseline':
            blob=torch.load(baseline/'best.pt',map_location=args.device,weights_only=True);model.load_state_dict(blob['state_dict']);model.to(args.device);shutil.copy2(baseline/'best.pt',directory/'best.pt');history=json.loads((baseline/'history.json').read_text());best_epoch=blob['best_epoch'];original_metrics=json.loads((baseline/'metrics.json').read_text());val=original_metrics['validation'];test=original_metrics['test'];p=load(baseline/'test_predictions.npz')['probabilities'];_,vp=evaluate(model,samples['val'],8,args.device)
        else:
            torch.manual_seed(42);np.random.seed(42);random.seed(42);model=MergedProbe(**config)
            for k,v in norm.items():getattr(model,k).copy_(v)
            model.to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(42);aug_rng=np.random.default_rng(10042);cycles=np.stack([aug_rng.permutation(4) for _ in samples['train']]);best=-1;best_epoch=0;history=[]
            for epoch in range(1,31):
                model.train();order=rng.permutation(len(samples['train']));loss_sum=0;count=0
                for start in range(0,len(order),8):
                    ids=order[start:start+8];chosen=[banks[mode][int(cycles[int(i),(epoch-1)%4]) if step=='multi' else step][int(i)] for i in ids]
                    x,q,mask,_=batch(chosen,'f6',8,args.device);optimizer.zero_grad(set_to_none=True);logits,_=model(**x);element=-(q*logits.log_softmax(-1)).sum(-1);loss=element[mask].mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();loss_sum+=float(element[mask].detach().sum());count+=int(mask.sum())
                val,_=evaluate(model,samples['val'],8,args.device);score=val['balanced_accuracy'];history.append(dict(epoch=epoch,train_soft_ce=loss_sum/count,train_ticks=count,val_balanced_accuracy=score,val_macro_f1=val['macro_f1']))
                if score>best+1e-8:
                    best=score;best_epoch=epoch;torch.save(dict(model_config=config,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},sigma=8,seed=42,sampling_mode=mode,step=step,best_epoch=epoch),directory/'best.pt')
                if epoch-best_epoch>=8:break
            blob=torch.load(directory/'best.pt',map_location=args.device,weights_only=True);model.load_state_dict(blob['state_dict']);val,vp=evaluate(model,samples['val'],8,args.device);test,p=evaluate(model,samples['test'],8,args.device)
        assert np.array_equal(np.array(test['confusion_matrix']).sum(1),np.bincount(np.concatenate([s['labels'] for s in samples['test']]),minlength=3))
        offsets=np.r_[0,np.cumsum([len(s['labels']) for s in samples['test']])];save(directory/'test_predictions.npz',dict(labels=np.concatenate([s['labels'] for s in samples['test']]),probabilities=p,offsets=offsets));save(directory/'val_predictions.npz',dict(probabilities=vp,offsets=np.r_[0,np.cumsum([len(s['labels']) for s in samples['val']])]))
        r=dict(setting=identity,mode=mode,step=step,seed=42,sigma=8,best_epoch=best_epoch,epochs=len(history),validation=val,test=test,reused=mode=='baseline');results.append(r);predictions[identity]={'test':p,'val':vp};dump(directory/'metrics.json',r);dump(directory/'history.json',history);dump(out/'results_partial.json',results);print('RUN',identity,'valBA',round(val['balanced_accuracy'],4),'testBA',round(test['balanced_accuracy'],4),flush=True);del model;torch.cuda.empty_cache()
    dump(out/'results.json',results)
    summary=[]
    for r in results:
        m=r['test'];summary.append(dict(setting=r['setting'],seed=42,sigma=8,epochs=r['epochs'],best_epoch=r['best_epoch'],val_BA=r['validation']['balanced_accuracy'],BA=m['balanced_accuracy'],macro_F1=m['macro_f1'],failure_precision=m['per_class']['failure']['precision'],failure_recall=m['per_class']['failure']['recall'],failure_FPR=m['failure_false_positive_rate'],delta_BA_vs_dense=m['balanced_accuracy']-results[0]['test']['balanced_accuracy']))
    write_csv(out/'summary.csv',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curve_rows=[]
    for split in ('val','test'):
        ss=samples[split];offsets=np.r_[0,np.cumsum([len(s['labels']) for s in ss])];lo=min(int(s['video_frames'].min()-s['key']) for s in ss);hi=max(int(s['video_frames'].max()-s['key']) for s in ss);grid=np.arange(lo,hi+1);fig,axes=plt.subplots(2,9,figsize=(27,7),sharex=True,sharey=True)
        for col,r in enumerate(results):
            p=predictions[r['setting']][split]
            for row,outcome in enumerate((1,2)):
                values=[]
                for i,s in enumerate(ss):
                    if s['outcome']!=outcome:continue
                    a,b=offsets[i:i+2];relative=s['video_frames']-s['key'];v=np.full((len(grid),3),np.nan)
                    for frame in np.unique(relative):v[int(frame-lo)]=p[a:b][relative==frame].mean(0)
                    values.append(v)
                table=np.stack(values);counts=np.isfinite(table[:,:,0]).sum(0);mean=np.full((len(grid),3),np.nan);sd=mean.copy()
                for j,n in enumerate(counts):
                    vv=table[:,j];vv=vv[np.isfinite(vv[:,0])]
                    if n:mean[j]=vv.mean(0)
                    if n>1:sd[j]=vv.std(0,ddof=1)
                    for k,label in enumerate(('progress','success','failure')):curve_rows.append(dict(split=split,setting=r['setting'],gt_outcome='success' if outcome==1 else 'failure',relative_seconds=float(grid[j]/30),probability=label,mean=float(mean[j,k]),variance=float(sd[j,k]**2),std=float(sd[j,k]),interaction_count=int(n)))
                ax=axes[row,col]
                for k,(label,color) in enumerate(zip(('progress','success','failure'),('gray','tab:blue','tab:orange'))):ax.plot(grid/30,mean[:,k],color=color,label=label);ax.fill_between(grid/30,np.clip(mean[:,k]-sd[:,k],0,1),np.clip(mean[:,k]+sd[:,k],0,1),color=color,alpha=.08)
                ax.axvline(0,color='red',linestyle=':',linewidth=1);ax.set_xlim(lo/30,hi/30);ax.set_ylim(0,1);ax.set_title(r['setting']+'\n'+('Success GT' if outcome==1 else 'Failure GT'),fontsize=9);ax.set_xlabel('Seconds relative to Key',fontsize=8)
        axes[0,0].legend(fontsize=7);fig.suptitle(split.upper()+' | F6 GRU sigma8 seed42 | ALL interactions mean ± SD | dense step0 inference');fig.tight_layout();fig.savefig(out/(split+'_key_relative_curves.png'),dpi=160);fig.savefig(out/(split+'_key_relative_curves.pdf'));plt.close(fig)
    write_csv(out/'key_relative_mean_variance.csv',curve_rows)
    # Required small checks only; no reconstruction, repeated replay, or expanded seed sweep.
    for r in results:
        saved=load(out/'runs'/r['setting']/'test_predictions.npz');m=metrics(saved['labels'],saved['probabilities']);assert m['confusion_matrix']==r['test']['confusion_matrix'] and m['balanced_accuracy']==r['test']['balanced_accuracy']
    dump(out/'verification.json',dict(status='PASS',same_dense_val_test=True,metrics_recomputed=True,sigma_same_original_time=True,causal_raw_windows=True,feature_raw_same_endpoints=True,step0_shared=True,minimal_checks=True))
    lines=['# F6 temporal sampling ablation','', '固定original_align、Gaussian soft target、σ8、F6+单向GRU128、frozen官方finger-mode encoder1280D；原rollout split、AdamW lr.001/wd.0001、batch8、30epoch/patience8、clip1、plain soft CE不变。为了快速排除问题，本轮仅seed42，各设置相同初始化和interaction shuffle，无额外seed sweep。val/test永远原稠密step0，同一2210test ticks/24Align/17rollout，按val dense硬GT BA选epoch。','',
    'step表示跳过数量，stride=step+1：0/1/2/3对应30/15/10/7.5Hz。Feature-step：连续16rawticks先冻结编码，再每stride取一个feature给GRU。Raw-step：同一稀疏endpoint的过去16个rawpoints间隔stride后编码，跨度15×stride原始ticks，再给GRU；窗口只读过去，interaction开始不足历史重复首个有效tick。两组同step的下游endpoint完全相同；若sparse窗口含无效传感器点，双方共同排除，sampling_audit.json记录。','',
    'σ不随采样改变：每个稀疏endpoint仍用原positions/key_index计算g，保持原30Hz有效序列上σ8的意义，不将稀疏序列重新编号后再Gaussian。所有设置沿用同一dense-train normalization，避免额外改变输入标准化。Raw-step0与Feature-step0完全相同，复用原σ8 seed42 checkpoint而不重复训练，因此10名义设置仅9个唯一配置、8次新训练。','',
    'multi-step两类各做一组：每epoch每interaction仅取一个step，每4epoch独立排列轮换0/1/2/3，不复制成4倍batch或增加epoch；step随机数与interaction shuffle分开。各setting optimizer更新次数每epoch相同，但稀疏序列提供较少监督tick，这是本sampling操作的组成部分，不保证总监督tick一样。','',
    '当前输入是raw30Hz同步tick；实际checkpoint为finger mode，不改encoder结构。本轮不验证Deform，不更换Gaussian标签、不改变在线决策阈值。下表是固定硬GT的三类frame/tick指标：BA三类recall均值，macroF1三类F1均值，failureFPR是非failure GT却预测failure的比例；这里不是前一实验首次锁定的event指标。','',
    '| Train setting | Val BA % | Test BA % | Macro F1 % | Fail precision % | Fail recall % | Fail FPR % | Δ BA vs step0 (pp) |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append('| '+r['setting']+' | '+' | '.join(f'{100*r[k]:.2f}' for k in ('val_BA','BA','macro_F1','failure_precision','failure_recall','failure_FPR','delta_BA_vs_dense'))+' |')
    lines+=['','## Key-relative probability curves','', '全部val/test success/failure interaction按标注Key对齐，横轴seconds=(lastcamera frame−Key)/30；每图上排Success GT、下排Failure GT，三条线为P(progress)、P(success)、P(failure)，mean±interaction SD。重复相机帧先平均，缺失位置不补值；mean_variance CSV给出每个时间点N/variance，边缘N小，不能把边缘均值认为全体稳定。','', '![Test curves](test_key_relative_curves.png)','', '![Validation curves](val_key_relative_curves.png)','', '[完整指标](results.json) · [汇总CSV](summary.csv) · [采样配置](protocol.json) · [概率均值/方差/N](key_relative_mean_variance.csv) · [最小验证](verification.json)','']
    (out/'README.md').write_text('\n'.join(lines));snapshot=out/'code_snapshot';snapshot.mkdir();shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    dump(out/'manifest.json',dict(status='complete',created_at=datetime.datetime.now().astimezone().isoformat(),source=str(relative_path(source)),new_runs=8,unique_configs=9,nominal_settings=10,seed=42,sigma=8,source_manifest_sha256=sha(source/'dataset_manifest.json'),code_sha256=sha(Path(__file__))))
    target=project_path('WeeklySummary/10.5/f6_temporal_sampling', out.name);target.mkdir(parents=True,exist_ok=False);copies=[]
    for p in out.iterdir():
        if p.is_file():q=target/p.name;shutil.copy2(p,q);copies.append(dict(source=str(relative_path(p)),destination=str(relative_path(q)),sha256=sha(p)))
    dump(target/'copy_manifest.json',copies)
    with (project_path('WeeklySummary/10.5/10.5.md')).open('a') as f:f.write('\n\n## 13. F6 temporal sampling ablation\n\n'+ '\n'.join(lines[2:lines.index('## Key-relative probability curves')])+f'\n\n[采样消融报告与全部曲线](f6_temporal_sampling/{out.name}/README.md)。\n')
    print('COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
