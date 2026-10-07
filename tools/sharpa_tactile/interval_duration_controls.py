"""Deform interval3/4: duration controls and independently trained prefixes."""
from .common import project_path, relative_path
import argparse
import copy
import datetime
import json
import random
import shutil
from pathlib import Path

import numpy as np

from .common import ROOT, dump, sha
from .early_warning import metrics, write_csv
from .failure_relabel import load_samples, SEEDS


SOURCE = project_path('outputs/sharpa_failure_relabel/20261006_165000')
AUDIT = project_path('outputs/sharpa_interval_padding_audit/20261007_161501')
VARIANTS = ('length_matched','normalized_K16','normalized_K32',
            'prefix25','prefix50','prefix75',
            'prefix25_K16','prefix50_K16','prefix75_K16',
            'prefix25_K32','prefix50_K32','prefix75_K32')


def duration(s):
    return int(s['end_frame']-s['start_frame']+1)


def pair_lengths(ss, bounds, caliper=3):
    """Maximum-cardinality one-to-one matching, then minimum frame distance."""
    from scipy.optimize import linear_sum_assignment
    groups = [[s for s in ss if s['label']==label and bounds[0]<=duration(s)<=bounds[1]]
              for label in (0,1)]
    if not all(groups):
        raise ValueError('Empty duration-overlap class')
    distance = np.abs(np.array([duration(s) for s in groups[0]])[:,None]-
                      np.array([duration(s) for s in groups[1]])[None,:])
    # The penalty exceeds the sum of every admissible distance: matching an
    # extra legal edge always outranks any improvement in total distance.
    penalty = (min(distance.shape)+1)*(caliper+1)
    cost = np.where(distance<=caliper,distance,penalty)
    ii,jj = linear_sum_assignment(cost)
    pairs=[]
    for i,j in zip(ii,jj):
        if distance[i,j]<=caliper:
            a,b=groups[0][int(i)],groups[1][int(j)]
            pairs.append(dict(event3_sample_id=a['sample_id'],event4_sample_id=b['sample_id'],
                              duration3=duration(a),duration4=duration(b),gap=int(distance[i,j])))
    selected={p[k] for p in pairs for k in ('event3_sample_id','event4_sample_id')}
    return [s for s in ss if s['sample_id'] in selected],pairs


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    raw=load_samples(SOURCE,'interval_3vs4')
    train_durations=[[duration(s) for s in raw['train'] if s['label']==y] for y in (0,1)]
    bounds=(max(min(v) for v in train_durations),min(max(v) for v in train_durations))
    selected={};pairs=[]
    for split,ss in raw.items():
        part,pp=pair_lengths(ss,bounds)
        selected[split]=[s['sample_id'] for s in part]
        pairs.extend(dict(split=split,**p) for p in pp)
    write_csv(out/'length_matches.csv',pairs)
    dump(out/'matched_samples.json',selected)
    dump(out/'protocol.json',dict(source=str(relative_path(SOURCE)),
         annotation='frozen original interval3vs4 snapshot, not refreshed3/4 annotations',
         source_manifest_sha256=sha(SOURCE/'dataset_manifest.json'),
         source_split_sha256=sha(SOURCE/'split_manifest.json'),
         event_mapping={'3':0,'4':1},seeds=list(SEEDS),no_test=True,
         split='unchanged 19 train / 4 validation rollout assignment; matched group retains a subset',
         matched=dict(train_only_overlap_bounds=list(bounds),caliper_frames=3,
                      matching='within each split; maximum legal pair count then minimum sum distance; one-to-one, no replacement',
                      val='same train bounds and predefined caliper; val labels/lengths used only for cohort matching, not rule tuning'),
         variants=list(VARIANTS),
         prefix='camera cutoff start+ceil(fraction*(end-start+1))-1; retain only cached valid frames<=cutoff',
         normalized='piecewise linear interpolation of observed Deform features on K uniform camera-frame coordinates between first/last retained frame; duplicates averaged; singleton repeated',
         caveat='offline fraction uses known annotated interval duration; not a live online endpoint selection protocol',
         model='frozen Deform5x512 cached2560 -> Linear128 -> unidirectional single GRU128 -> last-valid hidden -> Linear1; sigmoid P(event4)',
         normalization='train only, each interval equal weight; std floor.01; recalculated per variant',
         loss='one balanced BCE per interval; w_c=N/(2*Nc), counts from each variant training subset',
         optimizer='AdamW lr.001 weight_decay.0001 batch8 clip1',
         training=dict(max_epochs=30,patience=8,checkpoint='max natural validation BA; ties earliest',threshold=.5),
         reuse='original5 Deform checkpoints supply A, raw100% prefix; normalized full supplies normalized100% prefix; length-only audit reused',
         created_at=datetime.datetime.now().astimezone().isoformat()))
    for name in ('split_manifest.json','source_intervals.json','protocol.json','dataset_manifest.json'):
        shutil.copy2(SOURCE/name,out/('original_'+name))
    for name in ('length_only_results.json','length_only_predictions.csv'):
        shutil.copy2(AUDIT/name,out/name)
    source_hashes=[dict(path=str(relative_path(SOURCE/name)),sha256=sha(SOURCE/name))
                   for name in ('split_manifest.json','source_intervals.json','dataset_manifest.json')]
    dump(out/'source_hashes.json',source_hashes)
    distribution=[]
    for variant in ('original',)+VARIANTS:
        samples=construct(raw,variant,selected)
        for split,ss in samples.items():
            for label in (0,1):
                part=[s for s in ss if s['label']==label]
                distribution.append(dict(variant=variant,split=split,event_key=label+3,
                     intervals=len(part),rollouts=len({s['rollout_id'] for s in part}),
                     original_duration_min=min(duration(s) for s in part),
                     original_duration_median=float(np.median([duration(s) for s in part])),
                     original_duration_max=max(duration(s) for s in part),
                     input_length_min=min(len(s['deform']) for s in part),
                     input_length_median=float(np.median([len(s['deform']) for s in part])),
                     input_length_max=max(len(s['deform']) for s in part)))
    write_csv(out/'sample_distribution.csv',distribution)
    print('PREPARE_COMPLETE',json.dumps(selected),flush=True)


def construct(raw,variant,selected):
    samples={}
    for split,ss in raw.items():
        part=[]
        for original in ss:
            if variant=='length_matched' and original['sample_id'] not in selected[split]:
                continue
            s={k:v for k,v in original.items() if k not in ('f6','deform','frames','y')}
            x=original['deform'];frames=original['frames']
            if variant.startswith('prefix'):
                fraction=int(variant.split('_')[0].replace('prefix',''))/100
                cutoff=original['start_frame']+int(np.ceil(fraction*duration(original)))-1
                mask=frames<=cutoff
                if not mask.any():
                    raise ValueError(f'Prefix has no valid observation: {original["sample_id"]}')
                x=x[mask];frames=frames[mask]
                s['prefix_cutoff_frame']=cutoff
                assert frames[-1]<=cutoff
            if '_K' in variant:
                k=int(variant.rsplit('_K',1)[1])
                unique,inverse=np.unique(frames,return_inverse=True)
                aggregate=np.zeros((len(unique),x.shape[1]),np.float64)
                np.add.at(aggregate,inverse,x)
                aggregate/=np.bincount(inverse)[:,None]
                positions=np.linspace(unique[0],unique[-1],k)
                right=np.searchsorted(unique,positions,side='right').clip(1,max(1,len(unique)-1))
                if len(unique)==1:
                    x=np.repeat(aggregate,k,axis=0).astype(np.float32)
                else:
                    left=right-1
                    weight=((positions-unique[left])/(unique[right]-unique[left]))[:,None]
                    x=((1-weight)*aggregate[left]+weight*aggregate[right]).astype(np.float32)
                frames=positions
            s.update(deform=x,frames=frames,y=np.array([s['label']],np.float32))
            part.append(s)
        samples[split]=part
    return samples


def train(args):
    import torch
    from torch import nn
    torch.set_num_threads(2)
    raw=load_samples(SOURCE,'interval_3vs4')
    selected=json.loads((args.output/'matched_samples.json').read_text())
    for variant in args.variants:
        samples=construct(raw,variant,selected)
        ss=samples['train']
        mean=np.mean([s['deform'].mean(0,dtype=np.float64) for s in ss],axis=0)
        second=np.mean([(s['deform'].astype(np.float64)**2).mean(0) for s in ss],axis=0)
        std=np.maximum(np.sqrt(np.maximum(second-mean**2,0)),.01)
        ytrain=np.array([s['label'] for s in ss])
        weights=(len(ytrain)/(2*int((ytrain==0).sum())),len(ytrain)/(2*int((ytrain==1).sum())))
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('deform_mean',torch.from_numpy(mean.astype(np.float32)))
                self.register_buffer('deform_std',torch.from_numpy(std.astype(np.float32)))
                self.projections=nn.ModuleDict({'deform':nn.Linear(2560,128)})
                self.gru=nn.GRU(128,128,batch_first=True)
                self.head=nn.Linear(128,1)
            def forward(self,x,lengths):
                v=self.projections['deform']((x-self.deform_mean)/self.deform_std)
                h,_=self.gru(v)
                last=h[torch.arange(len(lengths),device=h.device),torch.tensor(lengths,device=h.device)-1]
                return self.head(last).squeeze(-1)
        def batch(part):
            lengths=[len(s['deform']) for s in part]
            x=np.zeros((len(part),max(lengths),2560),np.float32)
            for i,s in enumerate(part):x[i,:lengths[i]]=s['deform']
            return torch.from_numpy(x).to(args.device),torch.tensor([s['label'] for s in part],device=args.device,dtype=torch.float32),lengths
        def infer(model,part):
            model.eval();result=[]
            with torch.inference_mode():
                for a in range(0,len(part),8):
                    x,_,lengths=batch(part[a:a+8]);result.extend(model(x,lengths).sigmoid().cpu().tolist())
            return np.array(result)
        for seed in SEEDS:
            directory=args.output/'runs'/variant/f'seed_{seed}'
            directory.mkdir(parents=True,exist_ok=False)
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            model=Model().to(args.device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            rng=np.random.default_rng(seed);best=-1;best_epoch=0;history=[]
            yval=np.array([s['label'] for s in samples['val']])
            for epoch in range(1,31):
                model.train();order=rng.permutation(len(ss));loss_sum=0;total=0
                for a in range(0,len(order),8):
                    part=[ss[int(i)] for i in order[a:a+8]];x,y,lengths=batch(part)
                    optimizer.zero_grad(set_to_none=True)
                    logit=model(x,lengths)
                    weighted=nn.functional.binary_cross_entropy_with_logits(logit,y,reduction='none')*torch.where(y>0,weights[1],weights[0])
                    loss=weighted.mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                    loss_sum+=float(weighted.detach().sum());total+=len(part)
                p=infer(model,samples['val']);val=metrics(yval,p)
                clamped=np.clip(p,1e-7,1-1e-7)
                val_loss=float((-(yval*np.log(clamped)+(1-yval)*np.log1p(-clamped))*np.where(yval>0,weights[1],weights[0])).mean())
                history.append(dict(epoch=epoch,train_balanced_bce=loss_sum/total,validation_balanced_bce=val_loss,validation_BA=val['balanced_accuracy']))
                if val['balanced_accuracy']>best+1e-8:
                    best=val['balanced_accuracy'];best_epoch=epoch
                    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                    variant=variant,seed=seed,best_epoch=epoch),directory/'best.pt')
                if epoch-best_epoch>=8:break
            model.load_state_dict(torch.load(directory/'best.pt',map_location=args.device,weights_only=True)['state_dict'])
            p=infer(model,samples['val']);val=metrics(yval,p)
            cm=np.asarray(val['confusion_matrix']);val['confusion_row_normalized']=(cm/cm.sum(1,keepdims=True)).tolist()
            result=dict(variant=variant,seed=seed,best_epoch=best_epoch,epochs=len(history),
                        train_intervals=len(ss),val_intervals=len(yval),val=val)
            dump(directory/'metrics.json',result);dump(directory/'history.json',history)
            np.savez_compressed(directory/'val_predictions.npz',probabilities=p,labels=yval)
            dump(directory/'val_samples.json',[{k:v for k,v in s.items() if k not in ('deform','frames','y')} for s in samples['val']])
            print('RUN',variant,seed,'BA',round(val['balanced_accuracy'],4),'AUC',round(val['roc_auc'],4),flush=True)
            del model,optimizer;torch.cuda.empty_cache()
    print('TRAIN_COMPLETE',args.variants,flush=True)


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    out=args.output
    raw=load_samples(SOURCE,'interval_3vs4')
    selected=json.loads((out/'matched_samples.json').read_text())
    for name in ('split_manifest.json','source_intervals.json','protocol.json','dataset_manifest.json'):
        assert sha(SOURCE/name)==sha(out/('original_'+name))
    results=[];histories={};prediction_rows=[];checks=[]
    for variant in ('original',)+VARIANTS:
        samples=construct(raw,variant,selected)
        yy=np.array([s['label'] for s in samples['val']])
        for seed in SEEDS:
            if variant=='original':
                directory=SOURCE/'runs/interval_3vs4/deform'/f'seed_{seed}'
            else:directory=out/'runs'/variant/f'seed_{seed}'
            r=json.loads((directory/'metrics.json').read_text())
            r.update(variant=variant,seed=seed,train_intervals=len(samples['train']),val_intervals=len(yy))
            p=np.load(directory/'val_predictions.npz')['probabilities']
            assert len(p)==len(yy)
            replay=metrics(yy,p)
            for k in ('balanced_accuracy','macro_f1','roc_auc','accuracy'):
                assert abs(replay[k]-r['val'][k])<1e-8
            history=json.loads((directory/'history.json').read_text())
            ba=np.array([h['validation_BA'] for h in history])
            assert int(np.argmax(ba))+1==r['best_epoch']
            assert r['epochs']<=30 and (r['epochs']==30 or r['epochs']-r['best_epoch']==8)
            assert not {s['rollout_id'] for s in samples['train']} & {s['rollout_id'] for s in samples['val']}
            for i,s in enumerate(samples['val']):
                prediction_rows.append(dict(variant=variant,seed=seed,sample_id=s['sample_id'],
                    rollout_id=s['rollout_id'],label=int(yy[i]),probability=float(p[i]),
                    original_duration=duration(s),input_length=len(s['deform'])))
            histories[(variant,seed)]=history;results.append(r)
            checks.append(dict(variant=variant,seed=seed,metric_replay=True,earlystop_verified=True,
                               checkpoint_sha256=sha(directory/'best.pt')))
    dump(out/'results.json',results);dump(out/'verification.json',dict(runs=65,new_runs=60,reused_runs=5,
         labels_replayed=True,earlystop_verified=True,split_unchanged=True,source_unchanged=True,no_test=True,
         checks=checks))
    write_csv(out/'val_interval_predictions.csv',prediction_rows)
    summary=[]
    for variant in ('original',)+VARIANTS:
        rr=[r for r in results if r['variant']==variant]
        row=dict(variant=variant,seeds=5,train_intervals=rr[0]['train_intervals'],val_intervals=rr[0]['val_intervals'])
        for k in ('balanced_accuracy','macro_f1','roc_auc','accuracy','precision','recall','fpr'):
            values=[r['val'][k] for r in rr];row[k+'_mean']=float(np.mean(values));row[k+'_sd']=float(np.std(values,ddof=1))
        summary.append(row)
    write_csv(out/'summary.csv',summary)

    # Matched-cohort length control: its duration distribution need not be
    # perfectly identical, so quantify the remaining single-feature signal.
    matched=construct(raw,'length_matched',selected)
    xt=np.log1p([duration(s) for s in matched['train']]);xv=np.log1p([duration(s) for s in matched['val']])
    yt=np.array([s['label'] for s in matched['train']]);yv=np.array([s['label'] for s in matched['val']])
    mean,std=xt.mean(),max(xt.std(),1e-8)
    lr=LogisticRegression(C=1,class_weight='balanced',random_state=42,max_iter=1000).fit(((xt-mean)/std)[:,None],yt)
    matched_length=metrics(yv,lr.predict_proba(((xv-mean)/std)[:,None])[:,1])
    dump(out/'matched_length_only.json',matched_length)

    def lookup(v):return next(r for r in summary if r['variant']==v)
    figures=out/'figures';figures.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,metric,title in zip(axs,('balanced_accuracy','roc_auc'),('Balanced Accuracy','ROC-AUC')):
        for suffix,label,color in (('','Original-rate prefix','C0'),('_K16','Normalized prefix K=16','C1'),('_K32','Normalized prefix K=32','C2')):
            series=[lookup('prefix'+str(p)+suffix) if p<100 else lookup('original' if suffix=='' else 'normalized'+suffix) for p in (25,50,75,100)]
            ax.errorbar((25,50,75,100),[r[metric+'_mean'] for r in series],
                        yerr=[r[metric+'_sd'] for r in series],marker='o',label=label,color=color,capsize=3)
        ax.axhline(.5,color='gray',ls=':');ax.set(xlabel='Observed interval (%)',ylabel=title,xticks=[25,50,75,100],ylim=(0,1.04));ax.legend(fontsize=8)
    fig.suptitle('Deform + GRU: prefix separability (5 seeds, mean ± SD)')
    for ext in ('png','pdf'):fig.savefig(figures/f'prefix_curves.{ext}',dpi=180)
    plt.close(fig)

    fig,axs=plt.subplots(3,4,figsize=(16,10),layout='constrained')
    for ax,variant in zip(axs.flat,VARIANTS):
        for seed in SEEDS:
            hh=histories[(variant,seed)]
            ax.plot([h['epoch'] for h in hh],[h['train_balanced_bce'] for h in hh],color='C0',alpha=.3)
            ax.plot([h['epoch'] for h in hh],[h['validation_balanced_bce'] for h in hh],color='C1',alpha=.3)
        ax.set(title=variant,xlabel='Epoch',ylabel='Balanced BCE')
    fig.suptitle('Training loss (blue) / validation loss (orange); stopping remains validation BA')
    for ext in ('png','pdf'):fig.savefig(figures/f'loss_curves.{ext}',dpi=160)
    plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,variant in zip(axs,('original','normalized_K16','length_matched')):
        rr=[r for r in results if r['variant']==variant]
        cm=np.mean([np.asarray(r['val']['confusion_matrix'],float) for r in rr],axis=0)
        norm=cm/cm.sum(1,keepdims=True)
        ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        for i in (0,1):
            for j in (0,1):ax.text(j,i,f'{norm[i,j]*100:.1f}%\n({cm[i,j]:.1f})',ha='center',va='center')
        ax.set(title=variant,xticks=[0,1],yticks=[0,1],xticklabels=['event3','event4'],yticklabels=['event3','event4'],xlabel='Prediction',ylabel='GT')
    for ext in ('png','pdf'):fig.savefig(figures/f'confusions.{ext}',dpi=180)
    plt.close(fig)

    protocol=json.loads((out/'protocol.json').read_text())
    matched_n=lookup('length_matched')
    lines=['# 3 vs 4 interval separability：duration control 与 prefix','',
      f'固定原 interval3vs4 数据快照 `{relative_path(SOURCE)}`，19 train / 4 val rollout，event3→0、event4→1。只使用 Deform + GRU；不使用后续补标，也不新分test。Original与length-only复用已完成结果，新增12种输入 × seeds42–46，共60次训练。','',
      '## 模型、标签、loss 与输出','',
      '冻结 DeformEncoder → 每指pool2×2得到512D → 5指拼成2560D → Linear128 → 单层单向GRU128 → last-valid hidden → Linear1 → sigmoid P(event4)。每段interval重置GRU；每段仅贡献一次balanced BCE，权重按该组train interval计数 `w_c=N/(2N_c)`，3=0/4=1。阈值0.5。每组训练各自的模型；归一化仅按该组train计算，每段interval等权，std下限0.01。','',
      '沿用原interval实验：AdamW lr0.001、weight decay0.0001、batch8、clip1、max30 epoch、patience8；按val BA选best checkpoint，平手最早。此前max300/loss-patience50的修改针对critical/threshold，不套用到这个保持原设置的对照。padding使用原last-valid逻辑，已由独立审计验证。val用于epoch选择，以下不属于独立test结果。','',
      '## 输入构造','',
      'A Original：完整已标注闭区间的有效Deform序列，保留30Hz时序（沿用原记录的有效tick，少量无效tick缺失不补齐）；原5个checkpoint直接复用。','',
      'B Length-only：只输入原始区间相机帧数；训练集拟合balanced logistic或train-BA阈值stump，不用tactile，val不调参。原结果分别BA72.22%和83.33%。','',
      f'C Length-matched：train两类长度共同支持范围为{protocol["matched"]["train_only_overlap_bounds"]}帧。预先固定最大pair长度差3帧（0.1s），在每个split内做无放回1对1匹配：先最大化合法pair数，再最小化总长度差；保留原始30Hz序列，不裁切或插值。val应用train界限和固定caliper，匹配只用val的标签与长度，不根据模型表现筛样。最终train {matched_n["train_intervals"]}段，val {matched_n["val_intervals"]}段；具体pair与剩余rollout见length_matches.csv和sample_distribution.csv。C使用更小且有条件筛选的val，与全量val结果不能直接视为同一评估集。','',
      'D Time-normalized：完整interval的有效Deform feature按相机帧坐标分段线性插值到K=16或32个等间距点。只有一个观测时重复该点，重复相机帧先平均。只输入feature，不输入原始长度、时间戳、插值坐标或mask特征；每段GRU有效update次数相同。插值可能保留与时长相关的动作速率/平滑度线索，因此去掉的是显式绝对长度和update次数，不能保证所有duration相关信号完全消失。','',
      'E Prefix：每个比例25/50/75%单独训练；相机截止帧为 `start+ceil(p*(end-start+1))-1`，只读取截止前的有效Deform帧，标签继承完整interval。100%复用A。归一化prefix另外将每个已截断prefix独立插值到K16/K32，禁止从prefix之后读取tactile；100%分别复用D。','',
      'Prefix比例需要完整标注区间长度，属于offline separability评估；不是部署时已知“25%位置”的在线检测策略。不同prefix模型独立训练，曲线不是同一个模型逐步读入更多数据后的在线曲线。','',
      '## 结果（5 seeds均值 ± 样本SD）','',
      '| 输入 | Train/Val interval数 | BA | Macro F1 | ROC-AUC |','|---|---:|---:|---:|---:|']
    for r in summary:
        fmt=lambda k:f'{r[k+"_mean"]*100:.2f}±{r[k+"_sd"]*100:.2f}%'
        lines.append(f'| {r["variant"]} | {r["train_intervals"]}/{r["val_intervals"]} | {fmt("balanced_accuracy")} | {fmt("macro_f1")} | {fmt("roc_auc")} |')
    lines+=['',f'匹配子集的独立length-only logistic：val BA {matched_length["balanced_accuracy"]*100:.2f}%，ROC-AUC {matched_length["roc_auc"]*100:.2f}%；用来检查残余长度信号。','',
      'BA为event3/event4 recall均值；Macro F1为两类F1均值；ROC-AUC用连续P(event4)计算，与0.5阈值无关。每段interval一票，长段不加权。原全量val仅12段（3类3段、4类9段）来自4条rollout；length-matched更少。seed SD只量化优化波动，不增加独立验证样本。','',
      '![Prefix BA and AUC](figures/prefix_curves.png)','',
      '![Normalized confusion matrices](figures/confusions.png)','',
      '![Loss curves](figures/loss_curves.png)','',
      '## 解释边界','',
      '应优先比较A与D：二者使用完全相同train/val interval，D固定update次数，区别集中在时间归一化。C额外改变训练规模和验证样本组成，只作为duration控制后的补充证据。原始prefix仍含绝对prefix长度，需和对应K固定的prefix共同解释。各K/比例结果完整列出，不用val再挑一个配置宣称最佳泛化。','',
      '原始清单与split备份为original_*，source_hashes.json记录来源。summary.csv、results.json、sample_distribution.csv、length_matches.csv、val_interval_predictions.csv及runs/*保存逐组指标、预测、checkpoint与loss曲线。verification.json核对65组指标回放、早停、split和checkpoint来源；其中5组为复用baseline。']
    text='\n'.join(lines)+'\n'
    (out/'README.md').write_text(text)
    weekly=project_path('WeeklySummary/10.5/interval_duration_controls', out.name)
    weekly.mkdir(parents=True,exist_ok=False)
    for name in ('README.md','protocol.json','summary.csv','results.json','sample_distribution.csv',
                 'length_matches.csv','matched_samples.json','matched_length_only.json',
                 'val_interval_predictions.csv','verification.json','length_only_results.json',
                 'length_only_predictions.csv','source_hashes.json'):
        shutil.copy2(out/name,weekly/name)
    shutil.copytree(figures,weekly/'figures')
    page=project_path('WeeklySummary/10.5/10.5interval_duration_controls.md')
    page.write_text(text.replace('(figures/','(interval_duration_controls/'+out.name+'/figures/')+
                    '\n原始输出：`'+str(relative_path(out))+'`。\n')
    print('REPORT_COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','report'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    args=parser.parse_args();args.output=args.output.resolve()
    if args.action=='prepare':prepare(args.output)
    else:globals()[args.action](args)
