"""Final Align failure anchor: frozen-Deform causal early-warning experiment."""
from .common import project_path, relative_path
from .common import canonical_key, is_stage
import argparse
import csv
import datetime
import json
import random
import shutil
from pathlib import Path

import numpy as np

from .common import ROOT, dump, sha

HORIZONS = (0, 8, 15, 30, 45)
SEEDS = (42, 43, 44, 45, 46)


def target(frames, key, failure, horizon):
    """Frame coordinates are camera frames; future frames never enter the input."""
    if not failure:
        return np.zeros(len(frames), np.int64), np.ones(len(frames), bool)
    if horizon == 0:
        return (frames >= key).astype(np.int64), np.ones(len(frames), bool)
    return ((frames >= key-horizon) & (frames < key)).astype(np.int64), frames < key


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(y, probability):
    y = np.asarray(y, int)
    p = np.asarray(probability)
    pred = p >= .5
    cm = np.array([[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)])
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(2), where=cm.sum(1)>0)
    precision = np.divide(np.diag(cm), cm.sum(0), out=np.zeros(2), where=cm.sum(0)>0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(2), where=precision+recall>0)
    # Tied scores are processed as one group. PR-AUC uses average precision.
    order = np.argsort(-p, kind='stable')
    sorted_y, sorted_p = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(y)-1]
    tp = np.cumsum(sorted_y)[ends].astype(float)
    fp = (ends+1)-tp
    positives, negatives = int(y.sum()), int((y == 0).sum())
    pr_auc = roc_auc = None
    if positives:
        rec = tp/positives
        pr_auc = float(np.sum(np.diff(np.r_[0, rec]) * tp/(tp+fp)))
        if negatives:
            roc_auc = float(np.trapz(np.r_[0, rec], np.r_[0, fp/negatives]))
    return dict(accuracy=float((pred == y).mean()), balanced_accuracy=float(recall.mean()),
                precision=float(precision[1]), recall=float(recall[1]), f1=float(f1[1]),
                macro_f1=float(f1.mean()), fpr=float(recall[0]*-1+1),
                pr_auc=pr_auc, roc_auc=roc_auc, confusion_matrix=cm.tolist(),
                positive=positives, negative=negatives)


def read_data(source, features):
    manifest = json.loads((source/'dataset_manifest.json').read_text())
    upstream = json.loads((project_path('outputs/sharpa_merged_online_datasets/20261005_164000/dataset_manifest.json')).read_text())
    original = {r['rollout_id']: r for r in upstream['rollouts']}
    samples = {s: [] for s in ('train', 'val', 'test')}
    ids = set()
    for row in manifest['records']:
        if row['group'] != 'merged_align':
            continue
        rid = row['rollout_id']
        assert rid not in ids
        ids.add(rid)
        align = sorted((e for e in original[rid]['events'] if is_stage(e, 'align')),
                       key=lambda e: (e['end_frame'], e['event_index']))
        final = align[-1]
        assert row['key'] == final['end_frame']
        assert (row['outcome'] == 2) == (canonical_key(final) == 2)
        with np.load(source/row['feature_path']) as z:
            sample = dict(row, frames=z['video_frames'].copy())
            if features:
                sample['deform'] = z['deform'].copy()
                assert sample['deform'].shape == (len(sample['frames']), 2560)
                assert np.isfinite(sample['deform']).all()
        assert np.all(np.diff(sample['frames']) >= 0)
        samples[row['split']].append(sample)
    assert len(ids) == 114
    assert sum(s['outcome'] == 2 for ss in samples.values() for s in ss) == 22
    assert [len(samples[s]) for s in ('train', 'val', 'test')] == [80, 17, 17]
    return samples


def distribution(samples):
    rows = []
    for h in HORIZONS:
        for split, ss in samples.items():
            ys, masks = zip(*(target(s['frames'], s['key'], s['outcome']==2, h) for s in ss))
            y = np.concatenate([v[k] for v, k in zip(ys, masks)])
            pos, neg = int(y.sum()), int((y==0).sum())
            assert pos > 0 and neg > 0
            rows.append(dict(horizon_frames=h, horizon_seconds=h/30, split=split,
                             rollouts=len(ss), failure_rollouts=sum(s['outcome']==2 for s in ss),
                             positive=pos, negative=neg, evaluated_ticks=len(y),
                             excluded_terminal_ticks=sum(int((~k).sum()) for k in masks),
                             positive_fraction=pos/len(y), negative_weight=len(y)/(2*neg),
                             positive_weight=len(y)/(2*pos)))
    return rows


def evaluate_arrays(samples, probabilities, offsets, h, split, seed):
    yy, pp, events = [], [], []
    event_gt, event_score = [], []
    leads, in_horizon = [], []
    for i, s in enumerate(samples):
        a, b = offsets[i:i+2]
        p = probabilities[a:b]
        assert len(p) == len(s['frames'])
        y, mask = target(s['frames'], s['key'], s['outcome']==2, h)
        yy.append(y[mask]); pp.append(p[mask])
        hit = np.flatnonzero(mask & (p >= .5))
        index = int(hit[0]) if len(hit) else None
        positive_hit = bool(np.any(mask & (y==1) & (p>=.5)))
        lead = (s['key']-int(s['frames'][index]))/30 if index is not None else None
        first_in_band = bool(index is not None and y[index] == 1)
        if s['outcome'] == 2:
            in_horizon.append(positive_hit)
            if lead is not None:
                leads.append(lead)
        event_gt.append(int(s['outcome']==2)); event_score.append(float(p[mask].max()))
        events.append(dict(split=split, horizon_frames=h, seed=seed, rollout_id=s['rollout_id'],
                           final_failure=int(s['outcome']==2), alarm=index is not None,
                           first_alarm_frame=int(s['frames'][index]) if index is not None else None,
                           anchor=s['key'], lead_seconds=lead, first_alarm_in_positive_band=first_in_band,
                           positive_band_detected=positive_hit, max_score=float(p[mask].max())))
    frame = metrics(np.concatenate(yy), np.concatenate(pp))
    event = metrics(event_gt, event_score)
    event['positive_band_detection_rate'] = float(np.mean(in_horizon))
    event['median_first_alarm_lead_seconds'] = float(np.median(leads)) if leads else None
    event['detected_failure_rollouts'] = len(leads)
    event['failure_rollouts'] = len(in_horizon)
    return frame, event, events


def train(args):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    print('LOADING_DEFORM_CACHE', flush=True)
    samples = read_data(args.source, True)
    print('CACHE_LOADED', flush=True)
    dist = distribution(samples)
    write_csv(out/'label_distribution.csv', dist)
    bank = np.concatenate([s['deform'] for s in samples['train']])
    mean = torch.from_numpy(bank.mean(0, dtype=np.float64).astype(np.float32))
    std = torch.from_numpy(np.maximum(bank.std(0, dtype=np.float64), .01).astype(np.float32))
    del bank
    dump(out/'protocol.json', dict(source=str(relative_path(args.source)),
         definition='only final Align failure end is irrecoverable anchor; success and earlier Keys never positive',
         horizons=list(HORIZONS), seeds=list(SEEDS), fps=30, threshold=.5,
         inputs='dense current Deform frozen feature2560 ->Linear128->single-layer causal GRU128->Linear1',
         loss='train class-balanced BCE: w0=N/(2*N0),w1=N/(2*N1), valid ticks only',
         terminal_mask='H>0 finalfailure t>=anchor excluded from training and main evaluation; H0 allticks',
         optimizer='AdamW lr.001 wd.0001 batch8 max30 patience8 gradclip1',
         selection='each horizon/seed best epoch by natural validation BA, ties earliest; no test selection',
         split={'train':80,'val':17,'test':17}, created_at=datetime.datetime.now().astimezone().isoformat()))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('mean', mean.clone()); self.register_buffer('std', std.clone())
            self.projection = nn.Linear(2560, 128)
            self.gru = nn.GRU(128, 128, batch_first=True)
            self.prediction = nn.Linear(128, 1)

        def forward(self, x):
            x = self.projection((x-self.mean)/self.std)
            x, _ = self.gru(x)
            return self.prediction(x).squeeze(-1)

    def batch(ss, h):
        lengths = [len(s['deform']) for s in ss]
        width = max(lengths)
        x = np.zeros((len(ss),width,2560),np.float32)
        y = np.zeros((len(ss),width),np.float32)
        mask = np.zeros_like(y, dtype=bool)
        for i,s in enumerate(ss):
            x[i,:lengths[i]] = s['deform']
            q,k = target(s['frames'],s['key'],s['outcome']==2,h)
            y[i,:lengths[i]],mask[i,:lengths[i]] = q,k
        return tuple(torch.from_numpy(v).to(args.device) for v in (x,y,mask)),lengths

    def infer(model, ss, h):
        model.eval(); probabilities=[]
        with torch.inference_mode():
            for a in range(0,len(ss),8):
                (x,_,_),lengths=batch(ss[a:a+8],h)
                p=model(x).sigmoid().cpu().numpy()
                probabilities.extend(p[i,:length] for i,length in enumerate(lengths))
        return np.concatenate(probabilities), np.r_[0,np.cumsum([len(p) for p in probabilities])]

    results=[]
    for h in HORIZONS:
        d=next(r for r in dist if r['split']=='train' and r['horizon_frames']==h)
        for seed in SEEDS:
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            model=Model().to(args.device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            rng=np.random.default_rng(seed)
            directory=out/'runs'/f'H{h}'/f'seed_{seed}'
            directory.mkdir(parents=True)
            best=-1;best_epoch=0;history=[]
            for epoch in range(1,31):
                model.train();order=rng.permutation(len(samples['train']));total=0;loss_sum=0
                for a in range(0,len(order),8):
                    ss=[samples['train'][int(i)] for i in order[a:a+8]]
                    (x,y,mask),_=batch(ss,h)
                    optimizer.zero_grad(set_to_none=True)
                    logit=model(x)
                    element=nn.functional.binary_cross_entropy_with_logits(logit,y,reduction='none')
                    weights=torch.where(y>0,d['positive_weight'],d['negative_weight'])
                    loss=(element*weights)[mask].mean()
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                    loss_sum+=float((element*weights)[mask].detach().sum());total+=int(mask.sum())
                p,offsets=infer(model,samples['val'],h)
                val,_,_=evaluate_arrays(samples['val'],p,offsets,h,'val',seed)
                history.append(dict(epoch=epoch,train_balanced_BCE=loss_sum/total,validation_BA=val['balanced_accuracy']))
                if val['balanced_accuracy']>best+1e-8:
                    best=val['balanced_accuracy'];best_epoch=epoch
                    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                    horizon=h,seed=seed,best_epoch=epoch),directory/'best.pt')
                if epoch-best_epoch>=8:
                    break
            model.load_state_dict(torch.load(directory/'best.pt',map_location=args.device,weights_only=True)['state_dict'])
            result=dict(horizon_frames=h,seed=seed,best_epoch=best_epoch,epochs=len(history))
            for split in ('val','test'):
                p,offsets=infer(model,samples[split],h)
                frame,event,events=evaluate_arrays(samples[split],p,offsets,h,split,seed)
                np.savez_compressed(directory/(split+'_predictions.npz'),probabilities=p,offsets=offsets)
                result[split]=frame;result[split+'_event']=event
                write_csv(directory/(split+'_events.csv'),events)
            dump(directory/'metrics.json',result);dump(directory/'history.json',history)
            results.append(result);dump(out/'results_partial.json',results)
            print('RUN',len(results),25,'H',h,'seed',seed,'valBA',round(result['val']['balanced_accuracy'],4),
                  'testBA',round(result['test']['balanced_accuracy'],4),flush=True)
            del model,optimizer
            torch.cuda.empty_cache()
    dump(out/'results.json',results)
    dump(out/'manifest.json',dict(status='complete',runs=25,horizons=list(HORIZONS),seeds=list(SEEDS),
                                 code_sha256=sha(Path(__file__)),completed_at=datetime.datetime.now().astimezone().isoformat()))
    print('TRAIN_COMPLETE',flush=True)


# def report(args):
#     out=args.output
#     results=json.loads((out/'results.json').read_text())
#     assert len(results)==25 and {(r['horizon_frames'],r['seed']) for r in results}=={(h,s) for h in HORIZONS for s in SEEDS}
#     samples=read_data(args.source,False)
#     assert list(csv.DictReader((out/'label_distribution.csv').open()))
#     summary=[];event_rows=[];confusions=[];curves=[]
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     figures={split:plt.subplots(2,5,figsize=(23,8)) for split in ('val','test')}
#     for hi,h in enumerate(HORIZONS):
#         chosen=[r for r in results if r['horizon_frames']==h]
#         row=dict(horizon_frames=h,horizon_seconds=h/30,seeds=5)
#         for phase in ('val','test','val_event','test_event'):
#             keys=['balanced_accuracy','precision','recall','f1','macro_f1','fpr','pr_auc','roc_auc']
#             if phase.endswith('event'):
#                 keys+=['positive_band_detection_rate','median_first_alarm_lead_seconds']
#             for key in keys:
#                 values=[r[phase][key] for r in chosen if r[phase][key] is not None]
#                 row[phase+'_'+key+'_mean']=float(np.mean(values)) if values else None
#                 row[phase+'_'+key+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else None
#                 if key=='median_first_alarm_lead_seconds':
#                     row[phase+'_lead_valid_seeds']=len(values)
#             for gt in (0,1):
#                 denominator=sum(sum(r[phase]['confusion_matrix'][gt]) for r in chosen)
#                 for prediction in (0,1):
#                     count=sum(r[phase]['confusion_matrix'][gt][prediction] for r in chosen)
#                     confusions.append(dict(horizon_frames=h,phase=phase,gt=gt,prediction=prediction,
#                                            mean_count_per_seed=count/5,row_normalized=count/denominator))
#         summary.append(row)
#         for split in ('val','test'):
#             ss=samples[split];ps=[];correct_first_rates=[]
#             for r in chosen:
#                 directory=out/'runs'/f'H{h}'/f"seed_{r['seed']}"
#                 with np.load(directory/(split+'_predictions.npz')) as z:
#                     p=z['probabilities'].copy();offsets=z['offsets'].copy()
#                 expected=np.r_[0,np.cumsum([len(s['frames']) for s in ss])]
#                 assert np.array_equal(offsets,expected)
#                 frame,event,events=evaluate_arrays(ss,p,offsets,h,split,r['seed'])
#                 assert frame['confusion_matrix']==r[split]['confusion_matrix']
#                 for key in ('balanced_accuracy','pr_auc','roc_auc'):
#                     assert np.isclose(frame[key],r[split][key])
#                 assert event['confusion_matrix']==r[split+'_event']['confusion_matrix']
#                 for key in ('balanced_accuracy','pr_auc','roc_auc'):
#                     assert np.isclose(event[key],r[split+'_event'][key])
#                 failures=[e for e in events if e['final_failure']]
#                 correct_first_rates.append(sum(e['first_alarm_in_positive_band'] for e in failures)/len(failures))
#                 event_rows.extend(events);ps.append(p)
#             row[split+'_event_correct_first_alarm_rate_mean']=float(np.mean(correct_first_rates))
#             row[split+'_event_correct_first_alarm_rate_std']=float(np.std(correct_first_rates,ddof=1))

#             ps = np.stack(ps, axis=0)  # [5, total_ticks]

#             rollout_fig_dir = out / 'per_rollout_risk' / split / f'H{h}'
#             rollout_fig_dir.mkdir(parents=True, exist_ok=True)

#             for i, s in enumerate(ss):
#                 outcome = s['outcome']

#                 a, b = offsets[i:i+2]
#                 relative = s['frames'] - s['key']

#                 # H > 0 failure: only show pre-anchor region.
#                 # Otherwise preserve the available trajectory, including H0 terminal tail.
#                 if h > 0 and outcome == 2:
#                     mask = relative <= 0
#                 else:
#                     continue
#                     mask = np.ones(len(relative), dtype=bool)

#                 relative_plot = relative[mask]

#                 # Each rollout can contain repeated camera-frame coordinates.
#                 # First merge repeated frames independently for every seed.
#                 unique_frames = np.unique(relative_plot)

#                 seed_curves = np.full(
#                     (len(ps), len(unique_frames)),
#                     np.nan,
#                     dtype=float
#                 )

#                 for seed_idx in range(len(ps)):
#                     rollout_p = ps[seed_idx, a:b]

#                     for j, f in enumerate(unique_frames):
#                         same_frame = mask & (relative == f)
#                         seed_curves[seed_idx, j] = rollout_p[same_frame].mean()

#                 # Across-seed statistics for THIS rollout only.
#                 mean = np.nanmean(seed_curves, axis=0)
#                 sd = np.nanstd(seed_curves, axis=0, ddof=1)

#                 x = unique_frames / 30.0

#                 fig, ax = plt.subplots(figsize=(9, 4.5))

#                 ax.plot(
#                     x,
#                     mean,
#                     label='5-seed mean'
#                 )

#                 ax.fill_between(
#                     x,
#                     np.clip(mean - sd, 0, 1),
#                     np.clip(mean + sd, 0, 1),
#                     alpha=0.20,
#                     label='seed SD'
#                 )

#                 ax.axhline(
#                     0.5,
#                     ls='--',
#                     color='gray',
#                     label='threshold = 0.5'
#                 )

#                 ax.axvline(
#                     0,
#                     ls=':',
#                     color='black',
#                     label='final Align end'
#                 )

#                 # Show the supervised positive horizon for failure rollouts.
#                 if outcome == 2:
#                     if h > 0:
#                         ax.axvspan(
#                             -h / 30.0,
#                             0,
#                             alpha=0.10,
#                             label=f'H={h}/30 s'
#                         )
#                     else:
#                         # H0: terminal region starts at anchor.
#                         if len(x) and x.max() > 0:
#                             ax.axvspan(
#                                 0,
#                                 x.max(),
#                                 alpha=0.10,
#                                 label='terminal region'
#                             )

#                 ax.set_ylim(0, 1)

#                 ax.set_xlabel('Time relative to final Align end (s)')
#                 ax.set_ylabel('Risk probability')

#                 outcome_name = 'failure' if outcome == 2 else 'success'

#                 ax.set_title(
#                     f'{split.upper()} | H={h}/30s | '
#                     f'rollout {i} | {outcome_name}'
#                 )

#                 ax.legend(fontsize=8)
#                 fig.tight_layout()

#                 fig.savefig(
#                     rollout_fig_dir / f'rollout_{i:03d}_{outcome_name}.png',
#                     dpi=160
#                 )

#                 plt.close(fig)
#     #         meanp=np.stack(ps).mean(0)
#     #         fig,axes=figures[split]
#     #         lo=min(int(s['frames'].min()-s['key']) for s in ss)
#     #         upper=max(int(s['frames'].max()-s['key']) for s in ss) if h==0 else 0
#     #         grid=np.arange(lo,upper+1)
#     #         for oi,outcome in enumerate((1,2)):
#     #             values=[]
#     #             for i,s in enumerate(ss):
#     #                 if s['outcome']!=outcome:
#     #                     continue
#     #                 a,b=offsets[i:i+2];relative=s['frames']-s['key']
#     #                 v=np.full(len(grid),np.nan)
#     #                 for f in np.unique(relative[relative<=upper]):
#     #                     v[int(f-lo)]=meanp[a:b][relative==f].mean()
#     #                 values.append(v)
#     #             values=np.stack(values);mean=np.full(len(grid),np.nan);sd=mean.copy()
#     #             for j,f in enumerate(grid):
#     #                 local=values[:,j];local=local[np.isfinite(local)]
#     #                 if len(local):mean[j]=local.mean()
#     #                 if len(local)>1:sd[j]=local.std(ddof=1)
#     #                 curves.append(dict(split=split,horizon_frames=h,final_outcome=outcome,relative_seconds=f/30,
#     #                                    mean=float(mean[j]),variance=float(sd[j]**2),N=len(local)))
#     #             ax=axes[oi,hi];ax.plot(grid/30,mean,color='tab:red')
#     #             ax.fill_between(grid/30,np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),alpha=.15,color='tab:red')
#     #             ax.axhline(.5,ls='--',color='gray');ax.axvline(0,ls=':',color='black')
#     #             if h and outcome==2:ax.axvspan(-h/30,0,alpha=.10)
#     #             if h==0 and outcome==2:ax.axvspan(0,upper/30,alpha=.10)
#     #             ax.set_ylim(0,1);ax.set_title(f'H={h}/30s | '+('success' if outcome==1 else 'failure'))
#     #             ax.set_xlabel('Time relative to final Align end (s)');ax.set_ylabel('Risk mean ± rollout SD')
#     # for split,(fig,axes) in figures.items():
#     #     fig.suptitle(split.upper()+' | 5-seed mean within rollout; H0 includes terminal tail')
#     #     fig.tight_layout();fig.savefig(out/(split+'_key_relative_risk.png'),dpi=160)
#     #     fig.savefig(out/(split+'_key_relative_risk.pdf'));plt.close(fig)
#     # write_csv(out/'summary.csv',summary);write_csv(out/'event_predictions.csv',event_rows)
#     # write_csv(out/'confusion_normalized.csv',confusions);write_csv(out/'key_relative_variance.csv',curves)
#     # dump(out/'verification.json',dict(status='PASS',runs25=True,seeds5=True,final_align_anchor_audited=True,
#     #                                 validation_and_test_metrics_recomputed=True,terminal_masks_checked=True,
#     #                                 rollout_split_unchanged=True,original_annotations_unchanged=True))
#     # lines=['# Deform causal early warning：最后 Align failure anchor','',
#     #        '用户定义：仅最后一个Align interval为failure(annotation2)时，以其end作为t_irrev。最终success(annotation1)整条为0；早期Align Key、全部Insert Key不产生正标签。原标注不修改，t_irrev是按用户规则定义的hindsight proxy，未重新人工验证物理不可恢复性。','',
#     #        '## 输入、GT与训练','',
#     #        '沿用114rollout连续缓存：从最早任意标注start到最后任意标注end；train80(64success/16failure)、val17(14/3)、test17(14/3)，无目标标注11条排除。只用dense当前帧Deform：冻结DeformEncoder→每指AdaptiveAvgPool(2×2)512D→五指2560D→Linear128→单层单向GRU128→Linear1→sigmoid。GRU仅rollout开始reset，保留早期attempt的历史。模型不读取未来tactile，未来anchor仅用于构建训练标签。','',
#     #        'H>0：failure的[anchor−H,anchor)为1，更早为0；anchor及之后从训练loss、val epoch选择和主test评估中排除。输入可保持整段，但单向GRU的后续输入不能影响前面输出。Success全段0。H=0独立对照：[anchor,观察范围结束]为1，更早0，全部参与。H=0是持续terminal标签，不是上一实验的Key pulse。若早期interval的帧落入最终anchor的预警带，仍可因最终anchor标1，但不会建立早期anchor。','',
#     #        'H={0,8,15,30,45}相机帧，30fps，对应{0,0.2667,0.5,1,1.5}s；0.25s按建议近似为8帧。每H均seeds42–46。训练balanced BCEWithLogits：w0=N/(2N0)、w1=N/(2N1)，计数只用各H训练有效tick；每个有效tick的BCE乘其权重后平均。不改变val/test自然分布、不做Gaussian或step augmentation。Class-balanced BCE改变训练先验，因此sigmoid作为未校准risk score，未做自然分布概率校准。','',
#     #        'Train-only feature standardization；AdamW lr.001/wd.0001，batch8rollout，max30epoch/patience8，clip1。每H/seed仅按自然分布val BA选best epoch，平分取最早；阈值固定0.5，无test选阈值或H。','',
#     #        '## 评估定义','',
#     #        'BA为两类recall平均；precision/recall/F1均指positive intervention风险类，FPR=FP/(FP+TN)。PR-AUC使用average precision（按score tie分组、step积分），ROC-AUC为标准ROC梯形积分。两者不依赖0.5阈值。各H的GT和有效范围不同，不能仅按BA横向断言预警能力。','',
#     #        'Event alarm：在主评估有效范围任意risk≥0.5视为报警；与固定最终rollout failure/success比较。H>0仅failure anchor前，success全段。另报positive-band detection rate：最终failure中至少在其正标签时段命中过一次的比例。过早报警可计入event alarm，却会是帧级FP；positive-band detection可来自后续报警，所以同时保存first_alarm_in_positive_band。','',
#     #        'Lead time=(anchor−首次有效报警帧)/30；median只对发生报警的最终failure计算，未报警不填0，同时报告event recall。H>0 lead为正，可能超过H，此时首次报警不符合within-H target；H=0可为负（anchor后才触发），其首次过早报警也会单独标记。5seed表是每seed指标的均值±样本SD，lead是每seed中位数的均值±SD，不是把未检测样本剔除后宣称整体提前量。','',
#     #        '## Test，5 seeds','',
#     #        '| H实际秒 | BA % | Precision % | Recall % | F1 % | FPR % | PR-AUC % | ROC-AUC % | Event recall % | 正时段event命中 % | 首报在正时段 % | 首报lead秒 |',
#     #        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
#     def cell(r,key,percent=True):
#         v,s=r[key+'_mean'],r[key+'_std']
#         if v is None:return '无报警'
#         scale=100 if percent else 1
#         return f'{scale*v:.2f} ± {scale*s:.2f}' if s is not None else f'{scale*v:.2f}'
#     for r in summary:
#         keys=['test_balanced_accuracy','test_precision','test_recall','test_f1','test_fpr','test_pr_auc','test_roc_auc','test_event_recall','test_event_positive_band_detection_rate','test_event_correct_first_alarm_rate']
#         lines.append('| '+f"{r['horizon_seconds']:.4f}"+' | '+' | '.join(cell(r,k) for k in keys)+' | '+cell(r,'test_event_median_first_alarm_lead_seconds',False)+' |')
#     lines+=['','![Validation Key-relative risk](val_key_relative_risk.png)','',
#             '![Test Key-relative risk](test_key_relative_risk.png)','',
#             '曲线先每rollout平均5seeds，再在各时间对有数据的rollout平均±rollout SD；N和variance在CSV，边缘少样本需谨慎。H>0图只展示anchor前及anchor时刻以对照早期风险，其anchor时刻不参与指标；H=0图额外保留terminal tail以观察持续状态检测。','',
#             '[完整25runs](results.json) · [Val/Test全部指标](summary.csv) · [各H有效正负样本](label_distribution.csv) · [逐rollout首次报警/lead](event_predictions.csv) · [数量与归一化矩阵](confusion_normalized.csv) · [曲线variance和N](key_relative_variance.csv) · [验证](verification.json)','']
#     counts=list(csv.DictReader((out/'label_distribution.csv').open()))
#     count_lines=['## 有效帧分布与PR基线','',
#                  '| H帧 | Train正/负 | Val正/负 | Test正/负 | Test正例比例（常数score AP基线） |',
#                  '|---:|---:|---:|---:|---:|']
#     for h in HORIZONS:
#         dd={r['split']:r for r in counts if int(r['horizon_frames'])==h}
#         count_lines.append('| '+str(h)+' | '+' | '.join(dd[s]['positive']+'/'+dd[s]['negative'] for s in ('train','val','test'))+' | '+f"{100*float(dd['test']['positive_fraction']):.2f}%"+' |')
#     count_lines+=['','H>0的最终failure anchor后帧已排除；success全段保留。具体排除数量与train-only class weights见CSV。','']
#     index=lines.index('## Test，5 seeds')
#     lines[index:index]=count_lines
#     (out/'README.md').write_text('\n'.join(lines))
#     snap=out/'code_snapshot';snap.mkdir();shutil.copy2(Path(__file__),snap/Path(__file__).name)
#     shutil.copy2(ROOT/'WeeklySummary/10.5/10.5criticalreward.md',snap/'experiment_setting.md')
#     target_dir=ROOT/'WeeklySummary/10.5/early_warning'/out.name
#     target_dir.mkdir(parents=True,exist_ok=False);copies=[]
#     for p in out.iterdir():
#         if p.is_file():
#             q=target_dir/p.name;shutil.copy2(p,q)
#             digest=sha(p);assert sha(q)==digest
#             copies.append(dict(source=str(p.relative_to(ROOT)),destination=str(q.relative_to(ROOT)),sha256=digest))
#     dump(target_dir/'copy_manifest.json',copies)
#     with (ROOT/'WeeklySummary/10.5/10.5.md').open('a') as f:
#         start=lines.index('## Test，5 seeds')
#         stop=lines.index('![Validation Key-relative risk](val_key_relative_risk.png)')
#         f.write('\n\n## 17. Deform early warning：最后Align failure anchor\n\n'+
#                 '仅最终Align failure end为anchor，早期和success Key不产生正标签。冻结Deform2560D→Linear128→causal GRU128→Linear1，train-balanced BCE，原rollout split；H>0仅anchor前目标带为1且terminal帧不参与loss/main evaluation，H0为anchor后持续1。25runs，5seeds，未按test选择H。\n\n'+
#                 '\n'.join(lines[start:stop])+f'\n\n[实验17完整报告、分布、曲线及归一化矩阵](early_warning/{out.name}/README.md)。\n')
#     print('REPORT_COMPLETE',json.dumps(summary),flush=True)
def report(args):
    out = args.output

    print('REPORT START', flush=True)
    samples = read_data(args.source, False)
    print('DATA LOADED', flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    WINDOW_SECONDS = 5
    WINDOW_FRAMES = WINDOW_SECONDS * 30

    total = 0

    for h in HORIZONS:
        for split in ('val', 'test'):
            ss = samples[split]

            # Load predictions from all 5 seeds.
            seed_predictions = []
            offsets = None

            for seed in SEEDS:
                directory = out / 'runs' / f'H{h}' / f'seed_{seed}'

                with np.load(directory / f'{split}_predictions.npz') as z:
                    p = z['probabilities'].copy()
                    current_offsets = z['offsets'].copy()

                expected = np.r_[
                    0,
                    np.cumsum([len(s['frames']) for s in ss])
                ]
                assert np.array_equal(current_offsets, expected)

                if offsets is None:
                    offsets = current_offsets
                else:
                    assert np.array_equal(offsets, current_offsets)

                seed_predictions.append(p)

            # [5 seeds, total ticks]
            ps = np.stack(seed_predictions, axis=0)

            rollout_fig_dir = (
                out / 'per_rollout_risk' / split / f'H{h}'
            )
            rollout_fig_dir.mkdir(parents=True, exist_ok=True)

            for i, s in enumerate(ss):
                outcome = s['outcome']

                a, b = offsets[i:i + 2]
                frames = np.asarray(s['frames'])
                key = int(s['key'])

                # Camera-frame coordinates relative to final Align end.
                relative = frames - key

                # Every rollout must contain the full 5 s history.
                assert relative.min() <= -WINDOW_FRAMES, (
                    f"{s['rollout_id']} does not contain "
                    f"{WINDOW_SECONDS}s before final Align end: "
                    f"earliest={relative.min() / 30:.3f}s"
                )

                # Fixed [-5 s, 0 s] observation window.
                mask = (
                    (relative >= -WINDOW_FRAMES) &
                    (relative <= 0)
                )

                relative_plot = relative[mask]
                unique_frames = np.unique(relative_plot)

                # [5 seeds, unique camera frames]
                seed_curves = np.full(
                    (len(SEEDS), len(unique_frames)),
                    np.nan,
                    dtype=float
                )

                for seed_idx in range(len(SEEDS)):
                    rollout_p = ps[seed_idx, a:b]

                    for j, f in enumerate(unique_frames):
                        same_frame = mask & (relative == f)
                        seed_curves[seed_idx, j] = (
                            rollout_p[same_frame].mean()
                        )

                # Across-seed statistics for this rollout.
                mean = np.nanmean(seed_curves, axis=0)
                sd = np.nanstd(seed_curves, axis=0, ddof=1)

                x = unique_frames / 30.0

                fig, ax = plt.subplots(figsize=(9, 4.5))

                ax.plot(
                    x,
                    mean,
                    label='5-seed mean'
                )

                ax.fill_between(
                    x,
                    np.clip(mean - sd, 0, 1),
                    np.clip(mean + sd, 0, 1),
                    alpha=0.20,
                    label='seed SD'
                )

                ax.axhline(
                    0.5,
                    ls='--',
                    color='gray',
                    label='threshold = 0.5'
                )

                # t=0 is always final Align end.
                ax.axvline(
                    0,
                    ls=':',
                    color='black',
                    label='final Align end'
                )

                # Positive horizon for failure rollouts.
                if outcome == 2 and h > 0:
                    ax.axvspan(
                        -h / 30.0,
                        0,
                        alpha=0.10,
                        label=f'H={h}/30 s'
                    )

                # Fixed axes for every rollout.
                ax.set_xlim(-WINDOW_SECONDS, 0)
                ax.set_ylim(0, 1)

                ax.set_xlabel(
                    'Time relative to final Align end (s)'
                )
                ax.set_ylabel('Risk probability')

                outcome_name = (
                    'failure' if outcome == 2 else 'success'
                )

                ax.set_title(
                    f'{split.upper()} | H={h}/30s | '
                    f'rollout {i} | {outcome_name}'
                )

                ax.legend(fontsize=8)
                fig.tight_layout()

                path = (
                    rollout_fig_dir /
                    f'rollout_{i:03d}_{outcome_name}.png'
                )

                fig.savefig(path, dpi=160)
                plt.close(fig)

                total += 1
                print(
                    f'[{total}] {path.relative_to(out)}',
                    flush=True
                )

    print(
        f'PER_ROLLOUT_REPORT_COMPLETE: {total} figures',
        flush=True
    )

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['train','report'])
    p.add_argument('--source',type=Path,default=project_path('outputs/sharpa_gaussian_online_data/20261005_182000'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    (train if args.action=='train' else report)(args)


if __name__=='__main__':
    main()
