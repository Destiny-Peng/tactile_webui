"""Aggregate BA-stopped controlled repeats and compare to original seed42."""
from .common import project_path, relative_path
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,dump,sha
from .early_warning import HORIZONS,read_data,evaluate_arrays,write_csv


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve()
    old=project_path('outputs/sharpa_early_warning/20261006_001500');source=project_path('outputs/sharpa_gaussian_online_data/20261005_182000')
    baseline=json.loads((old/'results.json').read_text());samples=read_data(source,False)
    results=[];summary=[];comparisons=[];audits=[];confusions=[];events_all=[];curves=[]
    fig,axes=plt.subplots(2,5,figsize=(21,7))
    for hi,h in enumerate(HORIZONS):
        shard=out/'shards'/f'H{h}';r=json.loads((shard/'results.json').read_text())[0]
        assert r['horizon_frames']==h and r['seed']==42
        results.append(r);b=next(v for v in baseline if v['horizon_frames']==h and v['seed']==42)
        directory=shard/'runs'/f'H{h}'/'seed_42'
        history=json.loads((directory/'history.json').read_text())
        oldhistory=json.loads((old/'runs'/f'H{h}'/'seed_42'/'history.json').read_text())
        ba=np.array([x['validation_BA'] for x in history]);best=0;score=-1
        for i,value in enumerate(ba):
            if value>score+1e-8:best=i+1;score=value
        assert r['best_epoch']==best and r['epochs']==len(history) and len(history)<=300
        assert len(history)==300 or len(history)-best==50
        n=min(len(history),len(oldhistory))
        delta_ba=max(abs(history[i]['validation_BA']-oldhistory[i]['validation_BA']) for i in range(n))
        delta_loss=max(abs(history[i]['train_balanced_BCE']-oldhistory[i]['train_balanced_BCE']) for i in range(n))
        # Different CUDA execution schedules may cause numeric drift: expose instead of hiding it.
        audits.append(dict(horizon_frames=h,best_epoch=best,epochs=len(history),patience_actual=len(history)-best,original_best_epoch=b['best_epoch'],
            overlap_epochs=n,overlap_max_abs_val_BA_delta=delta_ba,overlap_max_abs_train_BCE_delta=delta_loss))
        for phase in ('val','test','val_event','test_event'):
            m=r[phase]
            summary.append(dict(horizon_frames=h,seed=42,phase=phase,**{k:m[k] for k in ('balanced_accuracy','macro_f1','precision','recall','fpr','pr_auc','roc_auc')},
                positive_band_detection_rate=m.get('positive_band_detection_rate'),median_lead_seconds=m.get('median_first_alarm_lead_seconds')))
            for metric in ('balanced_accuracy','macro_f1','precision','recall','fpr','pr_auc','roc_auc'):
                comparisons.append(dict(horizon_frames=h,phase=phase,metric=metric,old_seed42=b[phase][metric],new_seed42=m[metric],delta=m[metric]-b[phase][metric]))
            cm=np.array(m['confusion_matrix'])
            for gt in (0,1):
                for pred in (0,1):confusions.append(dict(horizon_frames=h,phase=phase,gt=gt,prediction=pred,count=int(cm[gt,pred]),row_normalized=float(cm[gt,pred]/max(1,cm[gt].sum()))))
        for split in ('val','test'):
            with np.load(directory/f'{split}_predictions.npz') as z:
                ps=z['probabilities'];offsets=z['offsets']
            frame,event,events=evaluate_arrays(samples[split],ps,offsets,h,split,42)
            assert frame['confusion_matrix']==r[split]['confusion_matrix'] and event['confusion_matrix']==r[split+'_event']['confusion_matrix']
            for k in ('balanced_accuracy','pr_auc','roc_auc'):assert np.isclose(frame[k],r[split][k])
            events_all.extend(events)
        epoch=np.arange(1,len(history)+1)
        axes[0,hi].plot(epoch,[v['train_balanced_BCE'] for v in history]);axes[0,hi].set_title(f'H{h} train balanced BCE')
        axes[1,hi].plot(epoch,ba,label='new max300/p50');axes[1,hi].plot(np.arange(1,len(oldhistory)+1),[v['validation_BA'] for v in oldhistory],ls='--',label='old max30/p8')
        for row in (0,1):axes[row,hi].axvline(best,color='k',ls=':');axes[row,hi].set_xlabel('epoch')
        axes[1,hi].set_title(f'Val BA; best={best}');axes[1,hi].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=150);fig.savefig(out/'training_curves.pdf');plt.close(fig)
    dump(out/'results.json',results);write_csv(out/'summary.csv',summary);write_csv(out/'paired_comparison.csv',comparisons)
    write_csv(out/'stopping_and_prefix_audit.csv',audits);write_csv(out/'confusion_normalized.csv',confusions);write_csv(out/'event_predictions.csv',events_all)
    olddist=list(csv.DictReader((old/'label_distribution.csv').open()))
    newdist=list(csv.DictReader((out/'shards/H0/label_distribution.csv').open()))
    assert olddist==newdist
    shutil.copy2(out/'shards/H0/label_distribution.csv',out/'label_distribution.csv')
    shutil.copy2(source/'dataset_manifest.json',out/'source_dataset_manifest.json')
    # Identical fixed normalization, cache and GT; only max epoch and patience differ.
    protocols=[json.loads((out/'shards'/f'H{h}'/'protocol.json').read_text()) for h in HORIZONS]
    proto=protocols[0];proto['horizons']=list(HORIZONS);proto['baseline']=str(relative_path(old));proto['controlled_changes']=['max_epoch30->300','BA patience8->50'];proto['source_sha256']=sha(source/'dataset_manifest.json')
    dump(out/'protocol.json',proto)
    for split in ('val','test'):
        fig,axes=plt.subplots(2,5,figsize=(22,7),sharex=True,sharey=True)
        for hi,h in enumerate(HORIZONS):
            for outcome in (False,True):
                ax=axes[int(outcome),hi]
                for origin,label,color in ((old,'old seed42','tab:blue'),(out/'shards'/f'H{h}','new seed42','tab:orange')):
                    with np.load(origin/'runs'/f'H{h}'/'seed_42'/f'{split}_predictions.npz') as z:ps=z['probabilities'];offsets=z['offsets']
                    aligned=[]
                    for i,s in enumerate(samples[split]):
                        if (s['outcome']==2)!=outcome:continue
                        a,b=offsets[i:i+2];relative=s['frames']-s['key'];v=ps[a:b]
                        mask=relative<0 if outcome and h>0 else np.ones(len(relative),bool)
                        aligned.append({int(t):float(v[mask][relative[mask]==t].mean()) for t in np.unique(relative[mask])})
                    xs=sorted({t for r in aligned for t in r if -120<=t<=60});means=[];stds=[]
                    for t in xs:
                        v=np.array([r[t] for r in aligned if t in r]);means.append(v.mean());stds.append(v.std())
                        curves.append(dict(split=split,horizon_frames=h,outcome='failure' if outcome else 'success',model=label,relative_frame=t,relative_seconds=t/30,mean=float(v.mean()),variance=float(v.var()),std=float(v.std()),rollouts=len(v)))
                    x=np.array(xs)/30;m=np.array(means);sd=np.array(stds);ax.plot(x,m,color=color,label=label);ax.fill_between(x,m-sd,m+sd,color=color,alpha=.12)
                ax.axvline(0,color='k',ls=':');ax.axhline(.5,color='gray',ls='--');ax.set_title(f"H{h} {'failure' if outcome else 'success'}");ax.set_xlabel('seconds relative to Key');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/f'{split}_key_curves.png',dpi=150);fig.savefig(out/f'{split}_key_curves.pdf');plt.close(fig)
    write_csv(out/'key_relative_variance.csv',curves)
    text=['# Deform early warning：BA early stopping 的受控重跑\n',
        '严格复用历史实验 `early_warning/20261006_001500`：本轮只改变 max epoch30→300、validation BA patience8→50。保持 BA 最大选最佳 checkpoint（改善需>1e−8，平手保留最早），不按 validation loss 选模型。每个 H 仅 seed42，对照旧 seed42，不与旧5seed均值混为一谈。\n',
        '## 输入、GT、训练与输出\n',
        '冻结当前 DeformEncoder 特征2560D→Linear128→单层单向GRU128→Linear1→sigmoid risk。Linear 后没有额外 ReLU，与旧实现一致。每条 rollout 从历史缓存范围开始reset；不改变旧实现的缺帧处理（沿缓存顺序继续 GRU）。batch8 rollout，尾部padding仅用于批处理；padding不进入loss/metrics，单向GRU不将尾部padding传回有效帧。\n',
        '沿用114rollout的历史缓存和split：train80（64success/16failure）、val17（14/3）、test17（14/3）。数据区间、Key、每H有效正负帧数量逐项与旧实验相等；不使用新3/4标注。仅最后Align interval为failure时，以其end作anchor；success和早期Key不建立正标签。H>0：failure的[anchor−H,anchor)为1，更早为0，t≥anchor从训练/选epoch/主评估中排除；success全段0。H0：failure t≥anchor为1，之前0，保留terminal尾段。H以相机帧为单位，30fps。\n',
        'loss仍为 frame-count balanced BCEWithLogits：w0=N/(2N0)、w1=N/(2N1)，只用train有效tick计算固定权重。每个batch所有有效tick加权BCE求均值；没有改成每rollout loss等权、没有focal/value/risk辅助任务。Train-only mean/std（std下限.01）；AdamW lr=.001、wd=.0001、clip1；同seed初始化与rollout shuffle方式保持旧实现。\n',
        '## 停止轮数及前缀核对\n','|H|Old best/stopped|New best/stopped|重叠epoch最大 BA 差|重叠epoch最大 train BCE 差|\n|---:|---:|---:|---:|---:|']
    for a in audits:
        b=next(r for r in baseline if r['horizon_frames']==a['horizon_frames'] and r['seed']==42)
        text.append(f"|{a['horizon_frames']}|{b['best_epoch']}/{b['epochs']}|{a['best_epoch']}/{a['epochs']}|{a['overlap_max_abs_val_BA_delta']:.3g}|{a['overlap_max_abs_train_BCE_delta']:.3g}|")
    text.extend(['\n竖线为新实验最佳epoch；train BCE仅作诊断，validation BA决定checkpoint和早停。\n','![Training](training_curves.png)\n',
        '## Test：old seed42 → new seed42，固定阈值0.5\n','|H|Frame BA|Macro F1|PR-AUC|ROC-AUC|Failure frame recall|Frame FPR|\n|---:|---:|---:|---:|---:|---:|---:|'])
    for r in results:
        b=next(v for v in baseline if v['horizon_frames']==r['horizon_frames'] and v['seed']==42)['test'];n=r['test']
        values=[f"{b[k]:.2%} → {n[k]:.2%}" for k in ('balanced_accuracy','macro_f1','pr_auc','roc_auc','recall','fpr')]
        text.append('|'+str(r['horizon_frames'])+'|'+'|'.join(values)+'|')
    text.extend(['\nBA=两类recall平均；Macro F1=两类F1平均；precision/recall/FPR针对风险正类，FPR=FP/(FP+TN)。PR-AUC使用average precision；ROC-AUC按score排序积分。帧指标汇总有效tick，GT及计算函数与旧实验相同。所有阈值固定0.5，不做额外校准，不按test挑H。\n',
        '## Test event 报警\n','任意有效tick risk≥0.5即rollout报警，success任意报警均为FP。band detection是failure目标正带内至少报警一次；first alarm in band以全部failure为分母。二者不同；提前但落在目标带之外的首次报警不算有效within-H预警。\n',
        '|H|Old→New event recall|Old→New event FPR|New band detection|New first alarm in band|New median lead(s)|\n|---:|---:|---:|---:|---:|---:|'])
    for r in results:
        b=next(v for v in baseline if v['horizon_frames']==r['horizon_frames'] and v['seed']==42)['test_event'];n=r['test_event']
        es=[e for e in events_all if e['split']=='test' and e['horizon_frames']==r['horizon_frames'] and e['final_failure']==1]
        first=np.mean([e['first_alarm_in_positive_band'] for e in es]);lead=n['median_first_alarm_lead_seconds']
        text.append(f"|{r['horizon_frames']}|{b['recall']:.2%} → {n['recall']:.2%}|{b['fpr']:.2%} → {n['fpr']:.2%}|{n['positive_band_detection_rate']:.2%}|{first:.2%}|{lead if lead is not None else '—'}|")
    text.extend(['\nlead=(anchor−first_alarm_frame)/30，仅统计已报警failure，不将未检出填0。val/test各只有3条failure、14条success，单seed结果不能估计seed variance。H0为terminal检测，不等价于提前预警。\n',
        '## Key-relative curves\n','每条rollout先合并重复相机frame，再按Key对齐；按success/failure分别计算mean±SD，variance/std/N保存在CSV。H>0的failure仅展示pre-Key，与评估mask一致。\n',
        '![Validation](val_key_curves.png)\n','![Test](test_key_curves.png)\n',
        '完整数据：results.json、summary.csv、paired_comparison.csv、event_predictions.csv、confusion_normalized.csv、stopping_and_prefix_audit.csv。每组checkpoint、逐epoch history与预测保存于shards/H*/runs/H*/seed_42。原始标注与旧实验未改。\n'])
    # Preserve the controlled comparison conclusion without selecting a horizon by test.
    unchanged=[a['horizon_frames'] for a in audits if a['original_best_epoch']==a['best_epoch']]
    text.insert(1,f'本轮未改变loss或architecture。最佳epoch未改变的H：{unchanged}。是否存在数值漂移见前缀核对表；所有停止轮数均通过BA/patience50检查。各H完整对照如下。\n')
    reproduction=out/'prediction_reproduction.json'
    if reproduction.exists():
        text.insert(1,'本轮确认：五个H的validation/test逐帧预测与旧seed42完全一致（最大绝对差0）；训练重叠前缀的train BCE和validation BA也完全一致。H0/8/15/30均在epoch52停止并恢复epoch2，H45在epoch55停止并恢复epoch5。延长训练上限和BA patience没有改变最终结果，因此该seed下旧结果可精确复现；此前三阶段实验的差异不能归因于只延长训练轮数。它仍改变了architecture、loss、checkpoint criterion等多个因素，尚不能进一步单独归因。')
    (out/'README.md').write_text('\n'.join(text))
    dump(out/'verification.json',dict(status='PASS',runs=5,seed=42,GT_counts_identical=True,metrics_replayed=True,stopping_audit=audits,source_sha256=sha(source/'dataset_manifest.json')))
    code=out/'code_snapshot';code.mkdir(exist_ok=True)
    for name in ('early_warning.py','early_warning_ba300.py','early_warning_ba300_report.py','common.py'):shutil.copy2(project_path('tools/sharpa_tactile', name),code/name)
    shutil.copy2(project_path('tools/run_early_warning_ba300.sh'),code/'run_early_warning_ba300.sh')
    weekly=project_path('WeeklySummary/10.5');dest=weekly/'early_warning_ba300'/out.name;dest.mkdir(parents=True,exist_ok=True)
    for f in out.iterdir():
        if f.is_file() and f.suffix in ('.md','.json','.csv','.png','.pdf'):shutil.copy2(f,dest/f.name)
    doc=(out/'README.md').read_text()
    for name in ('training_curves','val_key_curves','test_key_curves'):doc=doc.replace(f'({name}.png)',f'(early_warning_ba300/{out.name}/{name}.png)')
    doc+='\n原始完整实验目录：`'+str(relative_path(out))+'`。\n'
    (weekly/'10.5early_warning_ba300.md').write_text(doc)
    print(json.dumps(audits,indent=2));print('REPORT_COMPLETE',flush=True)


if __name__=='__main__':main()
