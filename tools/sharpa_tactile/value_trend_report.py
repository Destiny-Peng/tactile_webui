"""Publish completed tactile value-trend experiment; no training or test selection."""
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
from .early_warning import write_csv, read_data, evaluate_arrays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
    rows=[];results=[];stopping=[];confusions=[]
    samples=read_data(project_path('outputs/sharpa_gaussian_online_data/20261005_182000'),False)
    for h in (0,8,15,30,45):
        result=json.loads((out/'runs'/f'H{h}'/'result.json').read_text())
        for split in ('val','test'):
            with np.load(out/'runs'/f'H{h}'/f'{split}_predictions.npz') as archive:
                # Float64 comparison preserves thresholds just above max score (all-negative option).
                adjusted=archive['probabilities'].astype(np.float64)-result['calibration']['threshold']+.5
                _,event,events=evaluate_arrays(samples[split],adjusted,archive['offsets'],h,split,42)
                result[split]['calibrated_event']=event
                write_csv(out/'runs'/f'H{h}'/f'{split}_calibrated_events.csv',events)
            cm=np.array(result[split]['frame']['confusion_matrix'])
            for label in (0,1):
                for predicted in (0,1):
                    confusions.append(dict(horizon_frames=h,split=split,gt=label,prediction=predicted,count=int(cm[label,predicted]),row_normalized=float(cm[label,predicted]/max(1,cm[label].sum()))))
            for suffix,level in (('', 'event'),('_calibrated','calibrated_event')):
                eventpath=out/'runs'/f'H{h}'/f'{split}{suffix}_events.csv'
                events=list(csv.DictReader(eventpath.open()))
                failures=[e for e in events if int(e['final_failure'])==1]
                result[split][level]['first_alarm_in_positive_band_rate']=float(np.mean([e['first_alarm_in_positive_band']=='True' for e in failures]))
                if suffix:
                    with np.load(out/'runs'/f'H{h}'/f'{split}_predictions.npz') as z:
                        for i,e in enumerate(events):
                            a,b=z['offsets'][i:i+2];mask=(z['frames'][a:b]<z['keys'][i]) if z['failure'][i] and h>0 else np.ones(b-a,bool)
                            e['max_score']=float(z['probabilities'][a:b][mask].max());e['threshold']=result['calibration']['threshold']
                    write_csv(eventpath,events)
        dump(out/'runs'/f'H{h}'/'result.json',result)
        results.append(result)
        for split in ('val','test'):
            for level in ('frame','event','calibrated_event'):
                m=result[split][level]
                rows.append(dict(horizon_frames=h,split=split,level=level,threshold=.5 if level!='calibrated_event' else result['calibration']['threshold'],
                    **{k:m[k] for k in ('balanced_accuracy','macro_f1','precision','recall','fpr','pr_auc','roc_auc')},
                    positive_band_detection_rate=m.get('positive_band_detection_rate'),first_alarm_lead_seconds=m.get('median_first_alarm_lead_seconds')))
    dump(out/'results.json',results);write_csv(out/'summary.csv',rows);write_csv(out/'confusion_normalized.csv',confusions)
    dirs=[out/'proxy',out/'future']+[out/'runs'/f'H{h}' for h in (0,8,15,30,45)]
    fig,axes=plt.subplots(2,4,figsize=(17,7));axes=axes.ravel()
    for ax,d in zip(axes,dirs):
        hist=list(csv.DictReader((d/'history.csv').open()));train=json.loads((d/'training.json').read_text())
        epochs=np.array([int(r['epoch']) for r in hist]);val=np.array([float(r['validation_loss']) for r in hist]);start=5 if d.name=='proxy' else 0
        best=int(np.argmin(val[start:]))+start+1
        assert train['best_epoch']==best and len(hist)<=300
        assert len(hist)==300 or len(hist)-best==50
        stopping.append(dict(stage=d.name,epochs=len(hist),best_epoch=best,best_validation_loss=float(val[best-1]),patience_actual=len(hist)-best))
        ax.plot(epochs,[float(r['train_loss']) for r in hist],label='train');ax.plot(epochs,val,label='val')
        ax.axvline(best,color='k',ls=':',alpha=.6);ax.set_title(d.name);ax.set_xlabel('epoch');ax.legend(fontsize=8)
    axes[-1].axis('off');fig.tight_layout();fig.savefig(out/'loss_curves.png',dpi=160);fig.savefig(out/'loss_curves.pdf');plt.close(fig)
    write_csv(out/'stopping_audit.csv',stopping)
    # Key alignment: average duplicate camera frames first, then equal weight per rollout.
    curve_rows=[]
    for split in ('val','test'):
        fig,axes=plt.subplots(5,4,figsize=(18,17),sharex=True)
        for hi,h in enumerate((0,8,15,30,45)):
            with np.load(out/'runs'/f'H{h}'/f'{split}_predictions.npz') as z:
                offsets=z['offsets'];keys=z['keys'];failure=z['failure'];frames=z['frames']
                values={k:z[k] for k in ('proxy','q','risk','probabilities')}
                for ki,(name,vec) in enumerate(values.items()):
                    ax=axes[hi,ki]
                    for outcome,color in ((False,'tab:blue'),(True,'tab:red')):
                        aligned=[]
                        for i,fail in enumerate(failure):
                            if fail!=outcome:continue
                            a,b=offsets[i:i+2];relative=frames[a:b]-keys[i];vv=vec[a:b]
                            # H>0 retains pre-Key frames for failure, exactly matching evaluation mask.
                            mask=(relative<0) if fail and h>0 else np.ones(len(relative),bool)
                            aligned.append({int(t):float(vv[mask][relative[mask]==t].mean()) for t in np.unique(relative[mask])})
                        xs=sorted({t for row in aligned for t in row if -120<=t<=60})
                        means=[];stds=[]
                        for t in xs:
                            a=np.array([r[t] for r in aligned if t in r]);means.append(float(a.mean()));stds.append(float(a.std()))
                            curve_rows.append(dict(split=split,horizon_frames=h,outcome='failure' if outcome else 'success',quantity=name,relative_frame=t,relative_seconds=t/30,mean=float(a.mean()),variance=float(a.var()),std=float(a.std()),rollouts=len(a)))
                        x=np.array(xs)/30;mean=np.array(means);std=np.array(stds)
                        ax.plot(x,mean,color=color,label='failure' if outcome else 'success');ax.fill_between(x,mean-std,mean+std,color=color,alpha=.14)
                    ax.axvline(0,color='k',ls=':');ax.set_title(f'H{h} {name}');ax.set_xlabel('time relative to Key (s)');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/f'{split}_key_curves.png',dpi=140);fig.savefig(out/f'{split}_key_curves.pdf');plt.close(fig)
    write_csv(out/'key_relative_variance.csv',curve_rows)
    # Full normalized-progress proxy curves reveal collapse separately from risk metrics.
    feature_files=list((out/'features').glob('*.npz'))
    manifest=json.loads((out/'source_dataset_manifest.json').read_text());records=[r for r in manifest['records'] if r['group']=='merged_align']
    value_quality=[]
    for split in ('train','val','test'):
        for fail in (False,True):
            group=[r for r in records if r['split']==split and (r['outcome']==2)==fail]
            values=[];mse=[];drops=[];terminal=[]
            for r in group:
                with np.load(out/'features'/f"{r['rollout_id']}.npz") as z:
                    v=z['proxy'];fr=z['frames'];ticks=z['ticks'];gt=np.zeros(len(v)) if fail else (fr-fr[0])/max(1,fr[-1]-fr[0])
                    values.append(float(v.mean()));mse.append(float(np.mean((v-gt)**2)));terminal.append(float(v[-1]));drops.append(float((np.diff(v)[np.diff(ticks)==1]<-1e-4).mean()))
            value_quality.append(dict(split=split,outcome='failure' if fail else 'success',rollouts=len(group),mean_value=float(np.mean(values)),terminal_value=float(np.mean(terminal)),progress_MSE=float(np.mean(mse)),downward_fraction=float(np.mean(drops))))
    write_csv(out/'proxy_quality.csv',value_quality)
    # Same seed, encoder and split; training objectives AND checkpoint selection differ.
    baseline=json.loads((project_path('outputs/sharpa_early_warning/20261006_001500/results.json')).read_text())
    paired=[]
    for r in results:
        b=next(v for v in baseline if v['seed']==42 and v['horizon_frames']==r['horizon_frames'])
        for split in ('val','test'):
            for metric in ('balanced_accuracy','macro_f1','pr_auc','roc_auc','recall','precision','fpr'):
                paired.append(dict(horizon_frames=r['horizon_frames'],split=split,metric=metric,baseline=b[split][metric],value_trend=r[split]['frame'][metric]))
    write_csv(out/'baseline_comparison.csv',paired)
    text=['# Tactile value → future value → trend risk / focal intervention\n',
        '本轮使用历史 final Align Key，不使用最新 3/4 标注。仅最后一个 Align failure 的 end 是 anchor；早期 failure、success Key 不产生正标签。\n',
        '采用 Deform、seed42、H={0,8,15,30,45}。保持历史 rollout split：train80（64success/16failure）、val17（14/3）、test17（14/3）。所有阶段仅由 validation loss 选 checkpoint，max300、patience50；test 仅最终评估。\n',
        '## 数据、GT 与输出\n',
        '复用冻结 DeformEncoder 的当前 tick2560D 特征，输入仅包含 tactile history，不包含时间、总长度、Key 或 outcome。使用历史缓存的 merged interaction 范围（首个 Align start 至已标注尾段）；不等于完整原始录像。缺失 tactile tick 处重置 GRU，TD 不跨缺口，补齐帧不参与 loss。归一化仅由 train 计算。\n',
        '阶段1：Deform2560→Linear128/ReLU→单向GRU128→双 sigmoid value，V=min(V1,V2)。成功 interaction 的 progress=(frame-start)/(end-start)，失败 terminal return=0；progress 回归/单调项只用于成功轨迹。loss=TD SmoothL1（γ=.99）+0.3×progress SmoothL1+monotonic hinge+0.05×失败状态 CQL。前5 epoch 仅 warmup progress/monotonic，不参与最佳 checkpoint 选择。CQL 使用 failed-state log-mean-exp 减 observed-state mean，与 log-sum-exp 相差固定常数，因此总 loss 可能为负，不能当成概率或准确率。成功/失败轨迹均衡采样，每轨迹 loss 先平均，再累积8轨迹更新。\n',
        '阶段2：冻结 proxy。独立 Deform2560→Linear128/ReLU→单向GRU128→MLP→预测下一 tick 的2560D latent；latent→双 sigmoid value→取min。loss=归一化 latent MSE（单位向量平方距离）+双头 SmoothL1(next proxy value)+辅助 current-value SmoothL1。不跨缺失 tick 监督下一帧。\n',
        '阶段3：冻结前两阶段。取最近9个 proxy value 与8个差分→MLP32，拼接预测 future latent→Linear128；共享 MLP128 同时输出连续 risk 与 trigger logit。risk target=(1−Vt)×Σ(i=0..7)0.9^i×(0.005−ΔV(t−i))，保留有符号公式；不足历史用段首 value 补齐，全部因果。loss=SmoothL1(risk)+binary focal(trigger, α=.75, γ=2)。输出 sigmoid(trigger)=报警分数；连续 risk 回归值不直接当报警概率。\n',
        '按用户确认，binary trigger 不采用论文 value-trend 挖掘标签，而继续使用原 Key 的各 H 预警 GT：H>0 时 failure 的 [anchor−H,anchor) 为1、其余 pre-Key 为0、t≥anchor不参加 trigger 训练与主评估；H0 时 t≥anchor 为1、之前为0。success 全为0。风险回归使用与 trigger 相同的有效 mask。前两阶段只训练一次，五个 H 头独立训练，避免重复计算。\n',
        '## 与 UniIntervene 的区别\n',
        '本实验借鉴 [UniIntervene 三阶段逻辑](https://arxiv.org/html/2606.12372v1)，不是原方法复现：记录未提供 policy action，因此阶段2是 observation-conditioned future value，不能称为 action-conditioned Q；使用 frozen Deform 而非 VLM/V-JEPA2；本轮只使用 tactile，不编码 instruction，与既有 tactile baseline 一致。没有虚构失败负回报幅度，采用原文 TD 的失败 terminal reward0。为在线因果性，不使用整条 rollout 的 min-max value normalization。H 是本实验的 Key 预警 horizon，不是论文 recovery action horizon。\n',
        '## 自动训练停止\n','|Stage|Best epoch|Stopped epoch|Val loss|\n|---|---:|---:|---:|']
    for r in stopping:text.append(f"|{r['stage']}|{r['best_epoch']}|{r['epochs']}|{r['best_validation_loss']:.6f}|")
    text.extend(['\nproxy/future 使用 success/failure 两组 validation rollout 均值的平均，risk 使用自然 validation rollout loss 均值；均是固定 loss criterion。AdamW lr=.001、wd=.0001、clip1。\n','![Loss](loss_curves.png)\n',
        '## Test：固定阈值 0.5\n','BA=正负类 recall 的平均；Macro F1=两类 F1 平均。PR-AUC 使用 average precision；ROC-AUC 按 score ranking。frame metric 按全部有效 tick 汇总；event 只要 rollout 任一有效 tick 报警即检出，success 的任意检出均计 FP。\n',
        '|H|Frame BA|Macro F1|PR-AUC|ROC-AUC|Event precision|Event recall|Event FPR|Band detected|\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for r in results:
        f=r['test']['frame'];e=r['test']['event']
        text.append(f"|{r['horizon_frames']}|{f['balanced_accuracy']:.2%}|{f['macro_f1']:.2%}|{f['pr_auc']:.4f}|{f['roc_auc']:.4f}|{e['precision']:.2%}|{e['recall']:.2%}|{e['fpr']:.2%}|{e['positive_band_detection_rate']:.2%}|")
    text.extend(['\n## Validation-calibrated event threshold\n',
        '仅在 validation 搜索 event_score=max(valid risk score) 的阈值：event FPR≤15%，优先最大 recall，平手取更高 threshold；固定用于 test。val/test 分别只有3条 failure，因此 recall 分辨率33.3%；14条 success 的 FPR 分辨率7.14%。阈值高于最大 validation score 时允许完全不报警。校准后的 event recall 不能代替报警是否位于目标 band 的判断。\n',
        '|H|Threshold|Val event recall|Val FPR|Test event recall|Test FPR|Test band detected|First alarm in band|Test median lead(s)|\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for r in results:
        c=r['calibration'];e=r['test']['calibrated_event'];lead=e['median_first_alarm_lead_seconds']
        text.append(f"|{r['horizon_frames']}|{c['threshold']:.6f}|{c['validation_event_recall']:.2%}|{c['validation_event_fpr']:.2%}|{e['recall']:.2%}|{e['fpr']:.2%}|{e['positive_band_detection_rate']:.2%}|{e['first_alarm_in_positive_band_rate']:.2%}|{lead if lead is not None else '—'}|")
    text.extend(['\nlead=(anchor−first alarm frame)/30：正值为 Key 前、负值为 Key 后，仅统计已报警 failure。各 rollout 的具体 first_alarm_in_positive_band 见 runs/H*/test*_events.csv；band detected 指任何一次报警命中 band，与 first alarm 命中不同。\n',
        '## 连续 value 检查\n','|Split|Outcome|Mean V|Terminal V|Progress MSE|Downward fraction|\n|---|---|---:|---:|---:|---:|'])
    for r in value_quality:text.append(f"|{r['split']}|{r['outcome']}|{r['mean_value']:.4f}|{r['terminal_value']:.4f}|{r['progress_MSE']:.5f}|{r['downward_fraction']:.2%}|")
    text.extend(['\n## Key-relative curves\n','曲线分别按 success/failure rollout 对齐 Key；每条 rollout 每个 relative frame 先合并重复 camera frame，再等权计算均值。阴影为±1标准差，variance/std/参与 rollout 数见 key_relative_variance.csv。不同相对时间的有效 rollout 数不同；不外推不存在的时间段。failure H>0 只显示 pre-Key，与主评估一致。\n',
        '![Validation](val_key_curves.png)\n','![Test](test_key_curves.png)\n',
        '## 对照与限制\n','baseline_comparison.csv 对比历史 Deform weighted-BCE seed42，保持 encoder、split、H GT 相同，但旧模型用 max30/patience8/validation BA，本轮 max300/patience50/loss。因此不是只替换 loss 的消融，也不是五 seed 稳定性结论。本轮只有 seed42；不按 test 选 H 或阈值。原始标注与旧实验均未修改。\n',
        '可复查文件：protocol.json、source_dataset_manifest.json、source_interval_manifest.json、label_distribution.csv、proxy_quality.csv、stopping_audit.csv、summary.csv、runs/H*/result.json 与逐 rollout event CSV。\n'])
    text.insert(1,'本轮结论：H30/45 的 test PR-AUC 比历史直接 balanced-BCE seed42有所提升（0.136→0.338、0.152→0.361），但固定0.5下 BA 仅60.60%/60.47%；H8/15无failure检出。validation校准后H45检出1/3条failure、success误报0/14，首次报警提前0.733s，不能据此按test选H。proxy在train上mean V为success0.608/failure0.028，但val为0.473/0.428，显示第一阶段的泛化仍明显不足。单seed、val/test各3条failure，只能视为初步信号。\n')
    text.extend(['\n### 同 seed 旧 baseline 的 threshold-free 对照\n','|H|Old PR-AUC|New PR-AUC|Old ROC-AUC|New ROC-AUC|\n|---:|---:|---:|---:|---:|'])
    for r in results:
        old=next(b for b in baseline if b['seed']==42 and b['horizon_frames']==r['horizon_frames'])['test'];new=r['test']['frame']
        text.append(f"|{r['horizon_frames']}|{old['pr_auc']:.4f}|{new['pr_auc']:.4f}|{old['roc_auc']:.4f}|{new['roc_auc']:.4f}|")
    (out/'README.md').write_text('\n'.join(text))
    code=out/'code_snapshot';code.mkdir(exist_ok=True)
    for name in ('value_trend.py','value_trend_report.py','early_warning.py','common.py'):shutil.copy2(project_path('tools/sharpa_tactile', name),code/name)
    dump(out/'verification.json',dict(status='PASS',checkpoints=7,features=len(feature_files),causal_check=json.loads((out/'causal_check.json').read_text()),automatic_stopping=stopping,source_manifest_sha256=sha(out/'source_dataset_manifest.json')))
    weekly=project_path('WeeklySummary/10.5');dest=weekly/'value_trend'/out.name
    if dest.exists():shutil.rmtree(dest)
    # Share compact reviewable reports, metrics and curves; retain features/weights in original outputs.
    dest.mkdir(parents=True)
    for path in out.iterdir():
        if path.is_file() and path.suffix in ('.md','.csv','.png','.pdf','.json'):shutil.copy2(path,dest/path.name)
    summary=(out/'README.md').read_text()
    for name in ('loss_curves','val_key_curves','test_key_curves'):summary=summary.replace(f'({name}.png)',f'(value_trend/{out.name}/{name}.png)')
    summary+='\n原始完整实验目录：`'+str(relative_path(out))+'`。\n'
    (weekly/'10.5value_trend.md').write_text(summary)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
