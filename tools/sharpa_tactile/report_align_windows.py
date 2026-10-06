"""Reports for Align Key-window screening and validation-selected five-seed comparisons."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import shutil
import numpy as np
from .common import ROOT, INPUTS, dump, sha
from .align_windows_data import CLASSES, STEPS
from .align_windows import metrics


def write_csv(path, records):
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def statistics(results):
    grouped={}
    for row in results: grouped.setdefault((row['config']['id'],row['name']),[]).append(row)
    records=[]
    for (config,name),runs in sorted(grouped.items()):
        record=dict(config=config,group=name,seeds=' '.join(str(r['seed']) for r in sorted(runs,key=lambda x:x['seed'])),n_seeds=len(runs))
        for m in ('accuracy','balanced_accuracy','macro_f1','failure_false_positive_rate'):
            values=[r['test'][m] for r in runs]
            record[m+'_mean']=float(np.mean(values));record[m+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
        for cls in CLASSES:
            for m in ('precision','recall','f1'):
                values=[r['test']['per_class'][cls][m] for r in runs]
                record[cls+'_'+m+'_mean']=float(np.mean(values));record[cls+'_'+m+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
        records.append(record)
    return records


def reports(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import t
    results=json.loads((output/'results.json').read_text()); manifest=json.loads((output/'align_manifest.json').read_text())
    grid=json.loads((output/'dataset_grid.json').read_text()); ranking=json.loads((output/'validation_ranking.json').read_text())
    status=json.loads((output/'suite_status.json').read_text()); verification=json.loads((output/'verification.json').read_text())
    assert verification['status']=='PASS' and len(results)==status['runs']
    summaries=statistics(results);write_csv(output/'summary.csv',summaries)
    per_seed=[];per_step=[];per_event=[]
    for row in results:
        c=row['config'];record=dict(config=c['id'],phase=c['phase'],steps=' '.join(map(str,c['steps'])),jitter=c['jitter'],
            membership=c['membership'],context=c['context'],padding=c['padding'],weight_mode=c['weight_mode'],
            group=row['name'],seed=row['seed'],best_epoch=row['best_epoch'],epochs_run=row['epochs_run'],
            val_balanced_accuracy=row['validation']['balanced_accuracy'],val_macro_f1=row['validation']['macro_f1'],
            test_windows=row['test']['n'],accuracy=row['test']['accuracy'],balanced_accuracy=row['test']['balanced_accuracy'],
            macro_f1=row['test']['macro_f1'],failure_false_positive_rate=row['test']['failure_false_positive_rate'])
        for cls in CLASSES:
            for m in ('precision','recall','f1','support'): record[cls+'_'+m]=row['test']['per_class'][cls][m]
        per_seed.append(record)
        directory=output/'runs'/c['id']/f"seed_{row['seed']}"/row['name']
        with (directory/'test_predictions.csv').open() as file: predictions=list(csv.DictReader(file))
        for step in STEPS:
            chosen=[r for r in predictions if int(r['step'])==step]
            if not chosen: continue
            met=metrics([int(r['label']) for r in chosen],[[float(r[cls+'_probability']) for cls in CLASSES] for r in chosen])
            per_step.append(dict(config=c['id'],group=row['name'],seed=row['seed'],step=step,n=met['n'],
                accuracy=met['accuracy'],balanced_accuracy=met['balanced_accuracy'],macro_f1=met['macro_f1'],
                failure_precision=met['per_class']['failure']['precision'],failure_recall=met['per_class']['failure']['recall'],
                failure_false_positive_rate=met['failure_false_positive_rate']))
        for event_id in sorted({r['event_id'] for r in predictions}):
            chosen=[r for r in predictions if r['event_id']==event_id]
            met=metrics([int(r['label']) for r in chosen],[[float(r[cls+'_probability']) for cls in CLASSES] for r in chosen])
            per_event.append(dict(config=c['id'],group=row['name'],seed=row['seed'],event_id=event_id,
                rollout_id=chosen[0]['rollout_id'],n=met['n'],accuracy=met['accuracy'],macro_f1=met['macro_f1'],
                positive_windows=sum(int(r['label'])!=0 for r in chosen)))
    write_csv(output/'per_seed.csv',per_seed);write_csv(output/'per_step.csv',per_step);write_csv(output/'per_event.csv',per_event)
    paired=[]
    for config in status['repeat_configs']:
        selected=[r for r in results if r['config']['id']==config]
        reference={r['seed']:r for r in selected if r['name']=='f6_deform_gru'}
        for baseline in ('f6_gru','deform_gru','f6_deform_mlp'):
            other={r['seed']:r for r in selected if r['name']==baseline}; seeds=sorted(reference.keys()&other.keys())
            diff=np.array([reference[s]['test']['balanced_accuracy']-other[s]['test']['balanced_accuracy'] for s in seeds])
            mean=float(diff.mean()); margin=float(t.ppf(.975,len(diff)-1)*diff.std(ddof=1)/np.sqrt(len(diff)))
            paired.append(dict(config=config,comparison='f6_deform_gru minus '+baseline,seeds=seeds,differences=diff.tolist(),
                mean_difference=mean,wins=int((diff>0).sum()),ci95=[mean-margin,mean+margin],
                interpretation='training-seed variation on one fixed held-out rollout split; not independent window/data uncertainty'))
    dump(output/'paired_seed_differences.json',paired)
    # An interpretation changes GT/support: compare its own bank and the common intersection separately.
    anchor='step3_n5_span_align_local_edge_inverse_frequency';paired_interpretations=[]
    for config in [g['config'] for g in grid if g['config']['phase']=='interpretation']:
        for kind in INPUTS:
            for head in ('mlp','gru'):
                group=kind+'_'+head
                def read(c):
                    with (output/'runs'/c/'seed_42'/group/'test_predictions.csv').open() as file:
                        return {(r['event_id'],r['step'],r['window_end_frame'],json.loads(r['sample_ticks'])[-1]):r for r in csv.DictReader(file)}
                base=read(anchor);alternative=read(config['id']);shared=sorted(base.keys()&alternative.keys())
                y=[int(base[k]['label']) for k in shared]
                left=metrics(y,[[float(base[k][cls+'_probability']) for cls in CLASSES] for k in shared])
                right=metrics(y,[[float(alternative[k][cls+'_probability']) for cls in CLASSES] for k in shared])
                paired_interpretations.append(dict(config=config['id'],group=group,common_endpoints=len(shared),
                    target='anchor span GT; matching event/step/camera endpoint/tactile endpoint tick, though context/inputs can differ',
                    anchor_ba=left['balanced_accuracy'],alternative_ba=right['balanced_accuracy'],
                    anchor_macro_f1=left['macro_f1'],alternative_macro_f1=right['macro_f1']))
    write_csv(output/'interpretation_common_endpoints.csv',paired_interpretations)
    best=ranking[0]['config']['id']; selected=[s for s in summaries if s['config']==best]
    ordered=[kind+'_'+head for kind in INPUTS for head in ('mlp','gru')]
    selected.sort(key=lambda r:ordered.index(r['group']))
    fig,ax=plt.subplots(figsize=(10,4));xs=np.arange(6)
    ax.bar(xs,[r['balanced_accuracy_mean']*100 for r in selected],yerr=[r['balanced_accuracy_std']*100 for r in selected],capsize=4)
    ax.set_xticks(xs,[r['group'] for r in selected],rotation=20);ax.set_ylabel('3-class balanced accuracy (%)');ax.set_ylim(0,100)
    ax.set_title('Align Key windows: validation-selected config, 5 training seeds');fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(output/('comparison.'+suffix),dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for ax,group in zip(axes.flat,ordered):
        row=next(r for r in results if r['config']['id']==best and r['seed']==42 and r['name']==group)
        cm=np.array(row['test']['confusion_matrix']);ax.imshow(cm,cmap='Blues');ax.set_title(group)
        ax.set_xticks(range(3),CLASSES,rotation=20);ax.set_yticks(range(3),CLASSES);ax.set_xlabel('Predicted');ax.set_ylabel('True')
        for i in range(3):
            for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(output/('confusion_matrices.'+suffix),dpi=150)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    for steps in ('1','3','5','8','12','mix'):
        points=[r for r in ranking if ('mix' if len(r['config']['steps'])>1 else str(r['config']['steps'][0]))==steps]
        points.sort(key=lambda r:r['config']['jitter'])
        ax.plot([r['config']['jitter'] for r in points],[r['validation_mean_balanced_accuracy']*100 for r in points],marker='o',label='step '+steps)
    ax.set_xlabel('Key jitter radius n (training frames)');ax.set_ylabel('Mean validation BA across 6 groups (%)');ax.legend(ncol=3)
    ax.set_title('Common validation bank; no test-based configuration selection');fig.tight_layout();fig.savefig(output/'validation_grid.png',dpi=150);plt.close(fig)
    chosen_cell=next(g for g in grid if g['config']['id']==best); counts=chosen_cell['splits']['test']['class_counts']
    baseline=counts[0]/sum(counts)
    lines=['# Align Key-window 三分类：MLP / Causal GRU','',
        '仅使用 Align event 6/8，排除 Insert event 7/9。每个固定 16 个采样点的滑窗输出一个类别，而非对完整 interval 输出一个成功/失败标签。','',
        '## Label → GT → loss → output','',
        '| 区域/条件 | GT |','|---|---:|','| 窗口不包含移动后的 Key | 0：in_progress |',
        '| 窗口包含 Key，Align event 8 | 1：success |','| 窗口包含 Key，Align event 6 | 2：failure |','',
        '`Key = 原 Align end_frame + δ`，原 `[causal, observable]` 仍只表示标注起止帧。训练中 δ 遍历 `[-n,+n]` 的全部整数，每个 event 每轮使用一个位移，同一 event 的不同 step 窗口共用该位移。每个 event 使用 seed 决定的循环排列；至少训练 `2n+1` 轮，保证每个整数位移都被用到。不同 Key 位移形成不同的硬 GT；没有 Gaussian、soft label 或 -1。验证与测试始终使用原 Key（δ=0）。','',
        '主要规则 `span`：实际窗口首帧 ≤ Key ≤ 实际末帧，闭区间包含；对照 `sampled`：Key 必须等于 16 个实际采样帧中的一个。稀疏采样可能跨过 Key 而未实际采中，两种规则都已运行。它们定义不同 GT，不能直接混为同一指标。','',
        'Loss 为三分类 cross-entropy。主要配置按**训练滑窗 × 全部 Key 位移版本**的三类样本数量计数，权重 `N/(3*n_class)`；每轮取循环中的一个硬标签版本训练。权重不按原始帧数或 interval 数计算。另在 step=3、n=5 跑 unweighted CE 与 `1/sqrt(n_class)` 后均值归一化的对照。','',
        '输出 `Linear(hidden,3) → softmax → [P(in_progress),P(success),P(failure)]`，用 argmax 得到该窗口的预测标签。Key、event outcome、起止帧、step 数值均不作为模型输入。','',
        '## 采样与两层时间窗口','',
        '下游输入固定 16 个采样点：在终点 e、帧间距 s 下，理想采样为 `[e-15s, …, e-s, e]`，终点滑动步长为 1 相机帧。例如 step=3 为 46 帧跨度，step=8 为 121 帧跨度。帧间 step 和终点滑动步长是两个参数。','',
        '每个 F6 特征仍由冻结 T-Rex encoder 的**稠密连续 16 tactile ticks**产生；改变下游 step 不改变 encoder 本身的预训练输入。于是 GRU 的 16 个输入是 16 个 F6 特征，而不是把 16 个间隔采样 raw F6 直接送入 VQ-VAE。Deform 每个采样时刻独立编码。','',
        'F6 `[16,1280] → Linear → [16,128]`；Deform `[16,2560] → Linear → [16,128]`（每指冻结 encoder → AvgPool(2×2) → 512D，五指拼接）。Fusion 在每个时刻 concat 为 `[16,256]`。两个 encoder 始终 frozen，decoder 不参与，也未重新做重建检查。','',
        'MLP：整个 16 点窗口做 temporal mean → Linear(128/256,128) → ReLU → Dropout(0.1) → Linear(128,3)。代码先对原始 feature 求均值再做 affine projection，与先投影后求均值等价。','',
        'GRU：单向、1 层、hidden=128，读取完整 16 点序列，最后 hidden → Linear(128,3)。每个窗口从空 hidden state 开始；不跨窗口继承状态。仅使用当前终点及之前的 feature。','',
        '## 数据边界与参数探索','',
        '主要网格：step=`1/3/5/8/12` 或五种 step 混合增强，n=`0/2/5/10`，共 24 配置；全部使用 span / align_local / edge。另在 step=3、n=5 对 span/sampled × align_local/past_context × edge/drop 的 8 种解释做完整组合对照（含主网格 anchor），加 2 个 loss 对照，总计 33 配置 × 6 组 × seed42。','',
        '`align_local` 从该 Align 起点取数据，起点不足 16 点时重复本片段首个有效帧补齐；F6 起始特征使用已验证的 interval-only prefix。`past_context` 可读取 Align 起点前最多 180 帧，但不进入之前的 Align interval。`drop` 完全不补齐，只保留足够长度的窗口；它改变样本覆盖，不能和 edge 原始数值直接比较。','',
        '为了让正方向 Key 位移有机会被观察到，采样终点可延伸到原 Key+10 帧，所有主配置范围一致，并在下一个 Align 起点之前截断。Insert 标注帧及 F6 稠密历史与 Insert 相交的特征都排除。同步缺帧、相机重复索引、tick 缺失或被排除区段会重新分段（保证非补齐采样的帧间距准确），不构造跨缺口的窗口；不进行未来填充。Key 不强制吸附到有效采样帧，也不静默截断位移。','',
        '验证/测试共同窗口 bank 都包含全部五种 step；主网格的输入和 GT 完全相同。训练改变 step/jitter，评估固定 bank，使主要配置之间可比较。另报告各 step 的单独指标。解释对照改变输入覆盖或 GT，另给共同 event/step/endpoint 上、以 anchor span GT 为目标的比较表。','',
        '仍复用原 rollout train/val/test split（80/17/17），不重划分。筛选后实际 Align interval 为 train 117（65 success/52 failure）、val 22（14/8）、test 24（15/9）。同一 rollout 的窗口和 Key 增强版本始终只属于一个 split。','',
        '## 标注来源与现目录核对','',
        '本轮沿用实验 1–4 的已提取 intervals.jsonl 快照，提取快照 SHA256、传感器特征、split 与 encoder 权重均已验证，未重新从当前 annotations 目录生成 GT。当前 annotations 目录与 10/3 的多 event 备份存在历史差异，且部分引用记录缺失；额外的现目录逐文件比较在三次失败后按 AGENTS.md 停止并记为 BLOCKED，见 [original_annotation_integrity.json](original_annotation_integrity.json)。这项额外检查不属于已完成的实验指标/模型验证；未恢复、改写或删除当前原标注或备份。','',
        '## 训练与选择','',
        '所有 33 配置先跑 seed42 的 F6/Deform/Fusion × MLP/GRU 六组。只按六组的平均**验证 BA**排名，在主网格中选前 3 配置，再各组补 seed43–46，形成 5 seeds。测试不参与 epoch、配置或阈值选择。最佳 checkpoint 只能在完成全部 Key 位移的一轮循环后选择（epoch ≥ 2n+1）。最多 30 轮，patience=8（满足 Key 位移覆盖后才能停止），batch=128，AdamW lr=1e-3、weight_decay=1e-4、gradient clip=1。仅用训练窗口实际引用的唯一 feature rows 计算 normalization，std floor=0.01。','',
        f'完成 {len(results)} 个小型冻结特征 probe；验证见 [verification.json](verification.json)，CPU/GPU checkpoint 重放见 [checkpoint_replay.json](checkpoint_replay.json)。','',
        '## 当前结果','',f'验证集选择第一名配置：`{best}`。测试共 {sum(counts)} 个窗口，GT in_progress/success/failure = {counts}。','',
        '| 模态 / Head | ACC % | BA % | Macro F1 % | Failure precision % | Failure recall % |',
        '|---|---:|---:|---:|---:|---:|']
    for row in selected:
        values=[f"{100*row[m+'_mean']:.2f} ± {100*row[m+'_std']:.2f}" for m in ('accuracy','balanced_accuracy','macro_f1','failure_precision','failure_recall')]
        lines.append('| '+row['group']+' | '+' | '.join(values)+' |')
    lines.extend(['',f'均值 ± 标准差来自 5 个训练 seeds（sample std，ddof=1），不是 ensemble。测试全部预测 in_progress 的 ACC 为 {baseline*100:.2f}%，BA 为 33.33%；高 ACC 本身不能证明结束状态识别好。','',
        'BA 是三个类别 recall 的算术平均，Macro F1 是三个类别 F1 的算术平均；ACC 是正确窗口数/全部窗口数。每个滑窗贡献一次 GT/预测，混淆矩阵顺序为 in_progress/success/failure，行是真实、列是预测。指标均不加 class weight。Failure precision 只对应 failure 类，等于真实 failure 且预测 failure 的窗口数 / 所有预测 failure 的窗口数。','',
        '窗口高度重叠，7505 个测试窗口不是 7505 次独立试验，原始 test 仅 24 个 Align interval；repeat seeds 只反映初始化/训练随机性，不提供跨数据划分的可靠性结论。此前 frame-wise 或 interval-binary 的 BA 是不同 GT/评估单位，不能与本任务直接比较。','',
        '有 1 个 Align 的原结束帧不在有效同步特征中（align_0083）；不对 Key 做吸附或未来填充。窗口是否包含 Key 仍按既定规则判断，逐 event 支持量见 per_event.csv，不能把缺少有效结束窗口的 event 当作可靠的终态识别验证。','',
        '当前实验使用已知 Align 起点来构造数据，并围绕标注终点限制采样范围；它是受控的窗口状态分类，不等于无需标注边界的完整在线事件检测。GRU 因果性保证输入不读取窗口终点之后的数据，不代表可提前准确预测未来成功/失败。','',
        '## 导航','',
        '- [summary.csv](summary.csv)：全部配置按 seed 汇总。','- [per_seed.csv](per_seed.csv)：每次实验的 GT 条件、loss、ACC/BA/F1 与三类指标。',
        '- [per_step.csv](per_step.csv)：固定评估 bank 中各采样 step 的结果。','- [per_event.csv](per_event.csv)：逐 Align interval 的窗口支持量与结果。',
        '- [dataset_grid.json](dataset_grid.json)：每个配置各 split 的窗口数、类别数与覆盖。','- [validation_ranking.json](validation_ranking.json)：验证集配置排名。',
        '- [paired_seed_differences.json](paired_seed_differences.json)：Fusion GRU 与其他组的配对 seed 差异。','- [interpretation_common_endpoints.csv](interpretation_common_endpoints.csv)：不同解释的共同 endpoint 对照。',
        '- `runs/<config>/seed_<seed>/<group>/`：checkpoint、history、validation/test_predictions、metrics、Key 位移容差评估。','',
        '![5-seed comparison](comparison.png)','![Validation grid](validation_grid.png)','![Seed42 confusion matrices](confusion_matrices.png)','',
        '## 复现命令','',
        '```bash','source ./project_env.sh',
        f"bash tools/run_sharpa_tactile_ablation.sh align_windows_data --source {manifest['signature']['source']} --interval-source {manifest['signature']['interval_source']} --output {output.relative_to(ROOT)}",
        f"bash tools/run_sharpa_tactile_ablation.sh verify_align_windows --output {output.relative_to(ROOT)}",
        f"bash tools/run_sharpa_tactile_ablation.sh align_windows --output {output.relative_to(ROOT)} --device cuda:1 --epochs 30 --patience 8 --batch-size 128 --repeat-top 3",
        f"bash tools/run_sharpa_tactile_ablation.sh verify_align_windows --output {output.relative_to(ROOT)} --completed",
        f"bash tools/run_sharpa_tactile_ablation.sh report_align_windows --output {output.relative_to(ROOT)}",'```',''])
    (output/'README.md').write_text('\n'.join(lines))
    findings=['# Findings: Align Key windows','',f'Primary validation winner: `{best}`.','',
        'All model and configuration selection uses original-Key validation labels. Primary test banks are identical.','',
        '| Config | Validation mean BA % (six groups, seed42) |','|---|---:|']
    for row in ranking[:10]: findings.append(f"| {row['config']['id']} | {row['validation_mean_balanced_accuracy']*100:.2f} |")
    best_group=max(selected,key=lambda r:r['balanced_accuracy_mean'])
    findings+=['','## Current observations','',
        f"At the validation-selected configuration, {best_group['group']} has the highest five-seed test BA mean among the six groups: {100*best_group['balanced_accuracy_mean']:.2f}% ± {100*best_group['balanced_accuracy_std']:.2f}%.",
        'The top three primary configurations chosen by validation all use n=0. Within this grid, Key jitter did not improve the six-group mean validation criterion; this does not prove that every individual model or all possible jitter ranges are worse.',
        f"Failure precision at the selected configuration ranges from {100*min(r['failure_precision_mean'] for r in selected):.2f}% to {100*max(r['failure_precision_mean'] for r in selected):.2f}%. Terminal-state false positives remain substantial; the representation does not yet separate the Key-defined states reliably.",
        'Replacing LSTM with GRU also changed the task from full-interval binary classification to Key-window three-class classification. These results cannot isolate a GRU-versus-LSTM architecture effect.',
        '', '### Loss ablation at step=3, n=5 (Fusion GRU, seed42)','',
        '| CE | BA % | Macro F1 % | Failure precision % | Failure recall % | Failure FPR % |','|---|---:|---:|---:|---:|---:|']
    for row in [r for r in results if r['name']=='f6_deform_gru' and r['seed']==42 and r['config']['id'].startswith('step3_n5_span_align_local_edge_')]:
        m=row['test'];f=m['per_class']['failure']
        findings.append(f"| {row['config']['weight_mode']} | {100*m['balanced_accuracy']:.2f} | {100*m['macro_f1']:.2f} | {100*f['precision']:.2f} | {100*f['recall']:.2f} | {100*m['failure_false_positive_rate']:.2f} |")
    findings+=['','## Paired seed comparison','', '| Config | Fusion GRU minus baseline | BA difference pp | Wins / 5 | 95% CI pp |','|---|---|---:|---:|---|']
    for row in paired: findings.append(f"| {row['config']} | {row['comparison']} | {100*row['mean_difference']:.2f} | {row['wins']} / 5 | [{100*row['ci95'][0]:.2f}, {100*row['ci95'][1]:.2f}] |")
    findings+=['','These confidence intervals describe training-seed variation only. Overlapping windows do not increase the number of independent held-out rollouts.',
        'Interpretation variants change label/support definitions: inspect their common-endpoint table before comparing headline metrics.','',
        'Key shift stress uses fixed predictions and changed hard annotation targets; it is annotation-tolerance analysis, not a new trained result.','']
    (output/'FINDINGS.md').write_text('\n'.join(findings))
    dump(output/'experiment_manifest.json',dict(status='complete',created_at=datetime.datetime.now().astimezone().isoformat(),
        experiment='align_key_window_3class_mlp_causal_gru',runs=len(results),source_signature=manifest['signature'],
        verification=verification,code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools/sharpa_tactile').glob('*.py')},
        configs=len(grid),repeat_configs=status['repeat_configs'],unit='fixed 16-sample window',class_mapping=list(CLASSES),
        supplementary_live_annotation_audit=json.loads((output/'original_annotation_integrity.json').read_text())))
    print('ALIGN_REPORT_COMPLETE',len(results),best,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();reports(a.output.resolve())


if __name__=='__main__':main()
