"""Report one binary prediction per interval across six probes and repeated seeds."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import subprocess
import numpy as np
from .common import ROOT, INPUTS, HEADS, dump, sha

MEASURES=('balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_f1','failure_false_positive_rate','auroc')


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); out=args.output.resolve()
    manifest=json.loads((out/'interval_manifest.json').read_text()); config=json.loads((out/'training_config.json').read_text())
    results=json.loads((out/'results.json').read_text()); verification=json.loads((out/'verification.json').read_text())
    assert all(v=='PASS' for v in verification.values())
    assert len(results)==len(config['seeds'])*6
    assert len({(r['seed'],r['name']) for r in results})==len(results)
    if manifest['signature']['f6_context']=='interval_only':
        raw_check=json.loads((out/'raw_interval_prefix_check.json').read_text())
        assert raw_check['status']=='PASS'
    source=ROOT/manifest['signature']['source']
    assert sha(out/'split_manifest.json')==sha(source/'split_manifest.json')==manifest['signature']['split_sha256']
    for kind,filename in [('f6','f6_tactile_vqvae.pt'),('deform','sharpa_wave_deform_encoder.pth')]:
        assert sha(ROOT/'checkpoints/T-Rex/encoders'/filename)==manifest['signature'][kind+'_sha256']
    rows=[]
    for r in results:
        cm=np.asarray(r['test']['confusion_matrix_success_failure'])
        assert r['training_interval_counts']==[129,55] and cm.sum(axis=1).tolist()==[29,10]
        assert r['test']['n']==39 and (out/f"seed_{r['seed']}"/r['name']/'best.pt').exists()
        rows.append({'seed':r['seed'],'group':r['name'],'best_epoch':r['best_epoch'],
                     **{key:r['test'][key] for key in MEASURES}})
    with (out/'per_seed.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summaries=[]
    for kind in INPUTS:
        for head in HEADS:
            name=kind+'_'+head; chosen=[r for r in rows if r['group']==name]
            row={'group':name,'n_seeds':len(chosen)}
            for key in MEASURES:
                values=np.array([r[key] for r in chosen]); row[key+'_mean']=float(values.mean())
                row[key+'_std']=float(values.std(ddof=1)) if len(values)>1 else None
            summaries.append(row)
    dump(out/'summary.json',summaries)
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,('balanced_accuracy','macro_f1','failure_precision'),('Interval balanced accuracy','Interval macro F1','Failure interval precision')):
        for j,head in enumerate(HEADS):
            selected=[next(r for r in summaries if r['group']==kind+'_'+head) for kind in INPUTS]
            values=[r[key+'_mean'] for r in selected]; std=[r[key+'_std'] or 0 for r in selected]
            bars=ax.bar(np.arange(3)+(j-.5)*.32,values,yerr=std,width=.3,capsize=3,label='Mean-pool MLP' if head=='mlp' else 'Causal LSTM')
            ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=3,fontsize=8)
        ax.set_xticks(range(3),['F6','Deform','F6 + Deform']); ax.set_ylim(0,1); ax.set_title(title,fontsize=10)
        ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    axes[0].legend(fontsize=8); fig.suptitle(f"39 held-out annotated intervals | {len(config['seeds'])} training seeds | mean ± sample SD")
    fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=180); fig.savefig(out/'comparison.pdf'); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(10,7))
    for ax,r in zip(axes.flat,[r for r in results if r['seed']==config['seeds'][0]]):
        cm=np.array(r['test']['confusion_matrix_success_failure']); ax.imshow(cm,cmap='Blues')
        ax.set_title(r['name']); ax.set_xticks([0,1],['success','failure']); ax.set_yticks([0,1],['success','failure'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.suptitle(f"Interval confusion matrices | seed {config['seeds'][0]} | 29 success + 10 failure")
    fig.tight_layout(); fig.savefig(out/'confusion_matrices_seed42.png',dpi=160); plt.close(fig)
    table=['| Group | BA | Macro F1 | Failure precision | Failure recall | Failure FPR |','|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        def fmt(key):
            mean=100*r[key+'_mean']; std=r[key+'_std']
            return f'{mean:.2f} ± {100*std:.2f}' if std is not None else f'{mean:.2f}'
        table.append('| '+r['group']+' | '+' | '.join(fmt(key) for key in ('balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_false_positive_rate'))+' |')
    counts=['| Split | Rollouts | Success intervals | Failure intervals | Total intervals |','|---|---:|---:|---:|---:|']
    split=json.loads((out/'split_manifest.json').read_text())
    for name in ('train','val','test'):
        c=manifest['split_counts'][name]; counts.append(f"| {name} | {len(split[name])} | {c['success']} | {c['failure']} | {c['intervals']} |")
    rel=out.relative_to(ROOT)
    context='严格仅用 interval 内数据；首个有效 tick 重复左侧补齐至 16。原缓存 F6 起点后的不足 16-tick 前缀重新编码，其余窗口已全部在 interval 内。' if manifest['signature']['f6_context']=='interval_only' else 'F6 使用当前及之前 15 个 rollout ticks，允许区间开始前的历史输入；这些历史输入没有单独标签或 loss。'
    readme=f'''# Interval-level tactile binary classification

六组模型、{len(config['seeds'])} seeds 已完成。训练单位是原始标注 event 的整个 interval，输出一个 success/failure；没有 frame-wise 标签、background 类或逐帧 loss，也不对旧 frame-wise 概率做后处理平均。

## Samples and split

Event 6/7 → failure=1；8/9 → success=0。`[causal,observable]` 两字段仅表示 closed interval start/end，不作因果起点/可观察时间解释。各原始 event 保留为独立样本，不按类别合并相邻 interval。

{chr(10).join(counts)}

完整保留 259 个标注 interval，无空特征样本。原 split 文件逐字复用；SHA256 `{manifest['signature']['split_sha256']}`，同一 rollout 的所有 interval 仅存在于一个 split。通过 cam_high camera_frame_indices 对齐到记录 ticks，保留边界内有效同步特征；重复 camera frame 的不同采集 tick 仍保留。传感器同步有效过滤沿用已有缓存；interval_manifest.json 记录每个 event 的 tick 数和缺口数。

## Representation and aggregation

- F6：sliding 16-tick raw `[16,5,6]` → frozen T-Rex pretrained VQ-VAE continuous encoder feature `[5,256]` → flatten 1280D。{context}
- Deform：每 timestep 五指 `[5,1,240,240]` → frozen pretrained Sharpa/T-Rex DeformEncoder → 每指 AdaptiveAvgPool2d(2,2) 得到 512D → flatten 2560D，复用已有冻结特征。
- 各模态 Linear → 128D；Fusion 在 timestep 级 concat 为 256D。
- MLP：对 interval 的全部有效 timestep 投影特征做 masked temporal mean pooling，再 Linear(input,128) → ReLU → Dropout(.1) → Linear(128,2)。不是逐 tick MLP 后取概率均值。
- LSTM：完整 interval feature sequence → 1 层 hidden=128、单向 LSTM → 最后有效 tick 的 hidden → Linear(128,2)。每个 interval 的 state 从零开始，padding 通过 packed sequence 排除。
- softmax 输出 `[P(success),P(failure)]`。需给定 interval 边界，结束时给出分类；这不是未知边界的实时 failure onset detector。

## Training

Class weights 按 train **interval 数量**计算 `N_intervals/(2*n_class_intervals)`：success=`184/(2*129)`≈0.713178，failure=`184/(2*55)`≈1.672727。每个 interval 对应一个交叉熵项；长度只影响输入序列，不改变 label 数或 class weight。没有 background 样本和背景 loss。

Pretrained encoder 始终 frozen/eval。特征归一化仅由 train interval 内特征拟合，std floor=.01；无 background 进入统计。其余设置沿用：seeds {config['seeds']}，固定原 rollout split；AdamW lr=.001/weight_decay=.0001，batch=8 intervals，max 30 epochs/patience=8，gradient clip=1，FP32/TF32 关闭。最佳 epoch 按 validation interval balanced accuracy 选择；test 不参与调参，failure threshold 固定 .5。

## Held-out interval results

单位 %；均值±样本标准差（ddof=1）。每个 seed 在同样 39 个 test interval 上评估。

{chr(10).join(table)}

[CSV](comparison.csv) · [逐 seed CSV](per_seed.csv) · [对照图](comparison.png) · [PDF](comparison.pdf) · [Seed 42 混淆矩阵](confusion_matrices_seed42.png) · [全部结果](results.json)

各 seed/group 包含 best.pt、history.json、metrics.json、test_predictions.csv。CSV 每行仅一个 interval，包含 event、原始 start/end、tick 数、单个 probability 和 prediction。metrics.json 提供两类 precision/recall/F1/support、2×2 confusion matrix、BA、macro F1、AUROC/AUPRC、failure FPR；这些计数均为 interval，不是 frame。

## Verification

verification.json 检查一标注一 scalar label、所有输入 timestep 位于边界内、无 rollout 泄漏、原 split 一致、权重来自 129/55 interval 数、MLP 置换不变性、不同长度 padding 不影响输出、LSTM 最后有效状态与逐步状态等价、独立 interval state 和二分类指标计数。Encoder SHA256 未变。raw_interval_prefix_check.json 另核对真实 interval 首 tick 重复补齐的 CPU 编码与 GPU 缓存一致。F6 prefix 编码是新窗口处理，pretrained 重建检查沿用已完成记录，不重复重建。

## Commands

```bash
bash tools/run_sharpa_tactile_ablation.sh prepare_intervals --source {manifest['signature']['source']} --output {rel} --device cuda:1 --f6-context {manifest['signature']['f6_context']}
bash tools/run_sharpa_tactile_ablation.sh verify_intervals --output {rel}
bash tools/run_sharpa_tactile_ablation.sh train_intervals --output {rel} --device cuda:1 --seeds 42 43 44 45 46
bash tools/run_sharpa_tactile_ablation.sh report_intervals --output {rel}
```

沿用 repos/ProcVLM/.venv，没有新依赖。运行日志在 logs/sharpa_interval_prepare_{out.name}.log、logs/sharpa_interval_verify_{out.name}.log、logs/sharpa_interval_train_{out.name}.log。重训使用新输出目录以保留现有结果。

## Feature-sequence inference

```python
import torch
from sharpa_tactile.interval_models import IntervalProbe
checkpoint = torch.load("{rel}/seed_42/f6_deform_lstm/best.pt", map_location="cpu", weights_only=True)
model = IntervalProbe(**checkpoint["model_config"]).eval()
model.load_state_dict(checkpoint["state_dict"], strict=True)
with torch.inference_mode():
    probability = model(f6_features, deform_features, lengths).softmax(-1)[:, 1]
# f6_features [B,T,1280], deform_features [B,T,2560], lengths [B]
# 编码输入遵循训练时 interval-only / past-rollout 配置。
```

旧 frame-wise binary / 3-class 结果和原始标注保留。新任务利用完整已知 interval，不能把指标直接与 frame-wise 或在线 onset detection 结果比较。只有一个固定 split；test 仅 39 个 interval、其中 10 个 failure，五 seeds 描述训练随机性，不能替代跨 split / 跨任务泛化验证。
'''
    (out/'README.md').write_text(readme)
    dump(out/'experiment_manifest.json',{'status':'complete','created_at':datetime.datetime.now().astimezone().isoformat(),
        'output':str(rel),'unit':'interval','groups':len(results),'environment':'repos/ProcVLM/.venv',
        'project_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'trex_commit':subprocess.check_output(['git','-C',str(ROOT/'repos/T-Rex'),'rev-parse','HEAD'],text=True).strip(),
        'signature':manifest['signature'],'source_file_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tools/sharpa_tactile').glob('*.py'))}})
    print('INTERVAL_REPORT_COMPLETE',len(results),flush=True)


if __name__=='__main__': main()
