"""Summarize the completed six-group frozen tactile experiment."""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import numpy as np
from .common import ROOT,dump,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--superseded',type=Path)
    args=parser.parse_args();out=args.output
    data=json.loads((out/'data_manifest.json').read_text())
    results=json.loads((out/'results.json').read_text())
    split=json.loads((out/'split_manifest.json').read_text())
    online=json.loads((out/'online_verification.json').read_text())
    verify=json.loads((out/'verification.json').read_text())
    assert len(results)==6 and len({r['name'] for r in results})==6
    assert online['status']=='PASS' and all(v=='PASS' for k,v in verify.items() if k!='deform_512D_contract')
    assert sha(ROOT/'checkpoints/T-Rex/encoders/f6_tactile_vqvae.pt')==data['signature']['f6_sha256']
    assert sha(ROOT/'checkpoints/T-Rex/encoders/sharpa_wave_deform_encoder.pth')==data['signature']['deform_sha256']
    for r in results:
        assert (out/r['name']/'best.pt').is_file()
        assert r['test']['n']==2106
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(name,title) in zip(axes,[('balanced_accuracy','Balanced accuracy (threshold 0.5)'),('auroc','AUROC'),('macro_f1','Macro F1 (threshold 0.5)')]):
        for j,head in enumerate(('mlp','lstm')):
            chosen=[r for r in results if r['model_config']['head_kind']==head]
            values=[r['test_at_0_5'][name] for r in chosen]
            bars=ax.bar(np.arange(3)+(j-0.5)*0.32,values,width=0.3,
                        label='MLP' if head=='mlp' else 'Causal LSTM',color='#2563eb' if head=='mlp' else '#d97706')
            ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],fontsize=8,padding=3)
        ax.set_xticks(range(3),['F6','Deform','F6 + Deform']);ax.set_ylim(0,1);ax.set_title(title,fontsize=10)
        ax.grid(axis='y',alpha=0.2);ax.set_axisbelow(True)
        if name in ('balanced_accuracy','auroc'):ax.axhline(0.5,color='gray',linestyle='--',linewidth=0.8)
    axes[0].legend(fontsize=8)
    fig.suptitle('Frozen tactile binary probes | 17 held-out USB rollouts | seed 42')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
    rows=[]
    for r in results:
        rows.append(f"| {r['name']} | {r['best_epoch']} | {r['test_at_0_5']['balanced_accuracy']:.4f} | {r['test_at_0_5']['macro_f1']:.4f} | {r['test']['auroc']:.4f} | {r['test']['balanced_accuracy']:.4f} | {r['validation_selected_threshold']:.2f} |")
    counts=['| Split | Rollouts | Success ticks | Failure ticks |','|---|---:|---:|---:|']
    for name in ('train','val','test'):
        c=data['split_counts'][name];counts.append(f"| {name} | {c['rollouts']} | {c['success_rows']} | {c['failure_rows']} |")
    rel=out.relative_to(ROOT)
    readme=f'''# Sharpa tactile binary representation ablation

六组 first-stage frozen-encoder 对照已实现并完成训练。Binary 标签 success=0（8/9）、failure=1（6/7）。仅目标 interval 内帧有监督；其他标签和区间外帧不作为 success。原始标注未改写。

## Architecture

- F6：右手原始 `[16,5,6]`，使用官方冻结右手 min/max/mask 归一化 → T-Rex pretrained Temporal VQ-VAE encoder，使用量化前连续 `[5,256]` → flatten 1280D → trainable Linear 128D。
- Deform：五指原始灰度 scalar deformation maps（保留 0–255 浮点值）`[5,1,240,240]` → T-Rex checkpoint 内冻结 Sharpa DeformEncoder `[5,128,15,15]` → 按用户确认的 AdaptiveAvgPool2d(2,2) 得到每指 512D → flatten 2560D → trainable Linear 128D。
- F6+Deform：两路投影各 128D，直接 concat 256D；没有额外 fusion 网络。
- MLP：Linear(input,128) → ReLU → Dropout(0.1) → Linear(128,2)。
- Causal LSTM：1 层，hidden=128，单向，Linear(128,2)；支持 1–2 层和 hidden=128/256 的配置。episode 开始或无效同步缺口后重置状态。
- 编码器参数冻结且始终 eval；decoder 不用于 downstream。所有 encoder 运算为 FP32，TF32 关闭。训练监督帧拟合各模态的 feature mean/std，std 下限 0.01 防止近常量维度放大数值噪声。

本轮采用用户指定的池化方案；独立下载的 `SharpaWave-deform` 原生 512D ConvNeXt encoder/decoder **没有参与这六组实验**。

## Data and protocol

原始快照共 125 条 USB 标注轨迹，其中 114 条含目标 6–9 event，共 259 个 interval（failure=73，success=186）；其余 11 条不参与。cam_high 原始视频帧索引经 camera_frame_indices 对齐到记录 tick，不把标注帧直接当成同步行号。

按 closed `[causal,observable]` 边界分配 binary 标签。同类重叠可合并，异类重叠排除；本快照无异类重叠。重复 camera frame 对应的不同采集 tick 保留，因此 tick 计数可能与原始视频帧计数略有不同。

F6 每个输出仅使用当前及之前 15 个 tick；事件 receive_mono_ns 必须不晚于对应 tick_mono_ns。五指在 16 个 tick 内均有效、不 stale，且当前 deform 可读，才进入共同样本集。LSTM 保留过去无标签 context，但这些 tick 的 loss mask 为 ignore。缺失数据不作未来插值；序列在缺口处拆分。

按 task 和 binary class availability 分层，整个 rollout 分配到一个 split，所有六组共用 split 和样本。单 seed=42；30 epoch 上限、patience=8；AdamW lr=0.001、weight_decay=0.0001、batch=8 rollout segments、gradient clip=1。Class-weighted CE 的权重仅由 train 标签计数拟合；不在 val/test 重采样。

{chr(10).join(counts)}

最佳 epoch 仅由 validation balanced accuracy（阈值 0.5）选择。额外阈值在 validation 上校准，之后固定用于 test；同时保留固定 0.5 结果供公平比较。测试集评估 2106 个监督 tick、39 个 interval，不使用 test 标签调参。

## Held-out test results

| Group | Best epoch | BA @0.5 | Macro F1 @0.5 | AUROC | BA @val threshold | Val threshold |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

[对照图](comparison.png) · [PDF](comparison.pdf) · [CSV](comparison.csv) · [全部指标](results.json)

各组目录包含 best.pt、history.json、metrics.json、curves.png、test_predictions.csv、test_interval_predictions.json；metrics.json 另包含按 USB task 的独立指标。阈值和 mean/std 已保存进 checkpoint。

## Sanity and causal verification

F6 已在真实 train rollout 的一次接触窗口上经 pretrained VQ-VAE 重建；见 sanity.json 和 reconstruction_f6.png。Deform 重建按用户后续指示省略。

verification.json 验证六组 future perturbation/prefix invariance、逐步状态与整段前向等价、split 无重叠、encoder 冻结。online_verification.json 使用真实训练 prefix、已训练 F6+Deform LSTM，对比 CPU 原始输入在线推理与 GPU-cache 离线输出。最终在线最大概率差 {online['max_abs_difference']:.8f}，验证通过；不是延迟 benchmark。

## Run

从 LF3R 根目录：

```bash
bash tools/run_sharpa_tactile_ablation.sh prepare --output {rel} --device cuda:1 --batch-size 8
bash tools/run_sharpa_tactile_ablation.sh verify --output {rel}
bash tools/run_sharpa_tactile_ablation.sh train --output {rel} --device cuda:1 --epochs 30 --patience 8 --hidden 128 --layers 1 --batch-size 8
bash tools/run_sharpa_tactile_ablation.sh verify_online --output {rel}
```

相同设置可复用 feature cache；改变 encoder、数据、同步配置需新建输出目录。train 会写组内结果，复现实验建议使用独立输出目录。

## Online API

```python
from sharpa_tactile.models import OnlinePredictor
predictor = OnlinePredictor("{rel}/f6_deform_lstm/best.pt", device="cuda:1")
# 每个同步 tick 一次，顺序 thumb/index/middle/ring/pinky。
# f6: [5,6]，deform: [5,1,240,240]，valid 为五指有效且同步未过期。
p = predictor.step(f6, deform, tick=tick, valid=valid)
if p is not None:
    prediction = "failure" if p >= predictor.failure_threshold else "success"
# 每条新轨迹：
predictor.reset()
```

此 API 可在 `PYTHONPATH=tools` 并使用 ProcVLM 环境调用；F6-only/Deform-only 读取对应 best.pt 即可。为六组保持同样 16-tick 预热。OnlinePredictor 检查 encoder SHA256，避免部署时误用其他权重。

## Scope

这是一个固定 split、单 seed 的小数据初步对照；不能据此声称某种表征有稳定优势。评估范围是已标注 interval 内 binary 状态，尚未衡量无标签背景的误报警。历史上下文来自同一条轨迹；其他轨迹不共享 LSTM state。
'''
    if args.superseded:
        readme+=f'\n旧诊断输出 `{args.superseded.relative_to(ROOT)}` 因批量/逐帧浮点误差放大而被本次运行替代，不用于最终对照。split、seed、模型大小和训练预算保持一致；修改只针对数值一致性，未根据 test 指标调参。\n'
        (args.superseded/'SUPERSEDED.md').write_text(f'此运行被 {rel} 替代。真实在线一致性检查发现近常量特征放大 CPU/GPU、batch-size 数值误差；最终运行关闭 TF32 并设置 std 下限 0.01。请使用新目录结果。\n')
    (out/'README.md').write_text(readme)
    stamp=datetime.datetime.now().astimezone().isoformat()
    provenance={'created_at':stamp,'status':'complete','output':str(rel),
      'trex_commit':subprocess.check_output(['git','-C',str(ROOT/'repos/T-Rex'),'rev-parse','HEAD'],text=True).strip(),
      'project_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
      'environment':'repos/ProcVLM/.venv','encoder_signature':data['signature'],
      'source_file_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tools/sharpa_tactile').glob('*.py'))}}
    dump(out/'experiment_manifest.json',provenance)
    print(json.dumps({'status':'complete','output':str(rel),'groups':6,'online_max_difference':online['max_abs_difference']},indent=2))


if __name__=='__main__':main()
