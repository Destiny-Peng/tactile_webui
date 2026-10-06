"""Publish the completed three-class experiment and its audit trail."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import numpy as np
from .common import ROOT, dump, sha
from .train_three import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); out = args.output.resolve()
    data = json.loads((out/'data_manifest.json').read_text()); results = json.loads((out/'results.json').read_text())
    verification = json.loads((out/'verification.json').read_text()); online = json.loads((out/'online_verification.json').read_text())
    assert len(results) == 6 and all(value == 'PASS' for value in verification.values()) and online['status'] == 'PASS'
    previous = ROOT/data['signature']['previous_output']
    assert sha(out/'split_manifest.json') == sha(previous/'split_manifest.json')
    assert sha(ROOT/'checkpoints/T-Rex/encoders/f6_tactile_vqvae.pt') == data['signature']['f6_sha256']
    assert sha(ROOT/'checkpoints/T-Rex/encoders/sharpa_wave_deform_encoder.pth') == data['signature']['deform_sha256']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,('balanced_accuracy','macro_f1'),('3-class balanced accuracy / macro recall','3-class macro F1')):
        for j,head in enumerate(('mlp','lstm')):
            chosen = [r for r in results if r['model_config']['head_kind'] == head]
            values = [r['test'][key] for r in chosen]
            bars = ax.bar(np.arange(3)+(j-0.5)*0.32,values,width=0.30,label='MLP' if head == 'mlp' else 'Causal LSTM')
            ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=3,fontsize=8)
        ax.set_xticks(range(3),['F6','Deform','F6 + Deform']); ax.set_ylim(0,1); ax.set_title(title,fontsize=10)
        ax.grid(axis='y',alpha=0.2); ax.set_axisbelow(True)
        if key == 'balanced_accuracy': ax.axhline(1/3,color='gray',linestyle='--',linewidth=0.8)
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=180); fig.savefig(out/'comparison.pdf'); plt.close(fig)
    fig,axes = plt.subplots(2,3,figsize=(13,8))
    for ax,r in zip(axes.flat,results):
        cm = np.array(r['test']['confusion_matrix']); ax.imshow(cm,cmap='Blues')
        ax.set_title(r['name']); ax.set_xticks(range(3),CLASSES,rotation=20); ax.set_yticks(range(3),CLASSES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        for i in range(3):
            for j in range(3): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.tight_layout(); fig.savefig(out/'confusion_matrices.png',dpi=180); fig.savefig(out/'confusion_matrices.pdf'); plt.close(fig)
    rows = ['| Group | Best epoch | Balanced accuracy / macro recall | Macro F1 |','|---|---:|---:|---:|']
    per_class = ['| Group | Class | Precision | Recall | F1 | Support |','|---|---|---:|---:|---:|---:|']
    matrices = []
    for r in results:
        assert (out/r['name']/'best.pt').exists() and r['model_config']['num_classes'] == 3
        expected_counts = [data['split_counts']['test'][key] for key in ('background_rows','success_rows','failure_rows')]
        assert np.asarray(r['test']['confusion_matrix']).sum(axis=1).tolist() == expected_counts
        assert r['test']['n'] == sum(expected_counts)
        rows.append(f"| {r['name']} | {r['best_epoch']} | {r['test']['balanced_accuracy']:.4f} | {r['test']['macro_f1']:.4f} |")
        for name in CLASSES:
            m = r['test']['per_class'][name]
            per_class.append(f"| {r['name']} | {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")
        matrices.append(f"### {r['name']}\n\n```text\n"+'\n'.join(str(row) for row in r['test']['confusion_matrix'])+'\n```\n')
    counts = ['| Split | Rollouts | Background | Success | Failure | Total usable ticks |','|---|---:|---:|---:|---:|---:|']
    for name in ('train','val','test'):
        c = data['split_counts'][name]
        counts.append(f"| {name} | {c['rollouts']} | {c['background_rows']} | {c['success_rows']} | {c['failure_rows']} | {c['usable_feature_rows']} |")
    rel = out.relative_to(ROOT)
    readme = f'''# Sharpa tactile 3-class frame-wise classification

六组 frozen-encoder 三分类训练完成，替代本轮训练目标；旧 binary 实验和原始标注保留。

## Labels

- Background = 0：所有目标 interval 之外的帧。
- Success = 1：event 8/9 的 closed `[causal,observable]` 区间内所有帧。
- Failure = 2：event 6/7 的 closed `[causal,observable]` 区间内所有帧。
- 无 Gaussian、soft label、label smoothing 或 -1 ignore label。每个原始视频帧都生成确定类别，完整标签见 frame_labels/*.npy。同类重叠保持原类别；异类重叠报错，不默默排除。本数据无异类冲突。
- 仅事件 6–9 定义 success/failure；其他标签不定义新的类别，未落入目标 interval 的帧仍为 background。

## Data and fixed protocol

逐字复用 `{data['signature']['previous_output']}/split_manifest.json`；SHA256 `{data['signature']['split_sha256']}`。114 条原有 USB 轨迹，train/val/test 为 80/17/17；没有引入额外 11 条轨迹或重新随机划分。

保留完整轨迹的所有有效同步 tick，包括旧 binary 缓存未包含的最后 interval 之后的 background。通过 cam_high camera_frame_indices 映射原视频标注帧；重复 camera frame 的不同采集 tick 仍分别评估。frame_labels 涵盖每个原视频帧；模型有效样本仍遵循原来的五指同步有效、未 stale、receive 时间不晚于 tick 的过滤条件和 F6 的 16-tick 因果预热。缺失传感器数据和预热帧不具备有效输入，因此不进入六组共同样本集；其标注类别依然为确定 0/1/2。没有以 ignore 标签排除有效 background。

{chr(10).join(counts)}

F6 `[16,5,6]` → frozen T-Rex temporal VQ-VAE continuous pre-quantization `[5,256]` → flatten 1280D → Linear 128D。Deform `[5,1,240,240]`（原始 0–255 scalar maps）→ frozen T-Rex embedded Sharpa DeformEncoder `[5,128,15,15]` → AdaptiveAvgPool2d(2,2) → each finger 512D → flatten 2560D → Linear 128D。联合输入简单 concat 为 256D。两路 encoder 始终 frozen/eval，无 decoder 参与训练。

MLP：Linear(input,128) → ReLU → Dropout(0.1) → Linear(128,3)。Causal LSTM：1 层 hidden=128、单向 → Linear(128,3)。在线 softmax 顺序为 `[P(background),P(success),P(failure)]`；分类使用 argmax，不校准 binary failure threshold。

其余设置保持：seed 42、AdamW lr=0.001/weight_decay=0.0001、最多 30 epochs、patience 8、batch size 8 个序列段、gradient clip 1、FP32/TF32 关闭、train-only 特征标准化，std floor 0.01。归一化现在按全部有效训练帧（包含 background）拟合。

3-class weighted cross-entropy，train-only 权重为 `w_c = N_train / (3 * n_c)`；计数按当前三类有效训练帧计算；本轮 background/success/failure 权重为 `{results[0]['training_class_weights']}`。batch padding 仅按真实序列长度筛除后计算 CE，不使用 -1 标签或 CrossEntropyLoss ignore_index 来标记数据帧。最佳 epoch 按 validation 3-class macro recall 选择，测试集不参与调参。

## Test results

{chr(10).join(rows)}

[CSV](comparison.csv) · [对照图](comparison.png) · [PDF](comparison.pdf) · [完整指标](results.json)

## Per-class test metrics

{chr(10).join(per_class)}

## 3×3 confusion matrices

行 = true class，列 = predicted class；均按 background、success、failure 顺序。

[六组混淆矩阵图](confusion_matrices.png) · [PDF](confusion_matrices.pdf)

{chr(10).join(matrices)}

## Verification and artifacts

所有标签/完整轨迹、原 split 逐字一致、padding loss、已知 confusion matrix 的指标计算、六组 causal prefix/future perturbation/逐步状态等价、encoder 冻结以及旧 binary checkpoint 兼容检查通过。真实训练 prefix 的 CPU 原始输入在线输出与 GPU-cache 推理三类概率最大差 `{online['max_abs_difference']:.8g}`，PASS。详见 verification.json 和 online_verification.json。

沿用已完成的 F6 重建检查，不重新执行；Deform 重建按用户要求省略。feature cache 复用旧冻结 encoder 的相同 tick 特征，仅新增未缓存的轨迹末段。

每组包含 best.pt（含 class mapping、train class counts/weights、normalization）、history.json、metrics.json、curves.png 和 test_predictions.csv（每帧 label/prediction/三类概率）；metrics.json 也报告按 USB task 分组结果。

## Reproduce

从 LF3R 根目录执行：

```bash
bash tools/run_sharpa_tactile_ablation.sh prepare_three --output {rel} --previous {data['signature']['previous_output']} --device cuda:1 --batch-size 8
bash tools/run_sharpa_tactile_ablation.sh verify_three --output {rel}
bash tools/run_sharpa_tactile_ablation.sh train_three --output {rel} --device cuda:1 --epochs 30 --patience 8 --hidden 128 --layers 1 --batch-size 8
bash tools/run_sharpa_tactile_ablation.sh verify_online --output {rel}
bash tools/run_sharpa_tactile_ablation.sh report_three --output {rel}
```

重训请使用独立输出目录以保留现有结果。环境沿用 repos/ProcVLM/.venv，没有添加依赖；运行日志见 logs/sharpa_three_prepare_{out.name}.log 及 logs/sharpa_three_train_{out.name}.log。

## Online API

```python
from sharpa_tactile.models import OnlinePredictor
predictor = OnlinePredictor("{rel}/f6_deform_lstm/best.pt", device="cuda:1")
probs = predictor.step(f6, deform, tick=tick, valid=valid)
if probs is not None:
    label = int(probs.argmax())  # background=0, success=1, failure=2
predictor.reset()  # 每条新轨迹重置
```

保持每 tick 一次调用和 thumb/index/middle/ring/pinky 的五指顺序，前 15 个有效 tick 预热返回 None；无效输入或数据缺口重置状态。旧 binary checkpoint 仍返回 failure 概率 scalar。结果为单 seed、单 split 初步对照，不能据此声称表征具有稳定优势。三分类含大量背景帧，数值不能直接与旧二分类 interval-only 指标作性能增减比较。
'''
    (out/'README.md').write_text(readme)
    dump(out/'experiment_manifest.json',{'created_at':datetime.datetime.now().astimezone().isoformat(),'status':'complete',
        'output':str(rel),'num_classes':3,'environment':'repos/ProcVLM/.venv','encoder_signature':data['signature'],
        'trex_commit':subprocess.check_output(['git','-C',str(ROOT/'repos/T-Rex'),'rev-parse','HEAD'],text=True).strip(),
        'project_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'source_file_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tools/sharpa_tactile').glob('*.py'))}})
    print('THREE_CLASS_REPORT_COMPLETE',str(rel),flush=True)


if __name__ == '__main__': main()
