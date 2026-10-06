"""Copy verified Align experiment documents into the existing 10.5 weekly summary."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import shutil
import subprocess
from .common import ROOT, dump, sha
from .align_windows_data import CLASSES


def publish(output):
    verification=json.loads((output/'verification.json').read_text()); assert verification['status']=='PASS'
    experiment=json.loads((output/'experiment_manifest.json').read_text()); assert experiment['status']=='complete'
    best=json.loads((output/'validation_ranking.json').read_text())[0]['config']['id']
    source_files=[p for p in output.iterdir() if p.is_file() and p.suffix in ('.md','.csv','.json','.png','.pdf')]
    # Checkpoints/features stay in outputs; retain all per-run metrics/history, and selected seed42 prediction examples.
    for p in (output/'runs').rglob('metrics.json'):source_files.append(p)
    for p in (output/'runs').rglob('history.json'):source_files.append(p)
    for p in (output/'runs'/best/'seed_42').rglob('test_predictions.csv'):source_files.append(p)
    weekly=ROOT/'WeeklySummary/10.5';destination=weekly/'align_key_windows';destination.mkdir(exist_ok=True)
    copied=[]
    for source in sorted(source_files):
        target=destination/source.relative_to(output);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        copied.append(dict(source=str(source.relative_to(ROOT)),destination=str(target.relative_to(weekly)),
                           source_sha256=sha(source),destination_sha256=sha(target),byte_identical=True))
        assert sha(source)==sha(target)
    rows=list(csv.DictReader((output/'summary.csv').open()));chosen=[r for r in rows if r['config']==best]
    order=['f6_mlp','f6_gru','deform_mlp','deform_gru','f6_deform_mlp','f6_deform_gru'];chosen.sort(key=lambda r:order.index(r['group']))
    section=['## 实验 5：Align-only Key 滑窗三分类，MLP / Causal GRU','',
        '只保留 Align event `6/8`，排除 Insert event `7/9`。将实验 4 的完整 interval 二分类，改为固定长度滑窗的状态三分类。','',
        '| 项目 | 实际做法 |','|---|---|',
        '| Label → GT | `Key=Align end_frame+δ`。窗口不含 Key → `in_progress=0`；含 Key 且原 event 8 → `success=1`；含 Key 且原 event 6 → `failure=2`。每个窗口一个硬 GT，无 Gaussian/soft label/-1。主要包含规则是 Key 落在实际首末采样帧的闭区间内；另测试 Key 必须被实际采中的规则。 |',
        '| Window / augmentation | 固定 16 个采样点，理想帧索引 `[e-15s,…,e-s,e]`；滑动终点每次前进 1 帧。训练 step=`1/3/5/8/12` 或混合；Key 位移半径 n=`0/2/5/10`。训练遍历所有整数 δ∈[-n,+n]；验证/测试 Key 不移动。 |',
        '| 边界 | 主配置只读取 Align 本地序列，开头不足 16 点时重复首个有效帧；终点可至原 Key+10，遇下一 Align/Insert 或无效同步段截断。对照包含过去上下文 / 只保留完整窗口。Insert 标注帧及含 Insert 历史的 F6 特征不进入新数据集。 |',
        '| Loss | 三分类 weighted CE，按训练**滑窗 × Key 位移版本**的类别数计算 `N/(3*n_c)`，不是帧数/interval 数。另在 step=3、n=5 测 unweighted 与 normalized sqrt weights。 |',
        '| Frozen encoder | 沿用同一 F6 / Deform encoder、五指 2×2 pooling 及各模态 Linear→128D；Fusion 按 timestep concat→256D。F6 encoder 每个特征仍读稠密 16 ticks；下游窗口读 16 个间隔采样的特征。 |',
        '| MLP / GRU | MLP：窗口 temporal mean→Linear(128/256,128)→ReLU→Dropout(.1)→Linear(128,3)。GRU：1 层单向、hidden=128，读取 16 个特征，最后 hidden→Linear(128,3)；每个窗口重置状态。 |',
        '| 最终输出 | 一个窗口的 `[P(in_progress),P(success),P(failure)]`，argmax 映射为该窗口的标签。in_progress 由是否含 Key 决定，不是此前 frame-wise 的 background。 |',
        '| Split / 选择 | 原 rollout 80/17/17 不变，Align interval train/val/test=117/22/24。33 配置×六组先跑 seed42；按主网格六组平均 validation BA 选前 3 配置补齐 seeds42–46。checkpoint 只能在完整 Key 位移循环后选择，测试不参与选择。 |',
        '| 测试口径 | 主要配置使用完全相同的五种 step 的测试 window bank，共 7,505 个窗口，GT 为 6,940 in_progress / 295 success / 270 failure。每窗口一次计数；报告各 step、各 class 与 3×3 confusion matrix。 |','',
        f'本轮共完成 **{experiment["runs"]} 次训练结果**，原始数据/encoder 哈希、split、硬标签、GRU 因果性、全部预测 CSV 的指标重算及六组 checkpoint 重放均通过。','',
        f'按验证集选择的第一名配置：`{best}`。以下为该配置五个 seeds 的均值 ± 样本标准差（ddof=1）。','',
        '| Input / Head | ACC | BA | Macro F1 | Failure precision | Failure recall |','|---|---:|---:|---:|---:|---:|']
    for row in chosen:
        values=[f"{100*float(row[m+'_mean']):.2f}% ± {100*float(row[m+'_std']):.2f}%" for m in ('accuracy','balanced_accuracy','macro_f1','failure_precision','failure_recall')]
        section.append('| '+row['group']+' | '+' | '.join(values)+' |')
    predictions=list(csv.DictReader((output/'runs'/best/'seed_42'/'f6_deform_gru'/'test_predictions.csv').open()))
    example=next(r for r in predictions if r['label']=='2' and r['step']=='3' and r['padded_frames']=='0')
    probs=[float(example[c+'_probability']) for c in CLASSES]
    section+=['',f"真实例子：`{example['event_id']}` 是 event 6，原 Key={example['key']}；step=3 的 16 点窗口实际范围为 [{example['window_start_frame']},{example['window_end_frame']}]，包含 Key，所以 GT=2（failure）。seed42 Fusion GRU 输出概率 [{probs[0]:.4f},{probs[1]:.4f},{probs[2]:.4f}]，argmax 预测={example['prediction']}。这个预测属于该滑窗，不是整个原 interval。",'',
        '标注来源与前四阶段一致，沿用已提取 interval 快照。当前 annotations 目录与旧备份有历史差异及缺失路径，额外核对按仓库规则停止；没有恢复或改写原文件，见 [来源核对记录](align_key_windows/original_annotation_integrity.json)。','',
        '所有窗口预测 in_progress 的基线 ACC=92.47%、BA=33.33%，因此 ACC 需要配合终态 recall/precision 解读。7,505 个窗口高度重叠，独立测试数据仍只有 24 个 Align interval、17 个 rollout；五 seeds 反映训练随机性。不同 GT/覆盖的解释对照单独记录，不能直接横向比较原始 BA。','',
        '1 个测试 Align 的原 Key 无有效同步帧，不做吸附/未来填充；详见原报告。数据构造使用已知 Align 边界，当前验证是窗口状态分类，尚不是无需标注边界的全 rollout 自动事件检测。','',
        '![Align Key-window five-seed comparison](align_key_windows/comparison.png)','',
        '[详细原理与实验报告](align_key_windows/README.md) · [每次实验的配置和指标](align_key_windows/per_seed.csv) · [按 step 的指标](align_key_windows/per_step.csv) · [配对 seed 差异](align_key_windows/paired_seed_differences.json) · [复制校验](align_key_windows/copy_manifest.json)','']
    main=weekly/'10.5.md';text=main.read_text()
    text=text.replace('所有阶段均有六组：F6 / Deform / F6+Deform × MLP / 单向 LSTM；两个 pretrained encoder 始终 frozen。',
        '所有阶段均有六组：F6 / Deform / F6+Deform × 两种 head；实验 1–4 使用 MLP / 单向 LSTM，实验 5 使用 MLP / 单向 GRU。两个 pretrained encoder 始终 frozen。')
    # Allow report refresh without duplicate sections.
    start=text.find('## 实验 5：')
    if start>=0:
        end=text.index('## 结果应如何对照',start);text=text[:start]+text[end:]
    text=text.replace('## 结果应如何对照','\n'.join(section)+'\n## 结果应如何对照',1)
    if '- 实验 5：评估单位' not in text:
        text=text.replace('- 实验 4：评估单位是 **整个 interval**，两类；利用完整已知边界内的序列后分类。',
            '- 实验 4：评估单位是 **整个 interval**，两类；利用完整已知边界内的序列后分类。\n- 实验 5：评估单位是 **固定长度 Align 滑窗**，三类；根据窗口是否包含结束 Key 定义 in_progress / success / failure。')
    if '[实验 5：Align Key-window' not in text:
        text=text.replace('- [实验 4：interval-level binary](sharpa_interval_binary/README.md)',
            '- [实验 4：interval-level binary](sharpa_interval_binary/README.md)\n- [实验 5：Align Key-window 3-class](align_key_windows/README.md)')
    text=text.replace('### 当前 interval 实验结果','### Interval binary 实验结果')
    text=text.replace('[当前完整报告](sharpa_interval_binary/README.md)', '[实验 4 完整报告](sharpa_interval_binary/README.md)')
    text=text.replace('[当前测试指标汇总（含 ACC）]', '[实验 4 测试指标汇总（含 ACC）]')
    text=text.replace('[当前原始产物、代码、权重与日志索引](sharpa_interval_binary/SOURCE_INDEX.md)', '[实验 4 原始产物、代码、权重与日志索引](sharpa_interval_binary/SOURCE_INDEX.md)')
    text=text.replace('当前实验的 `sharpa_interval_binary/seed_*/<group>/`', '实验 4 的 `sharpa_interval_binary/seed_*/<group>/`')
    if '[实验 5 原始产物与代码索引]' not in text:
        text=text.replace('- [实验 5：Align Key-window 3-class](align_key_windows/README.md)', '- [实验 5：Align Key-window 3-class](align_key_windows/README.md)\n- [实验 5 原始产物与代码索引](align_key_windows/SOURCE_INDEX.md)')
    main.write_text(text)
    (destination/'SOURCE_INDEX.md').write_text(f'# 原始产物索引\n\n- [原始实验 README](../../../{output.relative_to(ROOT)}/README.md)\n- [原始运行目录](../../../{output.relative_to(ROOT)}/runs)\n- [滑窗准备代码](../../../tools/sharpa_tactile/align_windows_data.py)\n- [MLP/GRU 训练代码](../../../tools/sharpa_tactile/align_windows.py)\n- [验证代码](../../../tools/sharpa_tactile/verify_align_windows.py)\n\n本目录复制报告、图表、逐配置指标、全部 run 的 metrics/history 和第一名配置 seed42 的六组 test prediction。特征与 checkpoint 保留在原 outputs 下。\n')
    dump(destination/'copy_manifest.json',dict(created_at=datetime.datetime.now().astimezone().isoformat(),
        source=str(output.relative_to(ROOT)),files=copied,copied_file_count=len(copied),
        generated_files=['SOURCE_INDEX.md','copy_manifest.json','../10.5.md']))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    trex_commit=subprocess.check_output(['git','-C','repos/T-Rex','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    report=ROOT/'environment_reports/SHARPA_ALIGN_WINDOWS_20261004_232000.md'
    report.write_text(f'# Sharpa Align Key-window experiment\n\nCompleted: {datetime.datetime.now().astimezone().isoformat()}\n\n- Environment: repos/ProcVLM/.venv, Python 3.10, torch 2.10+cu128; no new packages installed.\n- GPU: cuda:1; all probe jobs sequential; pretrained encoder weights frozen and reused cached features.\n- LF3R commit: {commit} (working tree changes captured by experiment_manifest.json code SHA256).\n- T-Rex commit: {trex_commit}.\n- Runs: {experiment["runs"]}; 33 config screens plus 3 validation-selected five-seed comparisons.\n- [Original report](../{output.relative_to(ROOT)}/README.md)\n- [Verification](../{output.relative_to(ROOT)}/verification.json)\n- [Training log](../logs/sharpa_align_train_verified_keycycle_20261004_232000.log)\n- [Weekly summary](../WeeklySummary/10.5/10.5.md)\n\nCommands are preserved in the original README. Original annotation files, features and encoders unchanged; split hash audited.\n')
    marker='Sharpa Align Key-window 3-class (20261004_232000)'
    for filename in ('SETUP_STATUS.md','SYSTEM_INFO.txt'):
        path=ROOT/filename;content=path.read_text()
        if marker not in content:
            with path.open('a') as file:file.write(f'\n\n{marker}: COMPLETE, {experiment["runs"]} sequential frozen-feature MLP/GRU probes; 33 configs, top3 5 seeds; verification PASS; report {report.relative_to(ROOT)}; outputs {output.relative_to(ROOT)}.\n')
    print('ALIGN_PUBLISH_COMPLETE',len(copied),'files',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();publish(a.output.resolve())


if __name__=='__main__':main()
