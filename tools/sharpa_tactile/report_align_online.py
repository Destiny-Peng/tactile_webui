"""Audit and publish the corrected online-window probe experiment."""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, dump, sha
from .align_windows import AlignWindowProbe, metrics
from .normalize_align_online import normalized_confusions


def write_csv(path, rows):
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); output=args.output.resolve()
    manifest=json.loads((output/'run_manifest.json').read_text())
    results=json.loads((output/'results.json').read_text()); assert len(results)==60
    dataset=ROOT/manifest['dataset']; dm=json.loads((dataset/'dataset_manifest.json').read_text())
    assert all(sha(ROOT/path)==value for path,value in manifest['input_hashes'].items())
    torch.set_num_threads(4)
    audited=[]; per_seed=[]; per_class=[]
    for result in results:
        directory=output/'runs'/result['group']/f'seed_{result["seed"]}'/(result['input']+'_'+result['head'])
        history=json.loads((directory/'history.json').read_text())
        selected=max(history,key=lambda row:row['val_balanced_accuracy'])
        assert selected['epoch']==result['best_epoch']
        assert abs(selected['val_balanced_accuracy']-result['validation']['balanced_accuracy'])<1e-8
        assert result['class_weights']==[1.0,1.0,1.0]
        with np.load(directory/'test_predictions.npz') as stored, np.load(dataset/'ready'/result['group']/'test.npz') as ready:
            assert all(np.array_equal(stored[key],ready[key]) for key in ready.files)
            assert metrics(stored['labels'],stored['probabilities'])==result['test']
            assert np.array_equal(stored['predictions'],stored['probabilities'].argmax(1))
            if result['seed']==42:
                blob=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
                model=AlignWindowProbe(**blob['model_config']); model.load_state_dict(blob['state_dict']); model.eval()
                idx=stored['indices'][:4]; ri=int(stored['rollout_index'][0]); row=dm['rollouts'][ri]
                assert np.all(stored['rollout_index'][:4]==ri)
                with np.load(dataset/row['feature_path']) as bank:
                    inputs={key:torch.from_numpy(bank[key][idx-row['feature_offset']].copy()) if key in result['input'] else None for key in ('f6','deform')}
                with torch.inference_mode():
                    p=model(**inputs).softmax(-1).numpy()
                    np.testing.assert_allclose(p,stored['probabilities'][:4],atol=2e-5,rtol=2e-4)
                    if result['head']=='gru':
                        seq,_=model.sequence(**inputs)
                        changed={key:value.clone() if value is not None else None for key,value in inputs.items()}
                        for value in changed.values():
                            if value is not None: value[:,8:]+=100
                        modified,_=model.sequence(**changed)
                        torch.testing.assert_close(seq[:,:8],modified[:,:8],rtol=0,atol=0)
                        torch.testing.assert_close(seq[:,-1],model(**inputs))
                audited.append(str(directory.relative_to(output)))
        row={key:result[key] for key in ('group','input','head','seed','best_epoch','epochs')}
        row.update({key:result['test'][key] for key in ('accuracy','balanced_accuracy','macro_f1','failure_false_positive_rate')})
        per_seed.append(row)
        for name,values in result['test']['per_class'].items():
            per_class.append(dict(group=result['group'],input=result['input'],head=result['head'],seed=result['seed'],label=name,**values))
    write_csv(output/'per_seed.csv',per_seed); write_csv(output/'per_class.csv',per_class)
    summary=[]; paired=[]
    for group in manifest['groups']:
        for kind in ('f6','deform','f6_deform'):
            for head in ('mlp','gru'):
                selected=[r for r in per_seed if (r['group'],r['input'],r['head'])==(group,kind,head)]
                assert len(selected)==5
                row=dict(group=group,input=kind,head=head,seeds=5)
                for key in ('accuracy','balanced_accuracy','macro_f1','failure_false_positive_rate'):
                    values=[r[key] for r in selected]; row[key+'_mean']=float(np.mean(values)); row[key+'_std']=float(np.std(values,ddof=1))
                for label in ('in_progress','success','failure'):
                    for key in ('precision','recall','f1'):
                        values=[r[key] for r in per_class if (r['group'],r['input'],r['head'],r['label'])==(group,kind,head,label)]
                        row[label+'_'+key+'_mean']=float(np.mean(values))
                summary.append(row)
        for head in ('mlp','gru'):
            for single in ('f6','deform'):
                for key in ('balanced_accuracy','macro_f1'):
                    differences=[]
                    for seed in range(42,47):
                        a=next(r for r in per_seed if (r['group'],r['input'],r['head'],r['seed'])==(group,'f6_deform',head,seed))
                        b=next(r for r in per_seed if (r['group'],r['input'],r['head'],r['seed'])==(group,single,head,seed))
                        differences.append(a[key]-b[key])
                    paired.append(dict(group=group,head=head,single=single,metric=key,differences=differences,mean=float(np.mean(differences)),std=float(np.std(differences,ddof=1))))
    write_csv(output/'summary.csv',summary); dump(output/'paired_seed_differences.json',paired)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,group in zip(axes,manifest['groups']):
        selected=[r for r in summary if r['group']==group]
        ax.bar(range(6),[100*r['balanced_accuracy_mean'] for r in selected],yerr=[100*r['balanced_accuracy_std'] for r in selected],capsize=3)
        ax.set_xticks(range(6),[r['input'].replace('f6_deform','Fusion')+'\n'+r['head'].upper() for r in selected],fontsize=9)
        ax.set_ylim(0,100); ax.set_ylabel('Test BA (%)'); ax.set_title(group)
    fig.tight_layout(); fig.savefig(output/'comparison.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,6,figsize=(17,6))
    for i,group in enumerate(manifest['groups']):
        selected=[r for r in results if r['group']==group and r['seed']==42]
        for ax,result in zip(axes[i],selected):
            cm=np.asarray(result['test']['confusion_matrix']); ax.imshow(cm,cmap='Blues')
            for y in range(3):
                for x in range(3): ax.text(x,y,str(cm[y,x]),ha='center',va='center',fontsize=8)
            ax.set_xticks(range(3),['Prog','Succ','Fail']); ax.set_yticks(range(3),['Prog','Succ','Fail']); ax.set_xlabel('Prediction'); ax.set_ylabel('GT'); ax.set_title(result['input']+' '+result['head'],fontsize=9)
    fig.suptitle('Seed 42; top=symmetric, bottom=asymmetric+merged'); fig.tight_layout(); fig.savefig(output/'confusion_seed42.png',dpi=160); plt.close(fig)
    normalized_confusions(output,results)
    lines=['# 修正规则后的在线窗口三分类：训练与测试','',f'数据集：`{manifest["dataset"]}`；实验：`{output.relative_to(ROOT)}`。共 2 组 × 3 输入 × 2 模型 × 5 seeds（42–46）=60 次训练。','',
    '## GT 如何生成','',
    '每个样本输入 16 个时刻的冻结特征，所有输入限制在 rollout 最早任意标注 start 到最后任意标注 end 内。参考采样 n=5、step=3；训练 success/failure 使用已保存的 n/step 增强与去重样本。GT 直接读取 ready NPZ 中的 labels，训练时不重新移动 Key，也不使用 Gaussian。只扫描窗口后 8 个实际采样点，从 last frame 向前寻找最近的 Key：success=1、failure=2；没有 Key 命中为 in_progress=0。前 8 点不参与 GT 判定，但作为模型历史输入。末帧已离开 Key 时仍按后半窗口最近命中定类。','',
    '基础组（symmetric_separate）：Align 6 为 failure、Align 8 为 success，Key=[end−n,end+n]。新增组（asymmetric_merged_success）为一个联合对照：同 rollout 同时有 Align failure/success 时，按最早 start 到最晚 end 连间隙合并，outcome=success，只保留最终 success Key；Key=[end−n,对应 insert start+n]，无后续 Insert 则上限回退 end+n。Insert 不独立预测。','',
    '## 模型、loss 和输出','',
    '冻结 T-Rex F6 encoder（每时刻使用连续 16 tactile ticks，得到 1280D）和 Sharpa DeformEncoder（五指各经 2×2 平均池化得到 512D，拼为 2560D）。训练只读取预计算冻结特征；各输入先按训练集实际引用的唯一特征行计算 mean/std（std 下限 0.01），不使用 val/test 拟合。F6/Deform 分别 Linear→128D；Fusion concat→256D。','',
    'MLP：整个 16 点窗口作时间均值池化→投影→Linear(128或256,128)→ReLU→Dropout(0.1)→Linear(128,3)。GRU：逐时刻投影→单层单向 GRU(hidden=128)→最后 hidden→Linear(128,3)。GRU 使用完整窗口，每个窗口 hidden 从零初始化，不在窗口间携带状态。此次保持最新在线实验的 GRU 设置，区别于早期 interval 实验的 LSTM。','',
    '输出 softmax 得到 P(in_progress), P(success), P(failure)，argmax 映射为一个窗口类别。不会把窗口预测再拼成 interval，也不输出结束时间。训练用 logits 的 weighted cross-entropy，weight=(1/n_class)/mean(1/n_class)。两组 ready/train 三类均已平衡，因此权重都是 [1,1,1]，本轮等价于 unweighted CE。','',
    '训练：AdamW(lr=0.001, weight_decay=0.0001)，batch=128，梯度裁剪=1，最多 30 epochs，验证 BA 连续 8 epochs 无提高停止。以验证集 BA 最大的 epoch 选 checkpoint，平分取首次；随后运行测试。seed 仅影响初始化/dropout/shuffle；没有搜索新的 n/step 或模型结构。','',
    '## 数据分布与指标','',
    '| 组别 | Split | in_progress | success | failure | 总计 |','|---|---|---:|---:|---:|---:|']
    for group in manifest['groups']:
        for split in ('train','val','test'):
            with np.load(dataset/'ready'/group/(split+'.npz')) as data:
                c=np.bincount(data['labels'],minlength=3)
            lines.append(f'| {group} | {split} | {c[0]} | {c[1]} | {c[2]} | {c.sum()} |')
    lines += ['', 'rollout split 固定为 train/val/test=80/17/17；val/test 使用 n=5、step=3 的完整自然分布，不增强、不平衡。','',
    '测试单位为窗口，每个窗口计一次。ACC=正确窗口数/所有窗口数；每类 precision=TP/(TP+FP)，recall=TP/(TP+FN)，F1=2PR/(P+R)，零分母计 0；BA=三类 recall 的平均，macro F1=三类 F1 的平均。failure precision 仅针对 failure 类；failure FPR=非 failure GT 被预测为 failure 的数量/非 failure GT 数量。混淆矩阵行=GT、列=预测，顺序 in_progress/success/failure。测试指标不使用训练 class weights。先分别计算 5 个 seed，再取均值±样本标准差（ddof=1），不是 ensemble。','',
    '| 组别 | 输入 | 模型 | ACC % | BA % | Macro F1 % | failure precision % | failure recall % | failure FPR % |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        fmt=lambda key:f'{100*r[key+"_mean"]:.2f} ± {100*r[key+"_std"]:.2f}'
        lines.append(f'| {r["group"]} | {r["input"]} | {r["head"].upper()} | {fmt("accuracy")} | {fmt("balanced_accuracy")} | {fmt("macro_f1")} | {100*r["failure_precision_mean"]:.2f} | {100*r["failure_recall_mean"]:.2f} | {fmt("failure_false_positive_rate")} |')
    for group in manifest['groups']:
        candidates=[r for r in summary if r['group']==group]
        best_ba=max(candidates,key=lambda r:r['balanced_accuracy_mean'])
        best_f1=max(candidates,key=lambda r:r['macro_f1_mean'])
        lines += ['', f'{group}：BA 均值最高为 {best_ba["input"]} {best_ba["head"].upper()}（{100*best_ba["balanced_accuracy_mean"]:.2f}% ± {100*best_ba["balanced_accuracy_std"]:.2f}%）；macro F1 均值最高为 {best_f1["input"]} {best_f1["head"].upper()}（{100*best_f1["macro_f1_mean"]:.2f}%）。此处为测试结果描述，checkpoint 仍由验证 BA 选择。']
    lines += ['', '![五 seed BA](comparison.png)','', '![Seed 42 混淆矩阵](confusion_seed42.png)','', '按 GT 行归一化：每格除以该行真实样本数，每行总和为 100%，对角线为该类 recall。下图为 seed 42；全部 60 次归一化矩阵见 confusion_normalized.json。','', '![Seed 42 按 GT 行归一化](confusion_seed42_normalized.png)','',
    '## 如何解释对照','',
    '两组 GT 定义、训练数量、自然测试类比例不同，分数差异不能直接归因于模型提升。应先在同一组内比较 F6/Deform/Fusion；配对 seed 差异见 paired_seed_differences.json。五 seed 的标准差只反映初始化/训练随机性，没有覆盖新的 rollout split。滑窗高度重叠，窗口不是独立试验；新增组测试 failure 仅来自 3 个 failure-only rollout、60 个窗口，failure 指标仍需谨慎。旧实验 5 使用不同 Key/范围/采样规则，不能直接比较 BA。','',
    '## 产物和复现','',
    '`summary.csv`：12 组汇总；`per_seed.csv`：60 次指标；`per_class.csv`：每类 precision/recall/F1/support；`paired_seed_differences.json`：Fusion 对单模态配对 seed 差异。`runs/<group>/seed_<seed>/<input>_<head>/` 包含 best.pt、history.json、metrics.json、test_predictions.npz/CSV。CSV 每行包含 rollout、采样帧/ticks、n/step、GT、预测和三类概率；NPZ 另保留完整输入索引。','',
    '全部 60 次测试概率重新计算指标，逐项核对固定 ready GT 和输入；12 个 seed 42 checkpoints 在 CPU 重放，6 个 GRU 验证未来输入变化不会改变之前的 logits。数据集、encoder 权重与训练代码哈希在 run_manifest.json 中，训练前后保持不变。','',
    '```bash',f'CUBLAS_WORKSPACE_CONFIG=:4096:8 bash tools/run_sharpa_tactile_ablation.sh train_align_online --dataset {manifest["dataset"]} --output outputs/sharpa_align_online_training/<new_timestamp> --device cuda:0',f'bash tools/run_sharpa_tactile_ablation.sh report_align_online --output {output.relative_to(ROOT)}','```','',
    f'日志：`logs/sharpa_align_online_training_{output.name}.log`；环境：`repos/ProcVLM/.venv`；代码快照见 code_snapshot/。数据生成目录的 training_started=false 记录生成阶段，本轮训练另存此目录。']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    dump(output/'verification.json',dict(status='PASS',runs=60,metrics_recomputed=True,ready_arrays_unchanged=True,input_hashes_unchanged=True,cpu_checkpoint_replays=audited,causal_gru_checks=6,rollout_split_isolation=True,selection_checked=True))
    target=ROOT/'WeeklySummary/10.5/align_online_training'; target.mkdir(parents=True,exist_ok=False)
    names=['README.md','summary.csv','per_seed.csv','per_class.csv','paired_seed_differences.json','comparison.png','confusion_seed42.png','confusion_seed42_normalized.png','confusion_normalized.json','run_manifest.json','verification.json']
    copies=[]
    shutil.copy2(Path(__file__),output/'code_snapshot'/Path(__file__).name)
    for name in names:
        shutil.copy2(output/name,target/name); assert sha(output/name)==sha(target/name)
        copies.append(dict(source=str((output/name).relative_to(ROOT)),destination=str((target/name).relative_to(ROOT)),sha256=sha(output/name)))
    for result in results:
        relative=Path('runs')/result['group']/f'seed_{result["seed"]}'/(result['input']+'_'+result['head'])
        dest=target/relative; dest.mkdir(parents=True)
        for name in ('metrics.json','history.json','test_predictions.csv'):
            shutil.copy2(output/relative/name,dest/name)
            assert sha(output/relative/name)==sha(dest/name)
            copies.append(dict(source=str((output/relative/name).relative_to(ROOT)),destination=str((dest/name).relative_to(ROOT)),sha256=sha(dest/name)))
    dump(target/'copy_manifest.json',copies)
    (target/'SOURCE_INDEX.md').write_text(f'# 原始产物索引\n\n训练目录：`{output.relative_to(ROOT)}`\n\n数据目录：`{manifest["dataset"]}`\n\n权重与完整预测 NPZ 保留在训练目录 runs/。WeeklySummary 复制报告、指标、history 和 CSV，不复制权重与特征。\n')
    weekly=ROOT/'WeeklySummary/10.5/10.5.md'; text=weekly.read_text()
    text=text.replace('## 实验 6：修正规则后的在线三分类数据集（仅生成，未训练）','## 实验 6：修正规则后的在线三分类数据集')
    text=text.replace('**本次只生成数据和统计分布，不启动训练。**','生成阶段仅准备数据；后续训练结果见实验 7。')
    text=text.replace('- 实验 6：仅生成修正后的三分类数据集；没有训练指标，不能把实验 5 的分数当作其结果。','- 实验 6：修正后的三分类数据集；实验 7 使用这些数据训练，不能把实验 5 的分数当作其结果。')
    section=['## 实验 7：修正数据集的训练测试（60 次）','',
    '固定实验 6 的 ready 数据，2 组×F6/Deform/Fusion×MLP/单向 GRU×5 seeds（42–46）。GT 是窗口后半实际采样点倒推得到的三分类硬标签。模型读取完整 16 点窗口，MLP 时间均值池化，GRU 最后 hidden 分类；两个 encoder 冻结，输出三类 softmax→argmax。按 ready/train 类数量的 weighted CE 权重均为 [1,1,1]；val/test 不平衡，按验证 BA 选择 checkpoint。','',
    '| 组别 | 输入 | 模型 | Test ACC % | Test BA % | Test Macro F1 % |','|---|---|---|---:|---:|---:|']
    for r in summary:
        fmt=lambda key:f'{100*r[key+"_mean"]:.2f} ± {100*r[key+"_std"]:.2f}'
        section.append(f'| {r["group"]} | {r["input"]} | {r["head"].upper()} | {fmt("accuracy")} | {fmt("balanced_accuracy")} | {fmt("macro_f1")} |')
    section += ['', '各 seed 先计算窗口 ACC、三类平均 recall（BA）和三类平均 F1，再取均值±样本标准差。两组 GT 不同，不能直接将分数差异解释为性能提升；新增组只有 3 个 failure test rollout，60 个窗口，窗口重叠不能当作独立试验。','',
    '[完整 GT/loss/结构/指标说明](align_online_training/README.md) · [60 次指标](align_online_training/per_seed.csv) · [逐类指标](align_online_training/per_class.csv) · [配对 seed 差异](align_online_training/paired_seed_differences.json) · [校验](align_online_training/verification.json)','',
    '![修正数据集五 seed 结果](align_online_training/comparison.png)','']
    text=text.replace('## 结果应如何对照','\n'.join(section)+'\n## 结果应如何对照')
    weekly.write_text(text)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
