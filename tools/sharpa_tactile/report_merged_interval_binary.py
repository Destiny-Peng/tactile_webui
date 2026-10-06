"""Audit binary merged-interval probes and publish both equivalent group aliases."""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,dump,sha
from .merged_interval_binary import MergedIntervalProbe,read_samples,load
from .train_intervals import metrics,batch_data,normalization


def write_csv(path,records):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();output=a.output.resolve()
    run=json.loads((output/'run_manifest.json').read_text());dataset=ROOT/run['dataset'];dm=json.loads((dataset/'interval_manifest.json').read_text())
    results=json.loads((output/'results.json').read_text());assert len(results)==30
    assert all(sha(ROOT/path)==value for path,value in run['input_hashes'].items())
    assert all(sha(ROOT/path)==value for path,value in dm['input_hashes'].items())
    equivalent=json.loads((dataset/'group_equivalence.json').read_text());assert equivalent['identical'] and len(set(equivalent['interval_corpus_sha256'].values()))==1
    split=json.loads((dataset/'split_manifest.json').read_text());assert sha(dataset/'split_manifest.json')==dm['split_sha256']
    assert not(set(split['train'])&set(split['val']) or set(split['train'])&set(split['test']) or set(split['val'])&set(split['test']))
    for row in dm['records']:
        data=load(dataset/row['feature_path']);assert data['frames'].min()>=row['start_frame'] and data['frames'].max()<=row['end_frame']
        assert (np.diff(data['ticks'],axis=1)>=0).all() and len(data['f6'])==row['windows']
        assert row['rollout_id'] in split[row['split']]
    samples=read_samples(dataset,dm);norm=normalization(samples['train']);torch.set_num_threads(4)
    per_seed=[];per_class=[];normalized=[];replays=0
    for result in results:
        directory=output/'runs'/f'seed_{result["seed"]}'/(result['input']+'_'+result['head'])
        predictions=load(directory/'test_predictions.npz');y=np.array([r['label'] for r in samples['test']])
        assert np.array_equal(predictions['labels'],y) and np.allclose(predictions['probabilities'].sum(1),1,atol=1e-5)
        assert metrics(y,predictions['probabilities'][:,1])==result['test']
        assert np.array_equal(predictions['predictions'],(predictions['probabilities'][:,1]>=.5).astype(int))
        history=json.loads((directory/'history.json').read_text());assert max(history,key=lambda r:r['val_balanced_accuracy'])['epoch']==result['best_epoch']
        assert result['train_counts']==[64,16] and result['class_weights']==[.625,2.5]
        blob=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
        for k,value in norm.items():torch.testing.assert_close(blob['state_dict'][k],value,rtol=0,atol=0)
        if result['seed']==42:
            model=MergedIntervalProbe(**blob['model_config']);model.load_state_dict(blob['state_dict']);model.eval()
            batch=batch_data(samples['test'][:2],'cpu')
            with torch.inference_mode():prob=model(batch['f6'],batch['deform'],batch['lengths']).softmax(-1).numpy()
            np.testing.assert_allclose(prob,predictions['probabilities'][:2],atol=3e-5,rtol=3e-4);replays+=1
        record={k:result[k] for k in ('input','head','seed','best_epoch','epochs')}
        record.update({k:result['test'][k] for k in ('accuracy','balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_false_positive_rate','auroc','failure_auprc')});per_seed.append(record)
        for name,values in result['test']['per_class'].items():per_class.append(dict(input=result['input'],head=result['head'],seed=result['seed'],label=name,**values))
        cm=np.asarray(result['test']['confusion_matrix_success_failure'],dtype=float);rows=cm.sum(1,keepdims=True);ncm=np.divide(cm,rows,out=np.zeros_like(cm),where=rows!=0)
        np.testing.assert_allclose(ncm.sum(1),1)
        for i,name in enumerate(('success','failure')):assert np.isclose(ncm[i,i],result['test']['per_class'][name]['recall'])
        normalized.append(dict(input=result['input'],head=result['head'],seed=result['seed'],class_order=['success','failure'],support=rows[:,0].astype(int).tolist(),normalized_by_gt_row=ncm.tolist()))
    write_csv(output/'per_seed.csv',per_seed);write_csv(output/'per_class.csv',per_class);dump(output/'confusion_normalized.json',normalized)
    summary=[];paired=[]
    for kind in ('f6','deform','f6_deform'):
        for head in ('mlp','gru'):
            records=[r for r in per_seed if (r['input'],r['head'])==(kind,head)];assert len(records)==5
            rec=dict(input=kind,head=head,seeds=5)
            for k in ('accuracy','balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_false_positive_rate','auroc','failure_auprc'):
                values=[r[k] for r in records];rec[k+'_mean']=float(np.mean(values));rec[k+'_std']=float(np.std(values,ddof=1))
            summary.append(rec)
    for head in ('mlp','gru'):
        for single in ('f6','deform'):
            differences=[]
            for seed in range(42,47):
                a=next(r for r in per_seed if (r['input'],r['head'],r['seed'])==('f6_deform',head,seed))
                b=next(r for r in per_seed if (r['input'],r['head'],r['seed'])==(single,head,seed));differences.append(a['balanced_accuracy']-b['balanced_accuracy'])
            paired.append(dict(head=head,comparison='Fusion minus '+single,seeds=list(range(42,47)),BA_differences=differences,mean=float(np.mean(differences)),std=float(np.std(differences,ddof=1))))
    write_csv(output/'summary.csv',summary);dump(output/'paired_seed_differences.json',paired)
    y=[r['label'] for r in samples['test']];baselines={'always_success':metrics(y,np.zeros(len(y))),'always_failure':metrics(y,np.ones(len(y)))};dump(output/'constant_baselines.json',baselines)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4));names=[r['input'].replace('f6_deform','Fusion')+'\n'+r['head'].upper() for r in summary]
    for ax,key in zip(axes,('balanced_accuracy','macro_f1')):
        ax.bar(range(6),[100*r[key+'_mean'] for r in summary],yerr=[100*r[key+'_std'] for r in summary],capsize=3)
        ax.axhline(100*baselines['always_success'][key],color='gray',linestyle='--',label='Always success')
        ax.set_xticks(range(6),names);ax.set_ylim(0,100);ax.set_ylabel(key+' (%)');ax.legend()
    fig.suptitle('Merged interval binary | same corpus for both Key groups | 5 seeds');fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,6,figsize=(18,6))
    chosen=[r for r in results if r['seed']==42]
    for j,result in enumerate(chosen):
        cm=np.asarray(result['test']['confusion_matrix_success_failure']);ncm=cm/cm.sum(1,keepdims=True)
        for i,values in enumerate((cm,ncm)):
            ax=axes[i,j];image=ax.imshow(values,cmap='Blues',vmin=0,vmax=1 if i else None)
            for yy in range(2):
                for xx in range(2):ax.text(xx,yy,f'{values[yy,xx]*100:.1f}%' if i else str(values[yy,xx]),ha='center',va='center',color='white' if i and values[yy,xx]>.55 else 'black')
            ax.set_xticks(range(2),['Success','Failure']);ax.set_yticks(range(2),['Success\nN=14','Failure\nN=3']);ax.set_xlabel('Prediction');ax.set_ylabel('GT');ax.set_title(result['input']+' '+result['head'].upper())
    fig.suptitle('Seed 42 | top: counts; bottom: GT-row normalized');fig.tight_layout();fig.savefig(output/'confusion_seed42.png',dpi=160);plt.close(fig)
    dump(output/'verification.json',dict(status='PASS',unique_runs=30,source_groups_equivalent=True,source_hashes_unchanged=True,interval_boundaries_checked=True,rollout_split_isolation=True,metrics_recomputed=True,checkpoint_cpu_replays=replays,training_normalization_checked_all_checkpoints=True,weights_by_interval_count=True,selection_checked=True,normalized_confusions_checked=True))
    lines=['# 合并 Align interval 的 binary 可分性实验','',
    f'数据：`{dataset.relative_to(ROOT)}`；训练：`{output.relative_to(ROOT)}`。F6 / Deform / Fusion × MLP temporal pooling / 单向 GRU × seeds42–46，共30次独立训练。','',
    '## 为什么两组是同一个实验','',
    '当前 symmetric_merged 和 asymmetric_merged 共享完全相同的合并 Align interval，仅 Key 带不同。interval binary 不使用 Key，也不使用三分类窗口GT。因此两个组的interval ID、边界、标签和输入一致，group_equivalence.json记录相同的interval corpus SHA256；同一组结果同时适用于两种Key设置，不重复训练60次或制造两份独立证据。','',
    '## GT、输入、Loss 和输出','',
    '一条rollout对应一个合并Align interval，包含多次尝试之间的间隙。event6对应最终failure，label=1；event8对应最终success，label=0。114intervals=92success+22failure，11个无目标标注rollout继续排除。没有in_progress/background、Gaussian或窗口级GT；每个interval一个硬标签、一个loss项、一个最终预测。','',
    '输入仅限该interval的[start,end]。末帧逐个有效相机帧推进，step=3：每个局部F6窗口为[t−45,t−42,…,t]的16个原始采样点，起始不足长度仅重复interval内首有效点，真实tactile缺口分段补齐；采样帧缺失则丢弃该窗口，不用未来数据填补。每个窗口一次冻结F6编码，官方finger-mode checkpoint→[5,256]→1280D；Deform只取该窗口末帧→五指2×2pool→2560D。整个interval形成[T,1280]和[T,2560]，T因interval而异。此次固定step3，不作n/step增强，n不参与binary标签。','',
    '每个时刻两路分别Linear→128D，Fusion逐时刻concat→256D。MLP：对整个interval的有效窗口特征作时间均值池化→Linear(128或256,128)→ReLU→Dropout(.1)→Linear(128,2)。GRU：完整按时间排序的窗口特征序列→单层单向GRU(hidden128)→最后hidden→Linear(128,2)；每个interval状态从零开始，不跨rollout。变长序列用pack处理，MLP均值排除batch padding。','',
    'softmax输出P(success),P(failure)，P(failure)≥0.5预测failure，否则success；不作Key命中或frame→interval投票。本实验利用已知完整interval边界，只诊断表征对最终结果的可分性，不代表在线检测完成时点的能力。','',
    '训练用weighted CE，按训练interval数量计算weight=N/(2×n_class)：success64、failure16，因此权重[0.625,2.5]。不按序列长度或窗口数加权、增强或平衡采样；每个interval每epoch一次。两个encoder冻结，训练投影、MLP/GRU及二分类head。标准化只在训练interval的特征上拟合，std下限.01。','',
    '保持原rollout split=80/17/17；train为64success/16failure，val和test均14success/3failure。AdamW(lr=.001,wd=.0001)、batch8、clip1、最多30epochs、patience8；按val interval BA选最高epoch，平分取首次，再做test。5seeds只改变初始化/dropout/shuffle，没有调整split或测试阈值。','',
    '## 指标及结果','',
    '每个interval计一次，长度不影响测试权重。ACC=正确interval数/总数；BA=(success recall+failure recall)/2；macroF1=(success F1+failure F1)/2。per-class precision=TP/(TP+FP)、recall=TP/(TP+FN)、F1=2PR/(P+R)，零分母计0。failure FPR=success被判failure比例；AUROC和failure AUPRC基于P(failure)。先分别算各seed，再取均值±样本标准差(ddof1)，不是ensemble。','',
    f'测试全部预测success：ACC={100*baselines["always_success"]["accuracy"]:.2f}%，BA=50.00%，macroF1={100*baselines["always_success"]["macro_f1"]:.2f}%。因此不能仅凭高ACC判断可分性。','',
    '| 输入 | 模型 | ACC % | BA % | Macro F1 % | failure precision % | failure recall % | AUROC % |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        fmt=lambda k:f'{100*r[k+"_mean"]:.2f} ± {100*r[k+"_std"]:.2f}'
        lines.append(f'| {r["input"]} | {r["head"].upper()} | {fmt("accuracy")} | {fmt("balanced_accuracy")} | {fmt("macro_f1")} | {fmt("failure_precision")} | {fmt("failure_recall")} | {fmt("auroc")} |')
    best=max(summary,key=lambda r:r['balanced_accuracy_mean'])
    lines+=['',f'平均test BA最高为{best["input"]} {best["head"].upper()}：{100*best["balanced_accuracy_mean"]:.2f}% ± {100*best["balanced_accuracy_std"]:.2f}%；macroF1={100*best["macro_f1_mean"]:.2f}% ± {100*best["macro_f1_std"]:.2f}%。这是测试描述，checkpoint仍由val选择。','',
    '![五seed对比](comparison.png)','', '![Seed42原始与归一化混淆矩阵](confusion_seed42.png)','',
    '混淆矩阵行=GT、列=预测，顺序success/failure。下排按GT行归一化，各行100%，对角线是对应recall；全部30次矩阵和support见confusion_normalized.json。','',
    '## 证据的限制与产物','',
    '测试只有17个独立interval，其中failure仅3个；一次failure误判就改变failure recall 33.33个百分点、BA16.67个百分点。五seed不是五个独立测试集。这个诊断只能说明当前split和probe下的可分性，不能据此证明泛化或在线Key检测能力。不同任务的ACC/BA/F1不能直接比较。配对seed差异见paired_seed_differences.json；常数基线见constant_baselines.json。','',
    '原始数据interval_manifest.json保留边界、合并成员、split、每interval窗口数；features/<rollout>.npz保留raw采样帧/ticks、单次窗口F6特征、末帧Deform和interval标签。runs/seed_<seed>/<input>_<head>/保存best.pt、history.json、metrics.json、test_predictions.csv/npz，一行对应一个interval。summary.csv、per_seed.csv、per_class.csv提供全部指标。','',
    '全部30次指标重算、固定标签及split、源文件/encoder哈希、权重、标准化及epoch选择均检查通过；6个seed42 checkpoint在CPU重放。输入特征全部复用当前单次窗口encoder缓存，未使用旧的16特征叠加输入。','',
    '```bash','source ./project_env.sh',f'PYTHONPATH="$PROJECT_ROOT/tools" CUBLAS_WORKSPACE_CONFIG=:4096:8 bash tools/run_trex.sh python -m sharpa_tactile.merged_interval_binary prepare --source {dm["source"]} --output outputs/sharpa_merged_interval_binary_data/<new_timestamp> --device cuda:0',f'PYTHONPATH="$PROJECT_ROOT/tools" CUBLAS_WORKSPACE_CONFIG=:4096:8 bash tools/run_trex.sh python -m sharpa_tactile.merged_interval_binary train --source {dataset.relative_to(ROOT)} --output outputs/sharpa_merged_interval_binary/<new_timestamp> --device cuda:0',f'PYTHONPATH="$PROJECT_ROOT/tools" bash tools/run_trex.sh python -m sharpa_tactile.report_merged_interval_binary --output {output.relative_to(ROOT)}','```']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    (dataset/'README.md').write_text('# 合并Align interval binary数据\n\n'+'\n'.join(lines[4:lines.index('## 指标及结果')])+'\n\n训练报告：`'+str(output.relative_to(ROOT))+'/README.md`。\n')
    for name in ('report_merged_interval_binary.py','merged_online.py','train_intervals.py','train.py'):
        shutil.copy2(ROOT/'tools/sharpa_tactile'/name,output/'code_snapshot'/name)
    target=ROOT/'WeeklySummary/10.5/merged_interval_binary';target.mkdir(parents=True,exist_ok=False);copies=[]
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(src)==sha(dst)
        copies.append(dict(source=str(src.relative_to(ROOT)),destination=str(dst.relative_to(ROOT)),sha256=sha(src)))
    for name in ('README.md','summary.csv','per_seed.csv','per_class.csv','paired_seed_differences.json','constant_baselines.json','comparison.png','confusion_seed42.png','confusion_normalized.json','verification.json','run_manifest.json'):copy(output/name,target/name)
    for name in ('interval_manifest.json','group_equivalence.json'):copy(dataset/name,target/name)
    for result in results:
        rel=Path('runs')/f'seed_{result["seed"]}'/(result['input']+'_'+result['head'])
        for name in ('metrics.json','history.json','test_predictions.csv'):copy(output/rel/name,target/rel/name)
    dump(target/'copy_manifest.json',copies)
    (target/'SOURCE_INDEX.md').write_text(f'# 来源\n\n数据：`{dataset.relative_to(ROOT)}`\n\n训练：`{output.relative_to(ROOT)}`\n\n权重与特征保留原目录，本目录复制文档和指标。\n')
    weekly=ROOT/'WeeklySummary/10.5/10.5.md';text=weekly.read_text()
    section=['## 实验 9：合并 interval binary 可分性（30 次）','',
    '实验8两组只在Key带上不同，merged interval完全相同。binary移除Key/in_progress后两组同一数据，6模型×5seeds=30次独立训练，共享结果，不把重复跑相同数据视为两个对照。','',
    '一条merged Align interval一个样本，success0/failure1。原split80/17/17，类数量64/16、14/3、14/3。interval内step3的16点原始F6窗口各编码一次→1280D，Deform仅窗口末帧→2560D，组成变长特征序列。MLP对整个interval均值池化分类；GRU取完整序列最后hidden，状态不跨interval。encoder冻结，二分类softmax，failure概率≥.5判failure。CE按interval数量加权[.625,2.5]，val BA选epoch。','',
    '| 输入 | 模型 | ACC % | BA % | Macro F1 % |','|---|---|---:|---:|---:|']
    for r in summary:
        fmt=lambda k:f'{100*r[k+"_mean"]:.2f} ± {100*r[k+"_std"]:.2f}'
        section.append(f'| {r["input"]} | {r["head"].upper()} | {fmt("accuracy")} | {fmt("balanced_accuracy")} | {fmt("macro_f1")} |')
    section+=['','指标以interval计数、各seed独立计算后均值±样本标准差。全预测success的ACC82.35%、BA50%，用于防止误读ACC。test仅3个failure，少数类误判一个会使BA变化16.67个百分点；诊断最终结果可分性，不代表在线结束Key检测。','',
    '[完整GT/loss/结构与报告](merged_interval_binary/README.md) · [每seed](merged_interval_binary/per_seed.csv) · [逐类指标](merged_interval_binary/per_class.csv) · [两组等价核对](merged_interval_binary/group_equivalence.json)','',
    '![五seed结果](merged_interval_binary/comparison.png)','', '![原始及归一化混淆矩阵](merged_interval_binary/confusion_seed42.png)','']
    text=text.replace('## 结果应如何对照','\n'.join(section)+'\n## 结果应如何对照');weekly.write_text(text)
    dump(output/'delivery_audit.json',dict(status='PASS',copied_files=len(copies),weekly_updated=True,original_data_readme=True,original_training_readme=True))
    print('INTERVAL_REPORT_PASS',json.dumps(summary),flush=True)

if __name__=='__main__':main()
