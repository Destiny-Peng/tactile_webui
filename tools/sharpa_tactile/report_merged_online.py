"""Verify and publish one-encoder-window merged-rollout experiments."""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, dump, sha
from .merged_online import MergedProbe, load, regions, label, features
from .align_windows import metrics
from .normalize_align_online import normalized_confusions


def csv_write(path,records):
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output.resolve()
    run=json.loads((output/'run_manifest.json').read_text());dataset=ROOT/run['dataset']
    dm=json.loads((dataset/'dataset_manifest.json').read_text());rows=dm['rollouts'];results=json.loads((output/'results.json').read_text())
    assert len(results)==60
    sensor=json.loads((ROOT/dm['source']/'source_sensor_integrity.json').read_text())
    assert sensor['status']=='PASS'
    assert all(dm['input_hashes'][path]==value for path,value in sensor['source_sensor_hashes'].items())
    assert all(sha(ROOT/p)==value for p,value in run['input_hashes'].items())
    assert all(sha(ROOT/p)==value for p,value in dm['input_hashes'].items())
    banks=load(dataset/'features.npz');tables={k:torch.from_numpy(v) for k,v in banks.items()};meta=load(dataset/'windows.npz')
    paths=json.loads((dataset/'paths.json').read_text());torch.set_num_threads(4)
    assert (np.diff(meta['ticks'],axis=1)>=0).all()
    assert all(np.all(meta['frames'][meta['rollout_index']==i]>=row['annotation_start']) and np.all(meta['frames'][meta['rollout_index']==i]<=row['annotation_end']) for i,row in enumerate(rows))
    for row in rows:
        aligns=[e for e in row['events'] if e['event_key'] in (6,8)]
        unit=row['merged_align'];assert unit['start_frame']==min(e['start_frame'] for e in aligns) and unit['end_frame']==max(e['end_frame'] for e in aligns)
        assert set(unit['member_indices'])=={e['event_index'] for e in aligns}
    split=json.loads((dataset/'split_manifest.json').read_text())
    assert len(set(split['train'])&set(split['test']))==len(set(split['train'])&set(split['val']))==len(set(split['val'])&set(split['test']))==0
    per_seed=[];per_class=[];replays=0;causal_checks=0;pred_meta=[]
    for result in results:
        directory=output/'runs'/result['group']/f'seed_{result["seed"]}'/(result['input']+'_'+result['head'])
        predictions=load(directory/'test_predictions.npz');ready=load(dataset/'ready'/result['group']/'test.npz')
        assert all(np.array_equal(predictions[k],v) for k,v in ready.items())
        recalculated=metrics(ready['labels'],predictions['probabilities']);recalculated['unit']='one encoder window prediction'
        assert recalculated==result['test'] and np.array_equal(predictions['predictions'],predictions['probabilities'].argmax(1))
        history=json.loads((directory/'history.json').read_text());assert max(history,key=lambda h:h['val_balanced_accuracy'])['epoch']==result['best_epoch']
        assert result['class_weights']==[1.0,1.0,1.0]
        if result['seed']==42:
            blob=torch.load(directory/'best.pt',map_location='cpu',weights_only=True);model=MergedProbe(**blob['model_config']);model.load_state_dict(blob['state_dict']);model.eval()
            path=next(p for p in paths if p['step']==3 and rows[p['rollout_index']]['split']=='test')
            ids=np.array(path['indices'][:8]);pos={int(i):j for j,i in enumerate(ready['indices'])}
            with torch.inference_mode():
                index=torch.from_numpy(ids if result['head']=='mlp' else ids[None])
                values=features(tables,index,result['input']);logits,state=model(**values)
                p=logits.softmax(-1).numpy();p=p if result['head']=='mlp' else p[0]
                np.testing.assert_allclose(p,predictions['probabilities'][[pos[int(i)] for i in ids]],atol=3e-5,rtol=3e-4)
                if result['head']=='gru':
                    changed={k:v.clone() if v is not None else None for k,v in values.items()}
                    for v in changed.values():
                        if v is not None:v[:,4:]+=100
                    altered,_=model(**changed);torch.testing.assert_close(logits[:,:4],altered[:,:4],rtol=0,atol=0)
                    a,h=model(**{k:v[:,:4] if v is not None else None for k,v in values.items()})
                    b,_=model(**{k:v[:,4:] if v is not None else None for k,v in values.items()},state=h)
                    torch.testing.assert_close(torch.cat([a,b],1),logits,atol=1e-6,rtol=1e-5);causal_checks+=1
            replays+=1
        rec={k:result[k] for k in ('group','input','head','seed','best_epoch','epochs')}
        rec.update({k:result['test'][k] for k in ('accuracy','balanced_accuracy','macro_f1','failure_false_positive_rate')});per_seed.append(rec)
        for name,value in result['test']['per_class'].items():per_class.append(dict(group=result['group'],input=result['input'],head=result['head'],seed=result['seed'],label=name,**value))
        records=[]
        for j,i in enumerate(ready['indices']):
            ri=int(meta['rollout_index'][i]);row=rows[ri];assert row['split']=='test'
            assert label(row,result['group'],int(ready['n'][j]),meta['frames'][i:i+1])[0]==ready['labels'][j]
            p=predictions['probabilities'][j]
            records.append(dict(rollout_id=row['rollout_id'],window_index=int(i),sample_frames=json.dumps(meta['frames'][i].tolist()),sample_ticks=json.dumps(meta['ticks'][i].tolist()),last_frame=int(meta['frames'][i,-1]),n=int(ready['n'][j]),step=int(ready['step'][j]),label=int(ready['labels'][j]),prediction=int(p.argmax()),p_in_progress=float(p[0]),p_success=float(p[1]),p_failure=float(p[2])))
        csv_write(directory/'test_predictions.csv',records)
    csv_write(output/'per_seed.csv',per_seed);csv_write(output/'per_class.csv',per_class)
    summary=[];paired=[]
    for group in run['groups']:
        for kind in ('f6','deform','f6_deform'):
            for head in ('mlp','gru'):
                records=[r for r in per_seed if (r['group'],r['input'],r['head'])==(group,kind,head)];assert len(records)==5
                rec=dict(group=group,input=kind,head=head,seeds=5)
                for k in ('accuracy','balanced_accuracy','macro_f1','failure_false_positive_rate'):
                    values=[r[k] for r in records];rec[k+'_mean']=float(np.mean(values));rec[k+'_std']=float(np.std(values,ddof=1))
                for name in ('in_progress','success','failure'):
                    for k in ('precision','recall','f1'):
                        values=[r[k] for r in per_class if (r['group'],r['input'],r['head'],r['label'])==(group,kind,head,name)]
                        rec[name+'_'+k+'_mean']=float(np.mean(values));rec[name+'_'+k+'_std']=float(np.std(values,ddof=1))
                summary.append(rec)
        for head in ('mlp','gru'):
            for single in ('f6','deform'):
                diffs=[]
                for seed in range(42,47):
                    a=next(r for r in per_seed if (r['group'],r['head'],r['seed'],r['input'])==(group,head,seed,'f6_deform'))
                    b=next(r for r in per_seed if (r['group'],r['head'],r['seed'],r['input'])==(group,head,seed,single));diffs.append(a['balanced_accuracy']-b['balanced_accuracy'])
                paired.append(dict(group=group,head=head,comparison='Fusion minus '+single,seeds=list(range(42,47)),BA_differences=diffs,mean=float(np.mean(diffs)),std=float(np.std(diffs,ddof=1))))
    csv_write(output/'summary.csv',summary);dump(output/'paired_seed_differences.json',paired);normalized_confusions(output,results)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,group in zip(axes,run['groups']):
        records=[r for r in summary if r['group']==group]
        ax.bar(range(6),[r['balanced_accuracy_mean']*100 for r in records],yerr=[r['balanced_accuracy_std']*100 for r in records],capsize=3)
        ax.set_xticks(range(6),[r['input'].replace('f6_deform','Fusion')+'\n'+r['head'].upper() for r in records]);ax.set_ylim(0,100);ax.set_ylabel('Test BA (%)');ax.set_title(group)
    fig.tight_layout();fig.savefig(output/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,6,figsize=(18,6))
    for i,group in enumerate(run['groups']):
        for ax,result in zip(axes[i],[r for r in results if r['group']==group and r['seed']==42]):
            cm=np.asarray(result['test']['confusion_matrix']);ax.imshow(cm,cmap='Blues')
            for y in range(3):
                for x in range(3):ax.text(x,y,str(cm[y,x]),ha='center',va='center')
            ax.set_xticks(range(3),['Prog','Succ','Fail']);ax.set_yticks(range(3),['Prog','Succ','Fail']);ax.set_title(result['input']+' '+result['head']);ax.set_xlabel('Prediction');ax.set_ylabel('GT')
    fig.suptitle('Seed 42; top=symmetric merged; bottom=asymmetric merged');fig.tight_layout();fig.savefig(output/'confusion_seed42.png',dpi=160);plt.close(fig)
    dump(output/'verification.json',dict(status='PASS',runs=60,source_sensor_baseline_verified=True,source_hashes_unchanged=True,fixed_ready_arrays=True,metrics_recomputed=True,one_align_per_rollout=True,rollout_split_checked=True,checkpoint_cpu_replays=replays,causal_and_streaming_gru_checks=causal_checks,normalization='train selected unique windows only',selection='validation BA first maximum'))
    lines=['# 全 rollout 合并 + 单次窗口 encoder 的三分类实验','',
    f'训练目录：`{output.relative_to(ROOT)}`；数据集：`{dataset.relative_to(ROOT)}`。2 组 × F6/Deform/Fusion × MLP/单向 GRU × seeds 42–46，共 60 次。','',
    '## 本轮修正','',
    '114 个 rollout（92 success、22 failure）均只对应一个 Align interval。取全部 Align 6/8 的最早 start 至最晚 end，中间空隙纳入；根据原 manifest 的 ground_truth_outcome 和最终 Align 标签核对，success rollout 合并为 success、failure rollout 的多个 failure interval 也合并为一个 failure。仅保留最后 end 的 Key。那 11 个无 6/7/8/9 标注的 rollout 继续排除。Insert 仅提供 Key 上限和窗口上下文，不独立分类。','',
    'symmetric_merged：Key=[merged end−n, merged end+n]。asymmetric_merged：Key=[merged end−n,终态 Align 后第一个 Insert start+n]；无后续 Insert 则回退 end+n。两组都合并，区别仅为 Key 的上限；Key 裁剪到该 rollout 的所有标注首 start/末 end 范围。合并 interval outcome 为 success/failure，但不会将其全部帧直接赋成该类：窗口后 8 个采样点命中最终 Key 才赋该类，否则 in_progress=0；保留原始间隙上下文，没有早期 failure Key。','',
    '## 输入、模型和输出','',
    '对末帧 t，输入恰为 [t−15×step,…,t] 的 16 个实际采样点（起点不足时重复范围内首个有效帧）。F6 将这个 [16,5,6] 原始窗口交给冻结 encoder **一次**，得到 [5,256]，flatten→1280D。实际官方 midtrain checkpoint 是 finger mode，不改为 hand mode；代码默认 hand mode 与本 checkpoint 的配置不同。不会给下游再叠 16 个 F6 encoder 特征。Deform 只取同一窗口末帧的五指 [5,1,240,240]，冻结 encoder 后每指 AdaptiveAvgPool(2×2)→512D，五指拼为2560D；复用对应末帧的冻结 Deform cache。','',
    'F6→Linear(1280,128)；Deform→Linear(2560,128)；Fusion 对同一窗口的两路投影 concat→256D。MLP：单个窗口向量→Linear(128或256,128)→ReLU→Dropout(0.1)→Linear(128,3)，没有额外 temporal pooling。GRU：按 rollout/step 的连续窗口末帧时间顺序输入向量序列→单层单向 GRU(hidden128)→每个 timestep hidden→Linear(128,3)，窗口间保留历史，不在每个窗口重置。rollout 或真实 tactile tick 缺口重置状态，原 Align 边界/间隙不重置。batch 只打乱独立序列，序列内部保持时间顺序。','',
    'softmax 输出 P(in_progress),P(success),P(failure)，argmax 对应当前窗口一个类别。MLP 无跨窗历史；GRU 只使用当前及过去的窗口。GT 仍只由当前窗口后 8 个实际采样点倒推最近 Key 决定，无 Gaussian/soft/ignore。输出不拼接成预测 interval。','',
    '## 采样、Loss、Split','',
    '原 rollout split=80/17/17 保持不变。参考 n=5、step=3；窗口终点每个有效相机帧推进，step 控制 16 点内部间隔。训练 success/failure 使用 n=0/2/5/10、step=1/3/5/8/12 的候选，按完整 16 点输入去重，优先 n=5 和 step=3；in_progress 只用参考采样，移除与正类候选相同输入的冲突，再无放回下采样三类为1:1:1。val/test保留 n5/step3 的完整自然分布。','',
    'MLP 训练直接使用被选中的窗口。GRU 为每个 rollout/step/tactile 连续段保留完整窗口序列作为上下文，只有平衡抽样选中的窗口计算 loss；未选中窗口不作新的监督样本，同一监督窗口只计一次。完整序列 BPTT，无额外16点下游窗口、无 TBPTT。验证/测试按自然序列逐窗输出，状态不跨 rollout。','',
    'weighted CE 按监督窗口类数量计算逆频率并归一化到均值1；本轮三类平衡，权重=[1,1,1]。特征标准化只在被选中训练窗口的唯一特征上拟合，std 下限0.01，不读取 holdout 拟合。AdamW lr=.001、wd=.0001，梯度clip1，最多30epochs，patience8；MLP batch128，GRU完整序列batch8。只根据验证BA选最大值checkpoint，平分取首次，随后测试。两个encoder始终冻结。','',
    '| 组别 | Split | in_progress | success | failure | 总窗口 |','|---|---|---:|---:|---:|---:|']
    for r in dm['distribution']:lines.append(f'| {r["group"]} | {r["split"]} | {r["counts"][0]} | {r["counts"][1]} | {r["counts"][2]} | {r["total"]} |')
    lines+=['','## 测试指标','',
    '评估单位是一个 encoder 输入窗口的一次预测。ACC=正确数/窗口总数；各类 P=TP/(TP+FP)、R=TP/(TP+FN)、F1=2PR/(P+R)，零分母计0；BA=三类recall均值，macro F1=三类F1均值；failure precision只针对failure类，failure FPR=非failure被预测failure的比例。指标不使用训练权重。先独立计算各seed指标，再取均值±样本标准差(ddof1)，不是ensemble。','',
    '| 组别 | 输入 | 模型 | ACC % | BA % | Macro F1 % | failure precision % | failure recall % | failure FPR % |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        fmt=lambda k:f'{100*r[k+"_mean"]:.2f} ± {100*r[k+"_std"]:.2f}'
        lines.append(f'| {r["group"]} | {r["input"]} | {r["head"].upper()} | {fmt("accuracy")} | {fmt("balanced_accuracy")} | {fmt("macro_f1")} | {fmt("failure_precision")} | {fmt("failure_recall")} | {fmt("failure_false_positive_rate")} |')
    for group in run['groups']:
        best=max((r for r in summary if r['group']==group),key=lambda r:r['balanced_accuracy_mean'])
        lines+=['',f'{group} 平均 BA 最高：{best["input"]} {best["head"].upper()}，{100*best["balanced_accuracy_mean"]:.2f}% ± {100*best["balanced_accuracy_std"]:.2f}%。']
    lines+=['','![五seed对比](comparison.png)','', '## 混淆矩阵','',
    '行=GT，列=预测，顺序in_progress/success/failure。归一化每格除以该行GT数量，每行总和100%，对角线为recall；显示四舍五入可能有微小差异。下图seed42，全部60次归一化矩阵和support见confusion_normalized.json。','',
    '![原始计数](confusion_seed42.png)','', '![按GT行归一化](confusion_seed42_normalized.png)','',
    '## 解释与产物','',
    '本轮同时修正了失败rollout的合并和下游输入/GRU历史处理，旧实验7分数不能直接当作本轮基线。两组Key/GT仍不同，比较时先看同组模态和模型；5seeds只衡量训练随机性，未覆盖新的split。测试failure来自3个failure-only rollout，窗口重叠不构成独立试验，不能把窗口支持量当成独立数据量。','',
    'summary.csv为12组汇总，per_seed.csv为60次ACC/BA/F1/FPR，per_class.csv为逐类precision/recall/F1/support，paired_seed_differences.json为同组Fusion减单模态的配对seed BA差异。runs/<group>/seed_<seed>/<input>_<head>/保存checkpoint、history、metrics、逐窗概率CSV/NPZ。dataset_manifest.json保留每个rollout的合并成员和Key边界；windows.npz记录exact16采样帧/ticks，features.npz记录单次F6编码及末帧Deform。','',
    '验证全部60次测试指标、固定ready数组、原split和输入/encoder哈希；12个seed42 checkpoint CPU重放，6个GRU通过前缀因果与分段携带状态等价检查。','',
    '```bash',f'PYTHONPATH="$PROJECT_ROOT/tools" CUBLAS_WORKSPACE_CONFIG=:4096:8 bash tools/run_trex.sh python -m sharpa_tactile.merged_online prepare --source {dm["source"]} --output outputs/sharpa_merged_online_datasets/<new_timestamp> --device cuda:0',f'PYTHONPATH="$PROJECT_ROOT/tools" CUBLAS_WORKSPACE_CONFIG=:4096:8 bash tools/run_trex.sh python -m sharpa_tactile.merged_online train --source {dataset.relative_to(ROOT)} --output outputs/sharpa_merged_online_training/<new_timestamp> --device cuda:0',f'PYTHONPATH="$PROJECT_ROOT/tools" bash tools/run_trex.sh python -m sharpa_tactile.report_merged_online --output {output.relative_to(ROOT)}','```']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    (dataset/'README.md').write_text('# 全 rollout 合并的单窗口特征数据集\n\n'+ '\n'.join(lines[4:lines.index('## 测试指标')])+'\n\n训练报告：`'+str(output.relative_to(ROOT))+'/README.md`。数据生成时 training_started=false 为生成阶段记录。\n')
    target=ROOT/'WeeklySummary/10.5/merged_online';target.mkdir(parents=True,exist_ok=False);copies=[]
    def copy(source,dest):
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);assert sha(source)==sha(dest)
        copies.append(dict(source=str(source.relative_to(ROOT)),destination=str(dest.relative_to(ROOT)),sha256=sha(source)))
    for name in ('README.md','summary.csv','per_seed.csv','per_class.csv','paired_seed_differences.json','verification.json','run_manifest.json','comparison.png','confusion_seed42.png','confusion_seed42_normalized.png','confusion_normalized.json'):copy(output/name,target/name)
    copy(dataset/'dataset_manifest.json',target/'dataset_manifest.json')
    for result in results:
        rel=Path('runs')/result['group']/f'seed_{result["seed"]}'/(result['input']+'_'+result['head'])
        for name in ('history.json','metrics.json','test_predictions.csv'):copy(output/rel/name,target/rel/name)
    dump(target/'copy_manifest.json',copies)
    (target/'SOURCE_INDEX.md').write_text(f'# 来源\n\n原始训练：`{output.relative_to(ROOT)}`\n\n原始数据：`{dataset.relative_to(ROOT)}`\n\n权重、特征、完整预测NPZ在原目录；本目录复制文档和指标。\n')
    shutil.copy2(Path(__file__),output/'code_snapshot'/Path(__file__).name)
    weekly=ROOT/'WeeklySummary/10.5/10.5.md';text=weekly.read_text()
    text=text.replace('## 实验 7：修正数据集的训练测试（60 次）','## 实验 7：旧输入实现的训练测试（60 次，历史结果）')
    old='## 实验 7：旧输入实现的训练测试（60 次，历史结果）'
    text=text.replace(old,old+'\n\n本实验将16个F6 encoder特征又叠成下游窗口、Deform使用整个序列，且failure-only的多个interval未合并；与用户澄清后的设计不一致。本轮正确输入和全部rollout合并见实验8；此处保留历史记录。')
    section=['## 实验 8：全 rollout 合并 + 单次窗口 encoder（60 次）','',
    '92 success + 22 failure rollout各合并为一个Align interval，包含中间间隙，仅保留最终Key。排除11个无目标标注rollout。两组均合并，仅Key分别采用[end−n,end+n]与[end−n,insert start+n]。GT仍是后半窗口实际采样点命中Key的三分类；未命中为in_progress。','',
    '每个[16,5,6]原始F6窗口只编码一次，checkpoint为finger mode→1280D；Deform只编码窗口末帧→2560D。两路投影128D，Fusion concat256D。MLP直接分类，无额外时间池化；GRU按rollout连续窗口序列输出，跨窗口保留历史，仅在rollout/tactile缺口重置。训练平衡样本以loss mask实现，未选中窗口保留上下文但不计算loss。CE权重[1,1,1]，val/test自然分布，原split固定，验证BA选checkpoint。','',
    '| 组别 | 输入 | 模型 | ACC % | BA % | Macro F1 % |','|---|---|---|---:|---:|---:|']
    for r in summary:
        fmt=lambda k:f'{100*r[k+"_mean"]:.2f} ± {100*r[k+"_std"]:.2f}'
        section.append(f'| {r["group"]} | {r["input"]} | {r["head"].upper()} | {fmt("accuracy")} | {fmt("balanced_accuracy")} | {fmt("macro_f1")} |')
    section+=['','指标按窗口计数，BA为三类recall平均，macro F1为三类F1平均；分别算各seed后取均值±样本标准差。旧实验7输入和标签不同，不能直接解释分数差异为提升；failure测试只有三个rollout。','',
    '[完整GT、输入、loss及结果](merged_online/README.md) · [每seed](merged_online/per_seed.csv) · [逐类指标](merged_online/per_class.csv) · [合并清单](merged_online/dataset_manifest.json)','',
    '![五seed结果](merged_online/comparison.png)','', '归一化混淆矩阵：行=GT、列=预测，每行100%，对角线为recall，seed42。','', '![归一化混淆矩阵](merged_online/confusion_seed42_normalized.png)','']
    text=text.replace('## 结果应如何对照','\n'.join(section)+'\n## 结果应如何对照');weekly.write_text(text)
    dump(output/'delivery_audit.json',dict(status='PASS',copied_files=len(copies),original_data_readme=True,original_training_readme=True,weekly_updated=True))
    print('REPORT_PASS',json.dumps(summary),flush=True)

if __name__=='__main__':main()
