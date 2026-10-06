"""Audit endpoint supervision, causality, streaming, and prefix progress metrics."""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,dump,sha
from .causal_prefix import PrefixProbe,read_samples,endpoint_indices,prefix_loss,evaluate
from .prefix_inference import EncodedPrefixPredictor
from .train_intervals import metrics,normalization
from .merged_online import load


def csv_write(path,records):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();output=a.output.resolve()
    run=json.loads((output/'run_manifest.json').read_text());dataset=ROOT/run['dataset'];dm=json.loads((dataset/'prefix_manifest.json').read_text());samples=read_samples(dataset,dm)
    results=json.loads((output/'results.json').read_text());assert len(results)==30
    assert all(sha(ROOT/path)==value for path,value in run['input_hashes'].items())
    assert all(sha(ROOT/path)==value for path,value in dm['input_hashes'].items())
    split=json.loads((dataset/'split_manifest.json').read_text());assert sha(dataset/'split_manifest.json')==dm['split_sha256']
    assert not(set(split['train'])&set(split['val']) or set(split['train'])&set(split['test']) or set(split['val'])&set(split['test']))
    for row in dm['records']:
        data=load(dataset/row['feature_path']);assert np.array_equal(data['endpoints'],endpoint_indices(row['ticks']))
        assert len(data['endpoints'])==16 and data['endpoints'][-1]==row['ticks']-1
        assert np.array_equal(data['endpoint_labels'],np.full(16,row['label']))
        assert row['label']==int(row['event_key'] in (6,7)) and row['rollout_id'] in split[row['split']]
        assert data['video_frames'].min()>=row['start_frame'] and data['video_frames'].max()<=row['end_frame']
    torch.set_num_threads(4);norm=normalization(samples['train']);per_seed=[];per_endpoint=[];normalized=[];checks=[]
    for result in results:
        directory=output/'runs'/f'seed_{result["seed"]}'/(result['input']+'_'+result['head']);stored=load(directory/'test_predictions.npz');labels=np.array([r['label'] for r in samples['test']])
        assert np.array_equal(stored['labels'],labels) and np.array_equal(stored['endpoints'],np.stack([r['endpoints'] for r in samples['test']]))
        assert stored['probabilities'].shape==(39,16) and np.array_equal(stored['predictions'],(stored['probabilities']>=.5).astype(int))
        for k in range(16):assert metrics(labels,stored['probabilities'][:,k])==result['test']['per_endpoint'][k]
        pooled=metrics(np.repeat(labels,16),stored['probabilities'].reshape(-1));pooled['unit']='equal16 causal positions per interval (correlated within interval)';assert pooled==result['test']['pooled_positions']
        history=json.loads((directory/'history.json').read_text());assert max(history,key=lambda r:r['val_mean_prefix_balanced_accuracy'])['epoch']==result['best_epoch']
        assert all(r['train_intervals']==184 and r['train_supervised_slots']==2944 for r in history)
        assert np.isclose(result['pos_weight'],129/55)
        blob=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
        for k,v in norm.items():torch.testing.assert_close(blob['state_dict'][k],v,rtol=0,atol=0)
        if result['seed']==42:
            model=PrefixProbe(**blob['model_config']);model.load_state_dict(blob['state_dict']);model.eval();row=samples['test'][0]
            f6=torch.from_numpy(row['f6'])[None];deform=torch.from_numpy(row['deform'])[None];cut=int(row['endpoints'][7])+1
            with torch.inference_mode():
                logits,_=model(f6,deform);p=logits.sigmoid()[0,row['endpoints']].numpy()
                np.testing.assert_allclose(p,stored['probabilities'][0],atol=3e-5,rtol=3e-4)
                changed_f6=f6.clone();changed_deform=deform.clone();changed_f6[:,cut:]+=100;changed_deform[:,cut:]-=100
                altered,_=model(changed_f6,changed_deform);torch.testing.assert_close(logits[:,:cut],altered[:,:cut],rtol=0,atol=0)
                truncated,_=model(f6[:,:cut],deform[:,:cut]);torch.testing.assert_close(logits[:,:cut],truncated,rtol=2e-5,atol=3e-6)
                # Equal interval weighting: selected slots are averaged, independently of sequence lengths.
                selected=logits[:,row['endpoints']].repeat(2,1);target=torch.tensor([0,1]);w=torch.tensor(129/55)
                actual=prefix_loss(selected,target,w)
                expected=torch.nn.functional.binary_cross_entropy_with_logits(selected,target.float()[:,None].expand_as(selected),pos_weight=w,reduction='none').mean(1).mean()
                torch.testing.assert_close(actual,expected)
            streaming=EncodedPrefixPredictor(directory/'best.pt');probabilities=[]
            for ff,dd in zip(row['f6'],row['deform']):probabilities.append(streaming.step(ff,dd))
            np.testing.assert_allclose(probabilities,logits.sigmoid()[0].numpy(),atol=5e-6,rtol=3e-5)
            streaming.reset();assert streaming.positions_seen==0
            assert np.isclose(streaming.step(row['f6'][0],row['deform'][0]),probabilities[0],atol=2e-6)
            checks.append(dict(input=result['input'],head=result['head'],cpu_replay=True,no_future_dependence=True,truncated_prefix_equivalent=True,streaming_equivalent=True,interval_reset=True))
        row={k:result[k] for k in ('input','head','seed','best_epoch','epochs')}
        row.update(mean_prefix_balanced_accuracy=result['test']['mean_prefix_balanced_accuracy'],mean_prefix_macro_f1=result['test']['mean_prefix_macro_f1'])
        row.update({'final_'+k:result['test']['final_endpoint'][k] for k in ('accuracy','balanced_accuracy','macro_f1','failure_precision','failure_recall','auroc')});per_seed.append(row)
        for k,entry in enumerate(result['test']['per_endpoint']):
            per_endpoint.append(dict(input=result['input'],head=result['head'],seed=result['seed'],position=k+1,fraction=(k+1)/16,**{m:entry[m] for m in ('accuracy','balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_false_positive_rate','auroc')}))
        for stage in ('pooled_positions','final_endpoint'):
            cm=np.asarray(result['test'][stage]['confusion_matrix_success_failure'],dtype=float);support=cm.sum(1,keepdims=True);normal=cm/support
            np.testing.assert_allclose(normal.sum(1),1)
            normalized.append(dict(input=result['input'],head=result['head'],seed=result['seed'],stage=stage,class_order=['success','failure'],support=support[:,0].astype(int).tolist(),matrix=normal.tolist()))
    csv_write(output/'per_seed.csv',per_seed);csv_write(output/'per_endpoint.csv',per_endpoint);dump(output/'confusion_normalized.json',normalized)
    summary=[];curves=[]
    for kind in ('f6','deform','f6_deform'):
        for head in ('mlp','gru'):
            selected=[r for r in per_seed if (r['input'],r['head'])==(kind,head)];assert len(selected)==5;row=dict(input=kind,head=head,seeds=5)
            for key in ('mean_prefix_balanced_accuracy','mean_prefix_macro_f1','final_accuracy','final_balanced_accuracy','final_macro_f1','final_failure_precision','final_failure_recall','final_auroc'):
                values=[r[key] for r in selected];row[key+'_mean']=float(np.mean(values));row[key+'_std']=float(np.std(values,ddof=1))
            summary.append(row)
            for pos in range(1,17):
                selected=[r for r in per_endpoint if (r['input'],r['head'],r['position'])==(kind,head,pos)];row=dict(input=kind,head=head,position=pos,fraction=pos/16)
                for key in ('accuracy','balanced_accuracy','macro_f1','failure_precision','failure_recall','auroc'):
                    values=[r[key] for r in selected];row[key+'_mean']=float(np.mean(values));row[key+'_std']=float(np.std(values,ddof=1))
                curves.append(row)
    csv_write(output/'summary.csv',summary);csv_write(output/'prefix_curve_summary.csv',curves)
    event_counts=[]
    for splitname in ('train','val','test'):
        for event in (6,7,8,9):event_counts.append(dict(split=splitname,event=event,intervals=sum(r['split']==splitname and r['event_key']==event for r in dm['records'])))
    csv_write(output/'event_counts.csv',event_counts)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for kind in ('f6','deform','f6_deform'):
        for head in ('mlp','gru'):
            selected=[r for r in curves if (r['input'],r['head'])==(kind,head)];x=np.array([100*r['fraction'] for r in selected])
            for ax,key in zip(axes,('balanced_accuracy','macro_f1')):
                y=np.array([100*r[key+'_mean'] for r in selected]);std=np.array([100*r[key+'_std'] for r in selected]);ax.plot(x,y,label=kind.replace('f6_deform','Fusion')+' '+head.upper());ax.fill_between(x,y-std,y+std,alpha=.08)
                ax.set_xlabel('Requested fraction of valid interval sequence (%)');ax.set_ylabel(key+' (%)');ax.set_ylim(0,100)
    axes[0].axhline(50,color='gray',linestyle='--');axes[1].legend(fontsize=8);fig.suptitle('Causal hindsight outcome | mean ± seed SD | 16 positions');fig.tight_layout();fig.savefig(output/'prefix_curves.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,6,figsize=(18,6))
    for j,result in enumerate([r for r in results if r['seed']==42]):
        cm=np.asarray(result['test']['final_endpoint']['confusion_matrix_success_failure']);normalized_cm=cm/cm.sum(1,keepdims=True)
        for i,data in enumerate((cm,normalized_cm)):
            ax=axes[i,j];ax.imshow(data,cmap='Blues',vmin=0,vmax=1 if i else None)
            for yy in range(2):
                for xx in range(2):ax.text(xx,yy,f'{data[yy,xx]*100:.1f}%' if i else str(data[yy,xx]),ha='center',va='center',color='white' if i and data[yy,xx]>.55 else 'black')
            ax.set_xticks(range(2),['Success','Failure']);ax.set_yticks(range(2),['Success\nN=29','Failure\nN=10']);ax.set_xlabel('Prediction');ax.set_ylabel('GT');ax.set_title(result['input']+' '+result['head'].upper())
    fig.suptitle('Seed42 | 100% endpoint | top counts, bottom GT-row normalized');fig.tight_layout();fig.savefig(output/'confusion_final_seed42.png',dpi=160);plt.close(fig)
    dump(output/'verification.json',dict(status='PASS',runs=30,original_annotations_unchanged=True,original_event_intervals_259=True,rollout_split_isolated=True,sixteen_supervision_slots_per_interval=True,source_hashes_unchanged=True,metrics_recomputed=True,train_only_normalization_all_checkpoints=True,selection_mean_prefix_BA=True,causal_streaming_checks=checks))
    lines=['# Interval sequence + hindsight outcome：causal prefix 预测','',f'数据：`{dataset.relative_to(ROOT)}`；训练：`{output.relative_to(ROOT)}`。6模型×seeds42–46=30次训练。原始标注没有修改。','',
    '## Sample 和 GT','',
    '每个原始event interval一个样本，不合并Align/Insert，也不按rollout最终outcome覆盖早期event标签。event6/7=failure1，event8/9=success0；259个原始标注interval位于114个rollout。相同rollout所有interval和prefix保持同一split。没有Prog/background/Key-window三分类、Gaussian或soft label。这里event的类别决定最终outcome，start/end只定义序列范围。','',
    '先固定原rollout split，再构造序列与监督位置。train184interval(129success/55failure)、val36(28/8)、test39(29/10)，rollout split仍80/17/17。event_counts.csv给出各event数量。无窗口随机划分或增强副本，没有多个prefix文件。','',
    '每个interval只存一条完整稠密有效tick特征序列z1…zT。下游16个监督位置为ceil(k*T/16)−1（zero-based，k=1…16），目标进度6.25%、12.5%、…、100%；按有效序列位置均匀取点，非原相机墙钟时间百分比，缺失tick会使两者有差别。离散取整后实际进度为(index+1)/T；100%取最后有效位置。每个endpoint继承相同interval outcome，标签不代表当前动作已成功/失败；输出是P(final failure|start:current)。当前所有T≥20，16位置均不同；未来若T<16允许重复endpoint占据16slots，仍按16slots平均。','',
    '## Encoder 和 causal 下游','',
    'F6每个有效时刻独立使用截至当前的过去16连续tactile ticks，输入[16,5,6]→冻结官方finger-mode encoder→[5,256]→flatten1280D；interval开始不足16时，只重复首个有效interval tick左补齐，不读取interval之前或未来数据。Deform为同一当前tick五指map→冻结encoder+每指2×2pool→2560D。两个encoder始终冻结，复用已严格验证的interval-only缓存。','',
    'encoder window=16与prediction positions=16是两个独立参数。输入是完整稠密特征序列，既不将16个encoder输出叠为额外局部分类窗口，也不将序列稀疏截为16点。','',
    '每模态逐时刻Linear→128D，Fusion逐时刻concat256D。单层单向GRU(hidden128)处理完整序列，在每个时刻hidden→Linear(128,1)，sigmoid输出P(failure)；interval开始重置，内部保留历史。MLP对截至当前的投影特征作cumulative mean→Linear(128或256,128)→ReLU→Dropout(.1)→Linear(128,1)，因此也是causal prefix预测，均值不使用整个未来interval。模型不接收总长度、终点、进度fraction或GT作为输入。','',
    'EncodedPrefixPredictor接口位于tools/sharpa_tactile/prefix_inference.py：reset()开始新interval；step(f6_1280,deform_2560)输入当前冻结特征，返回当前P(final failure)，不需要知道interval end或16个监督位置。离线训练知道完整interval边界仅用于选监督点；在线可在每个有效时刻输出。原始传感器→冻结特征需遵守上述局部past16规则。','',
    '## Loss 与选择','',
    'BCEWithLogitsLoss，不把sigmoid后概率送进loss。failure pos_weight=训练success interval数/训练failure interval数=129/55≈2.34545；阴性权重1。先在每个interval的16个endpoint上取loss均值，再对batch的interval取均值：L=(1/B)Σ_i[(1/16)Σ_k BCE(logit_i,k,outcome_i)]。长interval不会增加监督项或额外loss权重；每epoch184interval/2944slots。padding和非endpoint位置不计算loss，但作为历史可影响未来endpoint。','',
    'BCE权重来自训练interval数量，不是feature时刻数量。没有概率单调约束，同标签监督不要求输出单调增加。两类不使用-1 ignore；batch padding只是长度补齐，不是数据标签。','',
    '训练只更新投影、MLP/GRU和head，标准化仅拟合train特征，std下限.01。AdamW lr=.001/wd=.0001、batch8、clip1，最多30epochs/patience8。checkpoint按val16位置的平均BA选最大值，平分取首次；不只按100%endpoint选，也不按多数类ACC选。阈值固定.5，没有test阈值调优。','',
    '## 评估怎么计算','',
    '每个位置独立在相同39个test interval上计算ACC、BA=(success recall+failure recall)/2、macroF1=(success F1+failure F1)/2及failure precision/recall。mean-prefix BA/F1是16个位置指标的算术均值；100%指标只在最终位置计算。mean-prefix BA等于把16位置合并后的BA，但mean-prefix F1一般不等于合并位置后的F1。pooled624positions仅供补充，它们属于39个interval，不是624个独立样本。','',
    '再对5seeds分别计算上述指标并取均值±样本标准差(ddof1)，不是ensemble。per_endpoint.csv含30×16=480条位置指标；prefix_curve_summary.csv是各位置的5seed汇总。','',
    '| 输入 | 模型 | mean-prefix BA % | mean-prefix Macro F1 % | 100% ACC % | 100% BA % | 100% Macro F1 % |','|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        fmt=lambda k:f'{100*r[k+"_mean"]:.2f} ± {100*r[k+"_std"]:.2f}'
        lines.append(f'| {r["input"]} | {r["head"].upper()} | {fmt("mean_prefix_balanced_accuracy")} | {fmt("mean_prefix_macro_f1")} | {fmt("final_accuracy")} | {fmt("final_balanced_accuracy")} | {fmt("final_macro_f1")} |')
    lines+=['','![Prefix曲线](prefix_curves.png)','', '![100%原始与归一化矩阵](confusion_final_seed42.png)','',
    '混淆矩阵行=GT、列=预测，success/failure；下排GT行归一化，每行100%，对角线为recall。全部30次pooled及100%归一化矩阵见confusion_normalized.json。','',
    '## 验证与解释限制','',
    '全部30次位置指标重算；原始event、split、边界、16slots、train-only标准化及BCE权重检查通过。6个seed42 checkpoint CPU重放：改变未来特征不影响已有输出，截断prefix等价，逐时刻streaming与完整序列结果一致，reset不携带前一interval状态。相同GT的16位置只是hindsight监督，不会让模型读取未来feature。','',
    '本轮与实验9不同：使用原始6/7/8/9interval，不是合并Align114样本；同时改为16位置BCE监督，因此不能把分数差异仅归因于prefix监督。test39interval来自17rollout，不同interval及16prefix都可能相关；5seeds没有增加独立测试数据。这里只验证已知interval起点内的outcome预测，不解决在线如何找到interval起点。','',
    '## 产物与复现','',
    'prefix_manifest.json逐interval记录原边界、event、split、完整序列长度及16endpoint indices/ticks/frames；features/每interval一个完整NPZ，原scalar outcome与16endpoint_labels一致。runs/保存best.pt/history/metrics/test_predictions.csv/npz；CSV一行一个监督endpoint，保留interval ID、当前帧/tick、进度和概率，便于查看同一interval的预测演化。','',
    '```bash','source ./project_env.sh',f'PYTHONPATH="$PROJECT_ROOT/tools" CUBLAS_WORKSPACE_CONFIG=:4096:8 bash tools/run_trex.sh python -m sharpa_tactile.causal_prefix prepare --source {dm["source"]} --output outputs/sharpa_causal_prefix_data/<new_timestamp> --device cuda:0',f'PYTHONPATH="$PROJECT_ROOT/tools" CUBLAS_WORKSPACE_CONFIG=:4096:8 bash tools/run_trex.sh python -m sharpa_tactile.causal_prefix train --source {dataset.relative_to(ROOT)} --output outputs/sharpa_causal_prefix/<new_timestamp> --device cuda:0',f'PYTHONPATH="$PROJECT_ROOT/tools" bash tools/run_trex.sh python -m sharpa_tactile.report_causal_prefix --output {output.relative_to(ROOT)}','```']
    (output/'README.md').write_text('\n'.join(lines)+'\n');(dataset/'README.md').write_text('# Interval hindsight prefix数据\n\n'+'\n'.join(lines[4:lines.index('## 评估怎么计算')])+'\n\n训练报告：`'+str(output.relative_to(ROOT))+'/README.md`。\n')
    for name in ('report_causal_prefix.py','prefix_inference.py','train_intervals.py','train.py'):
        shutil.copy2(ROOT/'tools/sharpa_tactile'/name,output/'code_snapshot'/name)
    target=ROOT/'WeeklySummary/10.5/causal_prefix';target.mkdir(parents=True,exist_ok=False);copies=[]
    def copy(source,destination):
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination);assert sha(source)==sha(destination)
        copies.append(dict(source=str(source.relative_to(ROOT)),destination=str(destination.relative_to(ROOT)),sha256=sha(source)))
    for name in ('README.md','summary.csv','per_seed.csv','per_endpoint.csv','prefix_curve_summary.csv','event_counts.csv','confusion_normalized.json','prefix_curves.png','confusion_final_seed42.png','verification.json','run_manifest.json'):copy(output/name,target/name)
    copy(dataset/'prefix_manifest.json',target/'prefix_manifest.json')
    for result in results:
        relative=Path('runs')/f'seed_{result["seed"]}'/(result['input']+'_'+result['head'])
        for name in ('metrics.json','history.json','test_predictions.csv'):copy(output/relative/name,target/relative/name)
    dump(target/'copy_manifest.json',copies)
    (target/'SOURCE_INDEX.md').write_text(f'# 原始来源\n\n数据：`{dataset.relative_to(ROOT)}`\n\n训练：`{output.relative_to(ROOT)}`\n\n逐时刻推理：`tools/sharpa_tactile/prefix_inference.py`。权重、特征、完整预测NPZ保留在原目录。\n')
    weekly=ROOT/'WeeklySummary/10.5/10.5.md';text=weekly.read_text()
    section=['## 实验10：原始interval hindsight outcome / causal prefix（30次）','',
    '原始标注不变，259个event6/7/8/9 interval各一条完整序列（不是合并114个Align样本）。failure6/7=1，success8/9=0；原rollout split不变，train184(129/55)、val36(28/8)、test39(29/10)。没有background/progress/Key-window GT。','',
    'F6每tick使用过去16raw ticks编码一次，Deform同tick编码；完整稠密序列送入causal GRU，或MLP截至当前的cumulative mean。16个监督位置均匀覆盖有效序列6.25%…100%，不是encoder window；全部继承interval最终outcome。BCE先mean16slots，再meanintervals，pos_weight=129/55；长interval不增加loss权重。sigmoid预测P(final failure|start:current)，阈值.5。验证以16位置平均BA选epoch。','',
    '| 输入 | 模型 | mean-prefix BA % | mean-prefix Macro F1 % | 100% BA % | 100% Macro F1 % |','|---|---|---:|---:|---:|---:|']
    for r in summary:
        fmt=lambda k:f'{100*r[k+"_mean"]:.2f} ± {100*r[k+"_std"]:.2f}'
        section.append(f'| {r["input"]} | {r["head"].upper()} | {fmt("mean_prefix_balanced_accuracy")} | {fmt("mean_prefix_macro_f1")} | {fmt("final_balanced_accuracy")} | {fmt("final_macro_f1")} |')
    section+=['','mean-prefix指标先在每位置对同39interval计算，再平均16位置；100%指标仅最终位置；最后对5seeds取均值±样本标准差。16prefix不构成16个独立interval。6checkpoint通过CPU重放、未来扰动不影响过去、截断prefix、streaming等价及interval reset验证。','',
    '[完整GT/input/loss/评估说明](causal_prefix/README.md) · [每位置指标](causal_prefix/per_endpoint.csv) · [逐interval endpoint及标签](causal_prefix/prefix_manifest.json)','',
    '![Prefix曲线](causal_prefix/prefix_curves.png)','', '![100%归一化及原始混淆矩阵](causal_prefix/confusion_final_seed42.png)','']
    text=text.replace('## 结果应如何对照','\n'.join(section)+'\n## 结果应如何对照');weekly.write_text(text)
    dump(output/'delivery_audit.json',dict(status='PASS',copied_files=len(copies),weekly_updated=True,original_data_readme=True,original_training_readme=True))
    print('PREFIX_REPORT_PASS',json.dumps(summary),flush=True)

if __name__=='__main__':main()
