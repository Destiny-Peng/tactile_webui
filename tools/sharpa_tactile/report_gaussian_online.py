"""Recompute Gaussian sweep metrics, audit causality, publish two-group reports."""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,dump,sha
from .gaussian_online import GROUPS,read_samples,evaluate
from .gaussian_targets import SIGMAS,CLASSES,one_sided_targets
from .merged_online import MergedProbe,load
from .align_windows import metrics
from .train_intervals import normalization


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();output=a.output.resolve()
    tm=json.loads((output/'training_manifest.json').read_text());source=ROOT/tm['source'];dm=json.loads((source/'dataset_manifest.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in tm['input_hashes'].items()) and all(sha(ROOT/p)==v for p,v in dm['input_hashes'].items())
    results=json.loads((output/'results.json').read_text());selection=json.loads((output/'sigma_selection.json').read_text());assert len(results)==120 and len(selection)==12
    split=json.loads((source/'split_manifest.json').read_text())
    assert not (set(split['train'])&set(split['val']) or set(split['train'])&set(split['test']) or set(split['val'])&set(split['test']))
    for r in dm['records']:
        assert r['rollout_id'] in split[r['split']]
        data=load(source/r['feature_path']);assert data['video_frames'].min()>=r['start'] and data['video_frames'].max()<=r['end']
        assert np.array_equal(data['labels'],np.where(data['video_frames']<r['key'],0,r['outcome']))
    torch.set_num_threads(4);samples={g:read_samples(source,dm,g) for g in GROUPS};norms={g:normalization(samples[g]['train']) for g in GROUPS}
    rows=[];normalized=[];checks=[];chosen_results=[]
    for result in results:
        directory=ROOT/result['directory'];stored=load(directory/'test_predictions.npz');test=samples[result['group']]['test']
        labels=np.concatenate([s['labels'] for s in test]);assert np.array_equal(stored['labels'],labels)
        recomputed=metrics(labels,stored['probabilities'])
        for key in ('accuracy','balanced_accuracy','macro_f1','confusion_matrix','per_class','failure_false_positive_rate'):assert recomputed[key]==result['test'][key]
        history=json.loads((directory/'history.json').read_text());assert max(history,key=lambda r:r['val_balanced_accuracy'])['epoch']==result['best_epoch']
        chosen=next(r for r in selection if (r['group'],r['input'],r['head'])==(result['group'],result['input'],result['head']))
        screening=[r for r in results if r['seed']==42 and (r['group'],r['input'],r['head'])==(result['group'],result['input'],result['head'])]
        assert chosen['sigma']==max(screening,key=lambda r:(r['validation']['balanced_accuracy'],-r['sigma']))['sigma']
        is_chosen=result['sigma']==chosen['sigma']
        row={k:result[k] for k in ('group','input','head','sigma','seed','best_epoch','epochs')};row['selected_sigma']=is_chosen
        row['val_balanced_accuracy']=result['validation']['balanced_accuracy']
        row.update({k:result['test'][k] for k in ('accuracy','balanced_accuracy','macro_f1','failure_false_positive_rate')})
        row.update({'failure_'+k:result['test']['per_class']['failure'][k] for k in ('precision','recall','f1')});rows.append(row)
        cm=np.asarray(result['test']['confusion_matrix']);support=cm.sum(1);matrix=cm/support[:,None];assert np.allclose(matrix.sum(1),1)
        normalized.append(dict(**{k:result[k] for k in ('group','input','head','sigma','seed')},support=support.tolist(),class_order=list(CLASSES),matrix=matrix.tolist()))
        if is_chosen:chosen_results.append(result)
        blob=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
        for k,v in norms[result['group']].items():torch.testing.assert_close(blob['state_dict'][k],v,atol=0,rtol=0)
        if is_chosen and result['seed']==42:
            model=MergedProbe(**blob['model_config']);model.load_state_dict(blob['state_dict']);model.eval()
            _,p=evaluate(model,test,result['sigma'],'cpu');np.testing.assert_allclose(p,stored['probabilities'],atol=5e-4,rtol=1e-3);assert np.array_equal(p.argmax(1),stored['probabilities'].argmax(1));cpu_error=float(np.abs(p-stored['probabilities']).max())
            sample=test[0];x={k:torch.from_numpy(sample[k])[None] for k in ('f6','deform') if k in result['input']};cut=max(1,len(sample['labels'])//2)
            with torch.inference_mode():
                logits,_=model(**x);altered={k:v.clone() for k,v in x.items()}
                for v in altered.values():v[:,cut:]+=100
                future,_=model(**altered);torch.testing.assert_close(logits[:,:cut],future[:,:cut],atol=0,rtol=0)
                prefix,_=model(**{k:v[:,:cut] for k,v in x.items()});torch.testing.assert_close(logits[:,:cut],prefix,atol=4e-6,rtol=3e-5)
                state=None;stream=[]
                for i in range(len(sample['labels'])):
                    value,state=model(**{k:v[:,i:i+1] for k,v in x.items()},state=state);stream.append(value)
                torch.testing.assert_close(logits,torch.cat(stream,1),atol=7e-6,rtol=4e-5)
                reset,_=model(**{k:v[:,:1] for k,v in x.items()},state=None);torch.testing.assert_close(logits[:,:1],reset,atol=4e-6,rtol=3e-5)
            checks.append(dict(group=result['group'],input=result['input'],head=result['head'],cpu_replay=True,cpu_gpu_max_probability_error=cpu_error,cpu_gpu_predicted_classes_identical=True,no_future_dependence=True,streaming_equivalence=True,state_reset=True))
    assert len(chosen_results)==60 and len(checks)==12
    csv_write(output/'all_runs.csv',rows);csv_write(output/'sigma_sweep_seed42.csv',[r for r in rows if r['seed']==42]);dump(output/'confusion_normalized.json',normalized)
    summaries=[]
    keys=('accuracy','balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_false_positive_rate')
    for selected in selection:
        rs=[r for r in rows if r['selected_sigma'] and (r['group'],r['input'],r['head'])==(selected['group'],selected['input'],selected['head'])];assert len(rs)==5
        summary={k:selected[k] for k in ('group','input','head','sigma')};summary['seeds']=5
        for k in keys:
            values=[r[k] for r in rs];summary[k+'_mean']=float(np.mean(values));summary[k+'_std']=float(np.std(values,ddof=1))
        summaries.append(summary)
    csv_write(output/'summary.csv',summaries);csv_write(output/'distribution.csv',dm['distribution'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for i,group in enumerate(GROUPS):
        for kind in ('f6','deform','f6_deform'):
            for head in ('mlp','gru'):
                rs=[r for r in rows if r['seed']==42 and (r['group'],r['input'],r['head'])==(group,kind,head)]
                for j,k in enumerate(('val_balanced_accuracy','failure_false_positive_rate')):
                    axes[i,j].plot([r['sigma'] for r in rs],[100*r[k] for r in rs],marker='o',label=kind+' '+head.upper())
        axes[i,0].set_title(group+' | validation BA');axes[i,1].set_title(group+' | test failure FPR (descriptive)')
        for ax in axes[i]:ax.set_xlabel('sigma (valid prediction samples)');ax.set_ylabel('%');ax.set_xticks(SIGMAS);ax.set_ylim(0,100)
    axes[0,1].legend(fontsize=8);fig.tight_layout();fig.savefig(output/'sigma_sweep.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,6,figsize=(19,7));chosen42=[r for r in chosen_results if r['seed']==42]
    for ax,result in zip(axes.ravel(),chosen42):
        cm=np.asarray(result['test']['confusion_matrix']);matrix=cm/cm.sum(1,keepdims=True);ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1)
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{100*matrix[i,j]:.1f}%\n({cm[i,j]})',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.55 else 'black')
        ax.set_xticks(range(3),['Prog','Succ','Fail']);ax.set_yticks(range(3),[f'{c}\nN={n}' for c,n in zip(('Prog','Succ','Fail'),cm.sum(1))]);ax.set_xlabel('Prediction');ax.set_ylabel('GT');ax.set_title(f"{result['input']} {result['head'].upper()} σ={result['sigma']}",fontsize=9)
    fig.suptitle('Seed42 | chosen sigma using validation only | top merged Align, bottom original Align');fig.tight_layout();fig.savefig(output/'confusion_seed42_normalized.png',dpi=160);plt.close(fig)
    curve_records=[];fig,axes=plt.subplots(4,6,figsize=(21,13))
    for gi,group in enumerate(GROUPS):
        test=samples[group]['test'];representatives=[next(i for i,s in enumerate(test) if s['outcome']==outcome) for outcome in (1,2)]
        selected=[r for r in chosen42 if r['group']==group]
        for col,result in enumerate(selected):
            stored=load(ROOT/result['directory']/'test_predictions.npz')
            for oi,index in enumerate(representatives):
                s=test[index];start,end=stored['offsets'][index:index+2];p=stored['probabilities'][start:end];q=one_sided_targets(s['positions'],s['key_index'],s['outcome'],result['sigma']);x=s['positions']-s['key_index'];ax=axes[gi*2+oi,col]
                for k,c in enumerate(CLASSES):ax.plot(x,p[:,k],label=c)
                ax.plot(x,q[:,s['outcome']],color='black',linestyle=':',label='target g');ax.axvline(0,color='gray',linestyle='--');ax.set_ylim(-.03,1.03)
                ax.set_xlabel('valid prediction ticks relative to Key');ax.set_title(f"{group} {'Success' if oi==0 else 'Failure'}\n{result['input']} {result['head'].upper()} σ={result['sigma']}",fontsize=8)
                for j in range(len(p)):curve_records.append(dict(group=group,input=result['input'],head=result['head'],sigma=result['sigma'],interaction_id=s['interaction_id'],rollout_id=s['rollout_id'],current_tick=int(s['ticks'][j]),current_frame=int(s['video_frames'][j]),key_frame=s['key'],relative_position=int(x[j]),hard_gt=int(s['labels'][j]),target_outcome=float(q[j,s['outcome']]),p_in_progress=float(p[j,0]),p_success=float(p[j,1]),p_failure=float(p[j,2])))
    axes[0,0].legend(fontsize=7);fig.suptitle('Deterministic examples: first success and first failure test interaction per group; seed42');fig.tight_layout();fig.savefig(output/'online_probability_curves.png',dpi=150);plt.close(fig);csv_write(output/'online_probability_curves.csv',curve_records)
    dump(output/'verification.json',dict(status='PASS',runs=120,selected_config_runs=60,hard_GT_fixed_across_sigma=True,metrics_recomputed=True,train_only_normalization=True,no_test_based_selection=True,source_hashes_unchanged=True,split_no_leakage=True,causal_checks=checks))
    lines=['# Online tactile outcome detection：one-sided Gaussian 对照','',f'数据：`{source.relative_to(ROOT)}`；训练：`{output.relative_to(ROOT)}`。','',
    '## Sample、Key、GT','',
    '两组都只将Align作为interaction。原始组：event6=failure、event8=success，每个Align独立；start为该Align start，Key为其end，尾段到下一个Align start−1，最后一个到本rollout最后已标注end。合并组：全部Align与中间空隙合并，start取最早Align start、Key取最终Align end、outcome取最终Align类别，尾段到最后已标注end。早期failure在合并组不再是独立failure目标。Insert7/9不构造目标，只可能处于已标注尾段的传感器观察范围。','',
    '每个有效时刻是一条预测，标签只由last frame t与本interaction Key关系决定；不使用window后半段命中规则、不使用对称Key带。固定硬GT：t<Key→in_progress0；t≥Key→success1或failure2。在所有σ下完全相同，包含Key及已标注尾段。不同interaction开始重置标签和GRU状态。','',
    '训练soft target：Key对应首次camera frame≥Key的有效预测tick，d=max(key_index−current_index,0)，g=exp(−d²/(2σ²))；success=[1−g,g,0]，failure=[1−g,0,g]。Key前平滑上升，Key及之后永久为1直到本interaction观察结束；相反outcome概率为0。σ∈{1,2,3,4,6,8}的单位是有效同步预测样本，未进行额外降采样；缺失tick不作为样本，σ不是秒也不是encoder窗口数。soft Gaussian只用于训练，评估不用argmax soft target改写GT。','',
    '## 输入与模型','',
    '每个时刻只用截至当前的16个连续raw F6 ticks→冻结官方checkpoint finger-mode encoder→5×256 flatten1280D，一次编码；Deform仅用同一当前tick五指图→冻结encoder→每指2×2池化512D→flatten2560D。interaction前15个有效tick重新编码并重复interaction首tick左补齐，不读interaction前或未来信息，其余复用严格校验的缓存。checkpoint实际为finger mode，未将hand默认256D强改为finger。','',
    '每模态Linear→128D，Fusion concat→256D。MLP仅使用当前这次F6编码/当前Deform投影→Linear(128或256,128)→ReLU→Dropout(.1)→Linear(128,3)。MLP没有额外16个encoder特征窗口或全interval pooling。单层单向GRU(hidden128)逐时刻接收相同投影，保留同interaction的全部历史状态→Linear(128,3)。批处理完整sequence与在线单步state保留已核对等价，换interaction时重置；不输入Key、最终outcome、位置百分比或总长度。输出softmax为[P(in_progress),P(success),P(failure)]，argmax决定当前类别。','',
    '## Loss、训练和选择','',
    '不加权soft-target cross-entropy：每个有效tick L_t=−Σ_c q_t,c log_softmax(logits_t)_c，对batch所有有效tick取均值；padding不计loss。这里没有inverse-frequency weight、类平衡增强或hard CE，也不是BCE。使用plain soft CE避免class weight进一步改变Gaussian目标概率；长interaction提供更多有效训练tick，本轮不是前一实验每interval16个endpoint等权监督。','',
    '冻结两个encoder，仅更新projection/MLP/GRU/head；标准化仅拟合train。rollout split保持80/17/17，相同rollout的全部interaction始终同split，不按window重分。AdamW lr=.001、wd=.0001、batch8 interaction、clip1、最多30epoch/patience8。每次run按val固定硬GT BA选最佳epoch，平分取最早。','',
    '先seed42对两组×6模型×6σ做72次screen；每组/模型仅按seed42 val BA选择σ，平分选较小σ。再固定所选σ追加seeds43–46，共48次，合计120次；所选12配置每个含5seeds。test只用于最终报告，不选择σ或epoch。σscreen只有一个seed，不能据此宣称σ优劣稳定；repeat阶段也没有重新选择σ。','',
    '## 指标怎么算','',
    '所有有效test预测tick用同一硬GT。ACC=正确数/全部tick；BA=(三类recall之和)/3（Balanced Accuracy，不是binary accuracy）；macroF1=三类F1均值。每类precision=TP/(TP+FP)、recall=TP/(TP+FN)、F1=2PR/(P+R)，零分母记0。failure precision只指预测failure中的正确比例；failure FPR=GT不是failure却预测failure的tick数/全部非failure GT tick数。Key前提前报outcome在固定硬GT下仍算误报，强σ可能训练鼓励提前形成confidence但硬指标下降。','',
    '混淆矩阵行GT、列prediction，顺序Prog/Success/Failure；行归一化=格子计数/该GT类tick总数，对角线为recall。归一化仅改变展示，不改变自然test分布。5seed结果报告均值±样本标准差(ddof=1)，不是独立rollout置信区间。连续tick相关；合并组test仅17rollout，其中3failure，failure指标证据有限。原始组test24Align(15success/9failure)仍来自同17rollout；两组GT定义不同，不能直接把BA差异当作模型变好。','',
    '## 数据分布','', '| 组 | split | interactions | Success/Failure interactions | Prog/Succ/Fail ticks |','|---|---|---:|---|---|']
    for r in dm['distribution']:lines.append(f"| {r['group']} | {r['split']} | {r['interactions']} | {r['success']}/{r['failure']} | {' / '.join(map(str,r['class_ticks']))} |")
    lines+=['','## 所选σ：5 seed test结果','','| 组 | 模型 | σ | BA % | Macro F1 % | Failure precision % | Failure recall % | Failure FPR % |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        cells=[f"{100*r[k+'_mean']:.2f} ± {100*r[k+'_std']:.2f}" for k in ('balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_false_positive_rate')]
        lines.append(f"| {r['group']} | {r['input']} {r['head'].upper()} | {r['sigma']} | "+' | '.join(cells)+' |')
    lines+=['','[全部120次指标](all_runs.csv)；[72次seed42 σ sweep](sigma_sweep_seed42.csv)；[5seed汇总](summary.csv)；[逐run含per-class指标](results.json)；[完整归一化矩阵](confusion_normalized.json)。','','![sigma sweep](sigma_sweep.png)','','![count and normalized confusion](confusion_seed42_normalized.png)','','## Online probability curves','','按数据固定排序，各组选择第一个test success interaction和第一个test failure interaction，展示所选σ的6模型seed42结果，没有按曲线效果挑样本。横轴是相对Key的有效tick，虚线为Key，黑色点线是训练g；[曲线原始数据](online_probability_curves.csv)。曲线可检查提前形成confidence及Key后的稳定性，但两个示例不足以证明所有rollout均稳定。','','![online probability](online_probability_curves.png)','','## 验证','','120次指标从保存概率重算、最佳epoch和σ选择核对、所有checkpoint train-only标准化、原始sensor/encoder/cache哈希不变、rollout split无泄漏；12个所选seed42模型CPU replay、未来特征扰动不影响过去输出、prefix/streaming等价及新interaction重置均通过。CPU/GPU浮点backend概率回放允许atol=5e-4/rtol=1e-3，且要求全部argmax类别完全相同；每个模型实际最大误差已写入verification.json。详见[verification.json](verification.json)。','']
    (output/'README.md').write_text('\n'.join(lines).replace('## Online probability curves','## 当前结果的理解\n\n按所选σ的5seed test BA均值，合并组F6 MLP最高（54.94%±9.32%），原始Align组F6 GRU最高（54.61%±8.47%）；方差较大，不能据此确认模型优劣。Fusion没有显示稳定优势。合并组Deform/Fusion的failure recall仅约0.63%–1.59%，这轮plain soft CE仍没有解决稀少failure的识别；低FPR也可能伴随漏报，不能单独视为改进。原始组Deform GRU failure recall均值41.77%，同时FPR12.44%、precision15.74%，仍有明显误报。\n\n概率曲线中的固定failure示例存在持续低failure confidence或错判success，尚不能声称Gaussian已使Key前形成稳定且正确的outcome confidence。训练的g单调不意味着预测概率必须单调。本轮没有加入Gaussian以外的标签增强，也没有重跑相同GT下σ=0的hard-target baseline，因此不能把与历史不同GT实验的分数差异归因于Gaussian本身。\n\n'+'## Online probability curves'))
    (source/'README.md').write_text('\n'.join(lines[:lines.index('## 所选σ：5 seed test结果')])+f'\n\n训练结果见 `{output.relative_to(ROOT)}/README.md`。\n')
    target=ROOT/'WeeklySummary/10.5/gaussian_online';target.mkdir(parents=True,exist_ok=False);copies=[]
    files=[p for p in output.iterdir() if p.is_file()]+list((output/'runs').rglob('metrics.json'))
    for p in files:
        destination=target/p.relative_to(output);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,destination);copies.append(dict(source=str(p.relative_to(ROOT)),destination=str(destination.relative_to(ROOT)),sha256=sha(p)))
    dump(target/'copy_manifest.json',copies);assert all(sha(ROOT/r['destination'])==r['sha256']==sha(ROOT/r['source']) for r in copies)
    weekly=ROOT/'WeeklySummary/10.5/10.5.md'
    with weekly.open('a') as f:
        f.write('\n\n## 11. Align online one-sided Gaussian：固定硬GT与σ搜索\n\n'+ '\n'.join(lines[4:lines.index('## 数据分布')])+ '\n\n'+ '\n'.join(lines[lines.index('## 数据分布'):lines.index('## Online probability curves')]).replace('](all_runs.csv)','](gaussian_online/all_runs.csv)').replace('](sigma_sweep_seed42.csv)','](gaussian_online/sigma_sweep_seed42.csv)').replace('](summary.csv)','](gaussian_online/summary.csv)').replace('](results.json)','](gaussian_online/results.json)').replace('](confusion_normalized.json)','](gaussian_online/confusion_normalized.json)').replace('](sigma_sweep.png)','](gaussian_online/sigma_sweep.png)').replace('](confusion_seed42_normalized.png)','](gaussian_online/confusion_seed42_normalized.png)')+'\n\n[完整实验README、曲线与验证](gaussian_online/README.md)。\n\n![Online probability curves](gaussian_online/online_probability_curves.png)\n')
    snapshot=output/'code_snapshot';shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    print('REPORT_COMPLETE',json.dumps(summaries),flush=True)

if __name__=='__main__':main()
