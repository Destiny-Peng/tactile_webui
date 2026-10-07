"""Add frame event3-only positive experiment using the previous frozen features."""
from .common import project_path, relative_path
import argparse
import datetime
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from .common import ROOT,dump,sha
from .early_warning import metrics,write_csv
from .failure_relabel import KINDS,SEEDS,load_samples,train


def prepare(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    source=args.source
    for name in ('source_intervals.json','split_manifest.json','annotation_snapshot.json'):
        shutil.copyfile(source/name,out/name)
    shutil.copytree(source/'original_annotations_backup',out/'original_annotations_backup')
    manifest=json.loads((source/'dataset_manifest.json').read_text())
    manifest['feature_source']=str(relative_path(source))
    manifest['records']=[r for r in manifest['records'] if r['group']=='frame']
    dump(out/'dataset_manifest.json',manifest)
    protocol=json.loads((source/'protocol.json').read_text())
    protocol.update(event_mapping={'3':'inside event3 -> positive1','4':'negative0 outside event3','outside':'negative0'},
        groups={'frame_3':'current last camera frame inside any event3 closed interval=1; all other frames=0, including event4'},
        source=str(relative_path(source)),source_manifest_sha256=sha(source/'dataset_manifest.json'),
        source_split_sha256=sha(source/'split_manifest.json'),
        sampling='same full valid failure rollouts,dense step0,no augmentation; label3 interval union only',
        loss='train-count balanced BCEWithLogits N/(2Nc),counting valid synchronized frames',
        normalization='same full training frames as previous frame groups;train-only mean/std floor.01',
        interval_context='not applicable: frame classification; GRU retains rollout history',
        created_at=datetime.datetime.now().astimezone().isoformat())
    dump(out/'protocol.json',protocol)
    samples=load_samples(out,'frame_3');distribution=[]
    for split,ss in samples.items():
        y=np.concatenate([s['y'] for s in ss]);n0=int((y==0).sum());n1=int(y.sum());assert n0 and n1
        distribution.append(dict(group='frame_3',split=split,rollouts=len(ss),frames=len(y),negative=n0,positive=n1,
            positive_fraction=n1/len(y),negative_weight=len(y)/(2*n0),positive_weight=len(y)/(2*n1)))
    write_csv(out/'prepared_distribution.csv',distribution)
    print('PREPARED',distribution,flush=True)


def report(args):
    out=args.output;source=args.source;results=json.loads((out/'results.json').read_text())
    assert len(results)==15
    previous=json.loads((source/'results.json').read_text())
    comparisons=[];normalized=[]
    for group in ('frame_3','frame_4','frame_3or4'):
        for kind in KINDS:
            rr=[r for r in results+previous if r['group']==group and r['input']==kind]
            assert {r['seed'] for r in rr}==set(SEEDS) and len(rr)==5
            row=dict(group=group,input=kind,seeds=5)
            for key in ('accuracy','balanced_accuracy','macro_f1','precision','recall','fpr','pr_auc','roc_auc'):
                v=[r['val'][key] for r in rr]
                row[key+'_mean']=float(np.mean(v));row[key+'_std']=float(np.std(v,ddof=1))
            comparisons.append(row)
            if group=='frame_3':
                cm=np.mean([r['val']['confusion_matrix'] for r in rr],axis=0);nr=cm/cm.sum(1,keepdims=True)
                for gt in (0,1):
                    for pred in (0,1):normalized.append(dict(input=kind,gt=gt,prediction=pred,mean_count=cm[gt,pred],row_normalized=nr[gt,pred]))
                for r in rr:
                    with np.load(out/'runs'/group/kind/f"seed_{r['seed']}"/'val_predictions.npz') as z:
                        replay=metrics(z['labels'],z['probabilities'])
                    for key in ('balanced_accuracy','macro_f1','pr_auc','roc_auc'):
                        assert np.isclose(replay[key],r['val'][key])
                    assert replay['confusion_matrix']==r['val']['confusion_matrix']
    write_csv(out/'summary.csv',[r for r in comparisons if r['group']=='frame_3'])
    write_csv(out/'frame_comparison.csv',comparisons);write_csv(out/'confusion_normalized.csv',normalized)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.8))
    for j,kind in enumerate(KINDS):
        rr=[r for r in results if r['input']==kind];cm=np.mean([r['val']['confusion_matrix'] for r in rr],axis=0)
        v=cm/cm.sum(1,keepdims=True);ax=axes[j];ax.imshow(v,vmin=0,vmax=1,cmap='Blues')
        for a in (0,1):
            for b in (0,1):ax.text(b,a,f'{100*v[a,b]:.1f}%\n({cm[a,b]:.1f})',ha='center',va='center',color='white' if v[a,b]>.5 else 'black')
        ax.set_xticks([0,1],['outside3','event3']);ax.set_yticks([0,1],['outside3','event3'])
        ax.set_xlabel('Prediction');ax.set_ylabel('GT');ax.set_title(kind)
    fig.tight_layout();fig.savefig(out/'confusion_normalized.png',dpi=180);fig.savefig(out/'confusion_normalized.pdf');plt.close(fig)
    samples=load_samples(out,'frame_3')['val'];fig,axes=plt.subplots(len(samples),3,figsize=(15,3*len(samples)),squeeze=False)
    curves=[];negative_regions=[]
    annotation_events=json.loads((out/'source_intervals.json').read_text())
    for j,kind in enumerate(KINDS):
        ps=[]
        for seed in SEEDS:
            with np.load(out/'runs'/'frame_3'/kind/f'seed_{seed}'/'val_predictions.npz') as z:
                ps.append(z['probabilities'].copy());offsets=z['offsets'].copy()
        ps=np.stack(ps)
        for i,s in enumerate(samples):
            a,b=offsets[i:i+2];local=ps[:,a:b];mean=local.mean(0);sd=local.std(0,ddof=1)
            in4=np.zeros(len(s['frames']),bool)
            for e in annotation_events:
                if e['rollout_id']==s['rollout_id'] and e['event_key']==4:
                    in4|=(s['frames']>=e['start_frame'])&(s['frames']<=e['end_frame'])
            for region,mask in (('event4_outside3',in4&(s['y']==0)),('background_outside3and4',~in4&(s['y']==0))):
                for si,seed in enumerate(SEEDS):
                    negative_regions.append(dict(input=kind,seed=seed,rollout_id=s['rollout_id'],region=region,
                        negative_frames=int(mask.sum()),false_positive_frames=int(((local[si]>=.5)&mask).sum())))
            ax=axes[i,j];x=s['frames']/30
            ax.plot(x,mean,label='mean P(in event3)');ax.fill_between(x,np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),alpha=.15)
            ax.step(x,s['y'],where='post',color='black',alpha=.5,label='GT');ax.axhline(.5,color='gray',ls='--')
            ax.set_ylim(-.03,1.03);ax.set_xlabel('Recorded camera time (s)')
            ax.set_title(s['rollout_id'].split('mixedfail_')[-1]+' / '+kind);ax.legend(fontsize=8)
            for k,frame in enumerate(s['frames']):
                curves.append(dict(input=kind,rollout_id=s['rollout_id'],camera_frame=int(frame),label=int(s['y'][k]),
                    mean_probability=float(mean[k]),seed_variance=float(sd[k]**2)))
    fig.suptitle('Event3-only frame validation; mean ± seed SD');fig.tight_layout()
    fig.savefig(out/'frame_3_val_curves.png',dpi=180);fig.savefig(out/'frame_3_val_curves.pdf');plt.close(fig)
    write_csv(out/'val_probability_curves.csv',curves)
    write_csv(out/'negative_region_counts.csv',negative_regions)
    negative_summary=[]
    for kind in KINDS:
        for region in ('event4_outside3','background_outside3and4'):
            rates=[]
            for seed in SEEDS:
                rows=[r for r in negative_regions if r['input']==kind and r['seed']==seed and r['region']==region]
                den=sum(r['negative_frames'] for r in rows);num=sum(r['false_positive_frames'] for r in rows)
                assert den>0;rates.append(num/den)
            negative_summary.append(dict(input=kind,region=region,negative_frames=den,
                fpr_mean=float(np.mean(rates)),fpr_std=float(np.std(rates,ddof=1))))
    write_csv(out/'negative_region_fpr.csv',negative_summary)
    import csv
    dist=list(csv.DictReader((out/'prepared_distribution.csv').open()))
    lines=['# Failure-only补充：仅3区间为正例','',
        '根据用户确认，本轮是frame二分类：仅标注3的`[causal_onset_frame, observable_onset_frame]`闭区间内为1，其余全部为0，包含4区间；同一rollout的多个3区间取并集。若3/4区间重叠，属于3就为1。不构造interval-level单类别分类。','',
        '复用上轮23条failure rollout、原标注快照和冻结特征。19 train/4 val，无test，split与原来的frame only4、frame3or4完全一致。未重新读取或覆盖正在使用的原始标注；上轮的33个JSON备份也复制到本轮original_annotations_backup/。','',
        '每时刻F6连续past16rawticks→冻结现有T-Rex finger encoder1280D；当前Deform→冻结每指encoder→2×2pool后512D×5=2560D。各Linear128，Fusion concat256，单层单向GRU128→Linear1→sigmoid输出P(当前帧在3区间内)。GRU在rollout开始reset，使用截至当前的历史；全段有效同步tick参与，失效tick省略。与原frame实验相同，没有Gaussian/Key扩展/augmentation。','',
        'Train-count balanced BCEWithLogits，w0=N/(2N0)、w1=N/(2N1)，按有效帧数量计数。相同train-only mean/std，std下限0.01；AdamW lr0.001/wd0.0001，batch8，max30epoch，patience8，clip1。seeds42–46，三模态共新增15次训练。固定阈值0.5，best epoch按val BA选择，平分取最早；每帧输出一个binary prediction。','',
        'BA是两类recall均值，Macro F1是两类F1均值；P/R/FPR针对positive=1，即3区间内。PR-AUC采用average precision，ROC-AUC采用ROC梯形积分；表中为5seeds均值±样本SD。当前数据用val选epoch并报告该val，没有独立test。','',
        '## 数据分布','', '| Split | Rollouts | Negative | Positive | Positive fraction |','|---|---:|---:|---:|---:|']
    for r in dist:lines.append(f"| {r['split']} | {r['rollouts']} | {r['negative']} | {r['positive']} | {100*float(r['positive_fraction']):.2f}% |")
    lines+=['','Val正例仅3.24%；全部预测为0的Accuracy也有96.76%，但BA只有50%。所以以BA、Macro F1及positive P/R/FPR为主，不能只看Accuracy。']
    lines+=['','## Validation（%，mean ± SD，5seeds）','', '| Input | Accuracy | BA | Macro F1 | Precision | Recall | FPR | PR-AUC | ROC-AUC |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        if r['group']!='frame_3':continue
        values=[f"{100*r[k+'_mean']:.2f} ± {100*r[k+'_std']:.2f}" for k in ('accuracy','balanced_accuracy','macro_f1','precision','recall','fpr','pr_auc','roc_auc')]
        lines.append('| '+r['input']+' | '+' | '.join(values)+' |')
    lines+=['','## 负例分区误报率','',
        '把负例分为4区间内（剔除与3重叠部分）及3/4都不属于的背景；FPR均为对应区域内预测1的帧比例，先逐seed计算，再求mean±SD。','',
        '| Input | Negative region | Frames | FPR % |','|---|---|---:|---:|']
    for r in negative_summary:
        lines.append(f"| {r['input']} | {r['region']} | {r['negative_frames']} | {100*r['fpr_mean']:.2f} ± {100*r['fpr_std']:.2f} |")
    lines+=['','## 与既有两组frame实验比较','', '| Positive label | Input | BA | Macro F1 | PR-AUC |','|---|---|---:|---:|---:|']
    for r in comparisons:
        values=[f"{100*r[k+'_mean']:.2f} ± {100*r[k+'_std']:.2f}" for k in ('balanced_accuracy','macro_f1','pr_auc')]
        lines.append('| '+r['group']+' | '+r['input']+' | '+' | '.join(values)+' |')
    best=max((r for r in comparisons if r['group']=='frame_3'),key=lambda r:r['balanced_accuracy_mean'])
    lines+=['',f"仅3的平均BA以{best['input']}最高：{100*best['balanced_accuracy_mean']:.2f}±{100*best['balanced_accuracy_std']:.2f}%。3正例较少；三种positive定义改变了GT及正例比例，跨组AP不能单独用于评价representation优劣。Val只有4条独立rollout，5seeds只测训练波动。",'',
        f"固定0.5阈值下，{best['input']}的precision仅{100*best['precision_mean']:.2f}%，recall为{100*best['recall_mean']:.2f}%，整体negative FPR为{100*best['fpr_mean']:.2f}%。高BA主要来自高召回，区间定位及误报仍有明显问题；需结合上述4区间和背景的FPR分解解读，不能据BA单独认定3/4在线可分。",'',
        '![Normalized confusion](confusion_normalized.png)','',
        '矩阵行是GT，列是prediction，按GT行归一化；括号为5seeds平均原始计数。','',
        '![Validation curves](frame_3_val_curves.png)','',
        '曲线横轴为原始相机时间；彩线/阴影为5seeds概率mean±SD，黑线为hard GT。概率、GT、offsets和sample映射保存在每个run下，val_probability_curves.csv可复现图；原始15次指标见results.json，9组frame汇总见frame_comparison.csv。','']
    (out/'README.md').write_text('\n'.join(lines))
    assert sha(source/'split_manifest.json')==json.loads((out/'protocol.json').read_text())['source_split_sha256']
    dump(out/'verification.json',dict(status='PASS',new_runs=15,seeds=list(SEEDS),same_rollout_split=True,
        no_test=True,encoders_frozen_cache_only=True,GT_event3_union=True,val_metrics_replayed=True,
        training_shutdown=args.training_shutdown))
    shutil.copyfile(Path(__file__),out/'failure_relabel_three_only.py')
    shutil.copyfile(Path(__file__).with_name('failure_relabel.py'),out/'failure_relabel.py')
    dump(out/'completion.json',dict(status='complete',completed_at=datetime.datetime.now().astimezone().isoformat(),
         code_sha256=sha(Path(__file__)),training_code_sha256=sha(Path(__file__).with_name('failure_relabel.py'))))
    weekly=project_path('WeeklySummary/10.5');destination=weekly/'failure_relabel'/out.name
    destination.mkdir(parents=True,exist_ok=False)
    names=['README.md','protocol.json','source_intervals.json','split_manifest.json','annotation_snapshot.json',
        'dataset_manifest.json','prepared_distribution.csv','label_distribution.csv','results.json','summary.csv',
        'frame_comparison.csv','confusion_normalized.csv','confusion_normalized.png','confusion_normalized.pdf',
        'frame_3_val_curves.png','frame_3_val_curves.pdf','val_probability_curves.csv','negative_region_counts.csv',
        'negative_region_fpr.csv','verification.json','completion.json']
    checks={}
    for name in names:
        shutil.copyfile(out/name,destination/name);assert sha(out/name)==sha(destination/name);checks[name]=sha(out/name)
    dump(destination/'copy_sha256.json',checks)
    rel='failure_relabel/'+out.name+'/README.md'
    section=['','## 补充：frame仅3区间为正例','',
        '新增15次（F6/Deform/Fusion×seeds42–46），仅3区间为1，其余0包含4；复用原19/4 rollout划分与冻结特征，无test。原45次结果保持不变；总共60次。','',
        '| Input | Val BA % | Val Macro F1 % | Val PR-AUC % |','|---|---:|---:|---:|']
    for r in comparisons:
        if r['group']=='frame_3':section.append('| '+r['input']+' | '+' | '.join(f"{100*r[k+'_mean']:.2f} ± {100*r[k+'_std']:.2f}" for k in ('balanced_accuracy','macro_f1','pr_auc'))+' |')
    plain='\n'.join(section)+'\n\n'
    for path,link in ((source/'README.md','../'+out.name+'/README.md'),
        (weekly/'failure_relabel'/source.name/'README.md','../'+out.name+'/README.md'),
        (weekly/'10.5failure_relabel.md',rel)):
        with path.open('a') as f:f.write(plain+f'[仅3的完整报告、归一化矩阵和概率曲线]({link})。\n')
    oldcopy=weekly/'failure_relabel'/source.name
    oldchecks=json.loads((oldcopy/'copy_sha256.json').read_text());oldchecks['README.md']=sha(source/'README.md')
    assert sha(source/'README.md')==sha(oldcopy/'README.md');dump(oldcopy/'copy_sha256.json',oldchecks)
    with (weekly/'10.5.md').open('a') as f:f.write('\n\n实验19补充：新增frame仅3区间为正例，15次、5seeds，与既有frame4及frame3or4共9组模态对照。[完整结果]('+rel+')。\n')
    now=datetime.datetime.now().astimezone().isoformat();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    commands='source ./project_env.sh\n'+'\n'.join(
        f'PYTHONPATH="$PROJECT_ROOT/tools" tools/run_trex.sh python -u -m sharpa_tactile.failure_relabel_three_only '
        f'{action} --source {relative_path(source)} --output {relative_path(out)} --device {args.device}'
        for action in ('prepare','train','report'))
    env=project_path('environment_reports', 'SHARPA_FAILURE_RELABEL_FRAME3_'+out.name.split('_frame3')[0]+'.md')
    env.write_text(f'# Event3-only frame experiment\n\nCOMPLETE at {now}. HEAD `{commit}`.\n\n15 frozen-feature GRU runs on {args.device};same23rollouts19train4val,no test;class3 union inside1,allelse0;valBA epochselection,5seeds42–46. No encoder extraction/retraining. Validation metrics replay PASS;weekly copySHA PASS.\n\nEnvironment: project-local ProcVLM venv via tools/run_trex.sh. Run prepare/train/report sequentially:\n\n```bash\n{commands}\n```\n\nReport: `{relative_path(destination)}/README.md`.\n')
    if args.training_shutdown=='terminated_after_complete':
        with env.open('a') as f:f.write('\nOperational note: GPU training reached TRAIN_COMPLETE15 and saved all15 metrics/checkpoints/predictions/results.json. The worker remained in disk-I/O exit cleanup; after confirming15 completed records and15 checkpoints, SIGTERM was used to release that completed worker. The supervising shell returned1 due to termination; no training run was aborted. Report was then run separately with --training-shutdown terminated_after_complete; all saved validation metrics were replayed successfully.\n')
    for name in ('SETUP_STATUS.md','SYSTEM_INFO.txt'):
        with (project_path(name)).open('a') as f:f.write(f'\nFailure-only frame3 COMPLETE ({now}):15 frozen-feature{args.device} GRU runs,3modalities×5seeds;positiveevent3only,frame4negative;original19/4rollout split,no test. Valmetrics andweeklycopySHA PASS. Report:{relative_path(env)}. HEAD:{commit}.\n')
    print('REPORT_COMPLETE',len(results),'new runs',len(names),'copied files',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','train','report'))
    parser.add_argument('--source',type=Path,default=project_path('outputs/sharpa_failure_relabel/20261006_165000'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--training-shutdown',choices=('normal','terminated_after_complete'),default='normal')
    args=parser.parse_args();args.source=args.source.resolve();args.output=args.output.resolve();args.groups=('frame_3',)
    globals()[args.action](args)


if __name__=='__main__':main()
