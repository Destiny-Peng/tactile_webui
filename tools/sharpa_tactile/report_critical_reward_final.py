"""Report asymmetric final-Align critical reward without repeating model inference."""
from .common import project_path, relative_path
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from .common import ROOT,dump,sha


def write_csv(path,rows):
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def labels(s,n,m):
    return ((s['frames']>=s['key']-n)&(s['frames']<=s['key']+m)&(s['outcome']==2)).astype(int)


def metrics(y,p):
    pred=(p>=.5).astype(int);cm=np.array([[int(((y==c)&(pred==k)).sum()) for k in (0,1)] for c in (0,1)]);recall=np.diag(cm)/cm.sum(1);precision=np.divide(np.diag(cm),cm.sum(0),out=np.zeros(2),where=cm.sum(0)!=0);f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(2),where=precision+recall!=0)
    return cm,float(recall.mean()),float(f1.mean())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();out=a.output.resolve();proto=json.loads((out/'protocol.json').read_text());source=project_path(proto['source']);dm=json.loads((source/'dataset_manifest.json').read_text());results=json.loads((out/'results.json').read_text());summary=list(csv.DictReader((out/'summary.csv').open()));selected=json.loads((out/'selected_config.json').read_text());n,m=selected['n'],selected['m'];samples={s:[] for s in ('train','val','test')}
    for row in dm['records']:
        if row['group']!='merged_align':continue
        with np.load(source/row['feature_path']) as z:samples[row['split']].append(dict(row,frames=z['video_frames'].copy()))
    assert sum(len(v) for v in samples.values())==114;distributions=[]
    for nn in proto['ns']:
        for mm in proto['ms']:
            for split,ss in samples.items():
                y=np.concatenate([labels(s,nn,mm) for s in ss]);distributions.append(dict(n=nn,m=mm,split=split,rollouts=len(ss),final_success=sum(s['outcome']==1 for s in ss),final_failure=sum(s['outcome']==2 for s in ss),positive=int(y.sum()),negative=int((y==0).sum()),positive_fraction=float(y.mean())))
    write_csv(out/'label_distribution.csv',distributions);assert len(results)==175
    for r in results:
        with np.load(out/'runs'/f"n{r['n']}_m{r['m']}"/f"seed_{r['seed']}"/'test_predictions.npz') as z:
            y=np.concatenate([labels(s,r['n'],r['m']) for s in samples['test']]);assert np.array_equal(y,z['labels']);cm,ba,f1=metrics(y,z['probabilities']);assert cm.tolist()==r['test']['confusion_matrix'] and np.isclose(ba,r['test']['balanced_accuracy']) and np.isclose(f1,r['test']['macro_f1'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup={(int(r['n']),int(r['m'])):r for r in summary};fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,key,title in zip(axes,('validation_balanced_accuracy_mean','test_balanced_accuracy_mean','test_event_balanced_accuracy_mean'),('Val pulse BA (selection)','Test pulse BA (different GT bands)','Test event BA (fixed rollout GT)')):
        values=np.array([[float(lookup[(nn,mm)][key]) for mm in proto['ms']] for nn in proto['ns']]);im=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',aspect='auto')
        for i in range(len(proto['ns'])):
            for j in range(len(proto['ms'])):ax.text(j,i,f'{100*values[i,j]:.1f}',ha='center',va='center',fontsize=9,color='white' if values[i,j]<.65 else 'black')
        ax.set_xticks(range(len(proto['ms'])),proto['ms']);ax.set_yticks(range(len(proto['ns'])),proto['ns']);ax.set_xlabel('m after end (camera frames)');ax.set_ylabel('n before end');ax.set_title(title);ax.scatter(proto['ms'].index(m),proto['ns'].index(n),marker='s',s=180,facecolors='none',edgecolors='red')
    fig.subplots_adjust(wspace=.25,top=.87,right=.90);fig.colorbar(im,cax=fig.add_axes([.94,.15,.012,.70]));fig.suptitle('Final Align failure Key only |5seeds mean |selected n='+str(n)+' m='+str(m));fig.savefig(out/'n_m_sweep.png',dpi=180);fig.savefig(out/'n_m_sweep.pdf');plt.close(fig)
    curves=[];event_rows=[];fig,axes=plt.subplots(2,2,figsize=(13,8))
    for si,split in enumerate(('val','test')):
        ss=samples[split];ps=[]
        for seed in range(42,47):
            with np.load(out/'runs'/f'n{n}_m{m}'/f'seed_{seed}'/(split+'_predictions.npz')) as z:ps.append(z['probabilities'].copy());offsets=z['offsets'].copy()
        meanp=np.stack(ps).mean(0);lo=min(int(s['frames'].min()-s['key']) for s in ss);hi=max(int(s['frames'].max()-s['key']) for s in ss);grid=np.arange(lo,hi+1)
        for i,s in enumerate(ss):
            a,b=offsets[i:i+2]
            for seed,p in zip(range(42,47),ps):
                local=p[a:b];hit=np.flatnonzero(local>=.5);index=int(hit[0]) if len(hit) else None;event_rows.append(dict(split=split,rollout_id=s['rollout_id'],seed=seed,final_failure=int(s['outcome']==2),detected_failure=int(index is not None),peak_score=float(local.max()),first_detection_relative_seconds=float((s['frames'][index]-s['key'])/30) if index is not None else None))
        for oi,outcome in enumerate((1,2)):
            values=[]
            for i,s in enumerate(ss):
                if s['outcome']!=outcome:continue
                a,b=offsets[i:i+2];relative=s['frames']-s['key'];v=np.full(len(grid),np.nan)
                for f in np.unique(relative):v[int(f-lo)]=meanp[a:b][relative==f].mean()
                values.append(v)
            values=np.stack(values);mean=np.full(len(grid),np.nan);sd=mean.copy();count=np.isfinite(values).sum(0)
            for j,c in enumerate(count):
                local=values[:,j];local=local[np.isfinite(local)]
                if c:mean[j]=local.mean()
                if c>1:sd[j]=local.std(ddof=1)
                curves.append(dict(split=split,final_outcome='success' if outcome==1 else 'failure',relative_seconds=float(grid[j]/30),mean=float(mean[j]),variance=float(sd[j]**2),std=float(sd[j]),rollout_count=int(c)))
            ax=axes[si,oi];ax.plot(grid/30,mean,color='tab:red',label='P(critical failure)');ax.fill_between(grid/30,np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),color='tab:red',alpha=.15);ax.axhline(.5,color='gray',linestyle='--');ax.axvline(0,color='black',linestyle=':')
            if outcome==2:ax.axvspan(-n/30,m/30,color='tab:blue',alpha=.08,label='positive Key band')
            ax.set_xlim(lo/30,hi/30);ax.set_ylim(0,1);ax.set_xlabel('Time relative to FINAL Align end (s)');ax.set_ylabel('Mean critical score ± rollout SD');ax.set_title(split.upper()+' | final '+('success' if outcome==1 else 'failure')+f' | N={len(values)}');ax.legend(fontsize=8)
    fig.suptitle(f'Critical reward | selected n={n},m={m} | 5seed mean within rollout');fig.tight_layout();fig.savefig(out/'key_relative_critical_score.png',dpi=180);fig.savefig(out/'key_relative_critical_score.pdf');plt.close(fig);write_csv(out/'key_relative_mean_variance.csv',curves);write_csv(out/'selected_event_predictions.csv',event_rows)
    dump(out/'verification.json',dict(status='PASS',final_align_key_only=True,success_and_earlier_keys_ignored=True,rollouts114=True,runs175=True,seeds=list(range(42,47)),metrics_recomputed=True,labels_checked=True,original_annotations_unchanged=True))
    lines=['# Final Align Key：binary critical reward','', '每个rollout只保留最后一个Align interval的end作为潜在Key；只在其annotation2=failure时创建正Key区域。更早的Align stage、全部Insert stage以及最终success Key不参与标记正例。如果早期interval的帧落在最后failure Key band里，可由最后band标1，但不创建早期Key。每个最终success rollout整段label0；每个最终failure rollout只有[end−n,end+m]内的当前last frame为1，其余0，包括之前的失败。端点闭区间，n/m单位相机帧(30fps)，上限30/15即前1秒/后.5秒。没有Gaussian、soft label、reward衰减或window后半段命中。','',
    '忽略早期interval指不使用其Key和标签，不删除其传感器history。输入仍为最早任意标注start至最后任意标注end的原范围，GRU每rollout开始重置，在中途Align边界不重置。复用已验证的连续rollout frozen特征缓存，缓存历史名merged_align只描述一条完整历史，不改变“仅最后Key”的新标签。Key band在原观察范围截断，不读取范围外或未来输入。','',
    '114rollout，92最终success/22最终failure；原split保持train80(64/16)、val17(14/3)、test17(14/3)。11个无目标标注不使用。成功数据作为负样本保留；不是删掉success rollout。最后一个指最后Align interval，因为本任务持续排除Insert目标。','',
    '例如：早期failure end=120、最终success end=200，则整条rollout均为0。若两次failure end=120和220，n=10,m=5，则只标[210,225]为1，早期[110,125]不生成正例。若最终failure end=220、观察范围截至230，n=30,m=15，则理论[190,235]截断为[190,230]；不会补读未来数据。','',
    '## 模型和Loss','', 'F6 past16rawticks→冻结现有finger-mode T-Rex encoder1280D→dense每时刻Linear128→单层单向GRU128→Linear1。训练输出logit，BCEWithLogits；推理sigmoid为critical score。阈值固定.5，不调阈值。不训练RL policy，这里critical reward是监督式Key二分类。只训练projection、GRU、head；训练集拟合standardization。','',
    'plain unweighted BCE，全部有效tick平均；无classweight、平衡重采样、step augmentation、Deform或Gaussianσ。沿用AdamW lr.001/wd.0001、batch8 rollout、max30epoch/patience8、clip1。训练单元从原始各Align序列改为完整rollout历史，是用户最后Key规则下的构造变化。','',
    '## Sweep、GT和指标','', f'n={proto["ns"]}，m={proto["ms"]}，共35配置×seeds42–46=175runs。每次最佳epoch仅按该Key band的val binary BA选，平分取首次。n/m按5seed平均val BA选，平分按macroF1、较小n+m/n/m。仅扫描此网格，未扫描496整数配置。', '',
    '不同n/m的frame GT与正例数量不同，因此不同pulse BA不能单独证明更好的故障检测。label_distribution.csv逐组列出train/val/test分布。Binary BA=(noncritical recall+critical recall)/2，macroF1平均两类F1；critical precision/recall/FPR均相对该band。常数0的BA=50%，可能高ACC但不识别failure Key。','',
    '同时提供固定rollout outcome指标：每rollout任一有效tick的critical score≥.5视为detected failure；最终failure rollout=1，最终success=0，GT不随n/m变化。event precision/recall/FPR以17test rollouts(3failure/14success)计算，而不是Key tick；没有用event test指标选配置。mean±SD为5seeds样本标准差，不是独立rollout置信区间。','',
    f'## Validation选中：n={n},m={m}','', '| 指标 | Val均值±SD % | Test均值±SD % |','|---|---:|---:|']
    for level,title in (('','Key tick'),('_event','Fixed rollout event')):
        for key,name in (('balanced_accuracy','BA'),('macro_f1','Macro F1'),('precision','Critical/failure precision'),('recall','Critical/failure recall'),('false_positive_rate','Critical/failure FPR')):
            cells=[f"{100*selected[phase+level+'_'+key+'_mean']:.2f} ± {100*selected[phase+level+'_'+key+'_std']:.2f}" for phase in ('validation','test')];lines.append('| '+title+' '+name+' | '+' | '.join(cells)+' |')
    lines+=['',f'Test band正例{selected["test_positive"]}、负例{selected["test_negative"]}。不是把整个failure interval标为1；Key后超过m恢复0。','','## 全35组结果','','| n | m | Test positive ticks | Val pulse BA % | Test pulse BA % | Test fixed-event BA % |','|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append('| '+r['n']+' | '+r['m']+' | '+r['test_positive']+' | '+' | '.join(f"{100*float(r[k+'_mean']):.2f} ± {100*float(r[k+'_std']):.2f}" for k in ('validation_balanced_accuracy','test_balanced_accuracy','test_event_balanced_accuracy'))+' |')
    lines+=['','![Asymmetric Key sweep](n_m_sweep.png)','','## Key-relative critical score','','所有val/test rollout按最终Align end对齐，按最终outcome分组；先对每rollout平均5seeds，再对当时有数据的rollout平均，阴影为rollout SD，不是seed SD。重复camera frame先平均，缺失位置不补值；N及variance在CSV。蓝色区域仅failure图表示Key正例带，success图即使Key附近也全为0。边缘N小，不能把边缘曲线当作全体效果。','','![Critical score](key_relative_critical_score.png)','','[完整175runs](results.json) · [35组全部指标](summary.csv) · [正负分布](label_distribution.csv) · [选中配置](selected_config.json) · [event逐rollout预测](selected_event_predictions.csv) · [曲线/N/variance](key_relative_mean_variance.csv) · [验证](verification.json)','']
    if proto.get('stopping_metric')=='loss':
        lines=[line.replace('max30epoch/patience8',f"max{proto['epochs']}epoch/patience{proto['patience']}").replace('每次最佳epoch仅按该Key band的val binary BA选，平分取首次。', '每次最佳epoch仅按该Key band的最低validation unweighted BCE选，平分取首次；patience50监控同一loss。') for line in lines]
        lines+=['','本轮按用户要求重跑max300/patience50，保存并回载最低validation BCE，而非最佳BA；n/m网格仍按平均val BA排名，因各配置GT不同，不横向按BCE选择Key宽度。原实验结果保留。','']
    (out/'README.md').write_text('\n'.join(lines));snapshot=out/'code_snapshot';snapshot.mkdir()
    for name in ('critical_reward_final_fast.py','report_critical_reward_final.py'):shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    target=project_path('WeeklySummary/10.5/critical_reward_final', out.name);target.mkdir(parents=True,exist_ok=False);copies=[]
    for p in out.iterdir():
        if p.is_file():q=target/p.name;shutil.copy2(p,q);copies.append(dict(source=str(relative_path(p)),destination=str(relative_path(q)),sha256=sha(p)))
    dump(target/'copy_manifest.json',copies)
    with (project_path('WeeklySummary/10.5/10.5.md')).open('a') as f:f.write('\n\n## 16. 最后Align failure Key：critical reward\n\n'+ '\n'.join(lines[2:lines.index('## 全35组结果')])+f'\n\n[全35配置×5seed报告和概率曲线](critical_reward_final/{out.name}/README.md)。\n')
    print('REPORT_COMPLETE',json.dumps(selected),flush=True)

if __name__=='__main__':main()
