"""Key-aligned class-conditional means across ALL validation/test interactions."""
import argparse
import csv
import datetime
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,dump,sha
from .gaussian_online import GROUPS,read_samples,evaluate
from .merged_online import MergedProbe,load,save


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def align_interactions(samples,seed_probabilities,grid):
    """One vote/interaction/frame after seed and duplicate-tick averaging; no padding."""
    offsets=np.r_[0,np.cumsum([len(s['labels']) for s in samples])]
    stacked=np.stack(seed_probabilities);assert stacked.shape==(5,offsets[-1],3)
    assert np.isfinite(stacked).all() and np.allclose(stacked.sum(-1),1,atol=1e-5)
    tables=np.full((5,len(samples),len(grid),2),np.nan)
    for i,s in enumerate(samples):
        relative=s['video_frames']-s['key'];a,b=offsets[i:i+2]
        for frame in np.unique(relative):
            j=int(frame-grid[0]);assert grid[j]==frame
            tables[:,i,j]=stacked[:,a:b,1:3][:,relative==frame].mean(1)
    # Average seed variability WITHIN interaction first, rather than pretending 5N independent cases.
    present=np.isfinite(tables[0,:,:,0]);assert np.array_equal(np.isfinite(tables),np.broadcast_to(present[None,:,:,None],tables.shape))
    mean_per_interaction=np.where(present[:,:,None],np.nan_to_num(tables).mean(0),np.nan)
    return tables,mean_per_interaction


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    source=args.source.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    tm=json.loads((source/'training_manifest.json').read_text());dataset=ROOT/tm['source'];dm=json.loads((dataset/'dataset_manifest.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in tm['input_hashes'].items())
    selections=json.loads((source/'sigma_selection.json').read_text());results=json.loads((source/'results.json').read_text())
    manifest={r['id']:r for r in map(json.loads,(ROOT/'datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl').read_text().splitlines())}
    fps={manifest[r['rollout_id']]['fps'] for r in dm['records']};assert fps=={30.0};hz=30.0
    hashes={str(p.relative_to(ROOT)):sha(p) for p in (source/'results.json',source/'sigma_selection.json',dataset/'dataset_manifest.json',Path(__file__))}
    torch.set_num_threads(4);all_rows=[];statistics={};population=[];replay_errors=[]
    for group in GROUPS:
        samples=read_samples(dataset,dm,group)
        for split in ('val','test'):
            population.append(dict(group=group,split=split,interactions=len(samples[split]),success=sum(s['outcome']==1 for s in samples[split]),failure=sum(s['outcome']==2 for s in samples[split]),ticks=sum(len(s['labels']) for s in samples[split])))
        for selection in [s for s in selections if s['group']==group]:
            configuration=selection['input']+'_'+selection['head'];prediction_banks={s:[] for s in ('val','test')}
            for seed in range(42,47):
                result=next(r for r in results if r['seed']==seed and all(r[k]==selection[k] for k in ('group','input','head','sigma')))
                directory=ROOT/result['directory'];checkpoint=directory/'best.pt';hashes[str(checkpoint.relative_to(ROOT))]=sha(checkpoint)
                blob=torch.load(checkpoint,map_location='cpu',weights_only=True);model=MergedProbe(**blob['model_config']);model.load_state_dict(blob['state_dict']);model.eval()
                validation,p=evaluate(model,samples['val'],selection['sigma'],'cpu');prediction_banks['val'].append(p)
                replay_errors.append(dict(group=group,input=selection['input'],head=selection['head'],seed=seed,val_cpu_ba=validation['balanced_accuracy'],saved_val_ba=result['validation']['balanced_accuracy']))
                stored_path=directory/'test_predictions.npz';hashes[str(stored_path.relative_to(ROOT))]=sha(stored_path);stored=load(stored_path)
                assert np.array_equal(stored['labels'],np.concatenate([s['labels'] for s in samples['test']]))
                prediction_banks['test'].append(stored['probabilities'])
                for split in ('val','test'):
                    save(output/'predictions'/group/configuration/f'{split}_seed_{seed}.npz',dict(probabilities=prediction_banks[split][-1],offsets=np.r_[0,np.cumsum([len(s['labels']) for s in samples[split]])],labels=np.concatenate([s['labels'] for s in samples[split]])))
            for split in ('val','test'):
                ss=samples[split];lo=min(int(s['video_frames'].min()-s['key']) for s in ss);hi=max(int(s['video_frames'].max()-s['key']) for s in ss);grid=np.arange(lo,hi+1)
                tables,per_interaction=align_interactions(ss,prediction_banks[split],grid);outcomes=np.array([s['outcome'] for s in ss]);stats={}
                for outcome,name in ((1,'success'),(2,'failure')):
                    values=per_interaction[outcomes==outcome];count=np.isfinite(values[:,:,0]).sum(0);mean=np.full((len(grid),2),np.nan);variance=mean.copy();seed_variance=mean.copy()
                    for j,n in enumerate(count):
                        if n:
                            selected=values[:,j];selected=selected[np.isfinite(selected[:,0])];mean[j]=selected.mean(0)
                            if n>1:variance[j]=selected.var(0,ddof=1)
                            seeds=tables[:,outcomes==outcome,j];seed_mean=np.nan_to_num(seeds).sum(1)/n;seed_variance[j]=seed_mean.var(0,ddof=1)
                    stats[name]=dict(mean=mean,variance=variance,std=np.sqrt(variance),seed_variance=seed_variance,count=count,total=int((outcomes==outcome).sum()))
                    for j,frame in enumerate(grid):
                        for c,label in enumerate(('success','failure')):
                            all_rows.append(dict(group=group,split=split,input=selection['input'],head=selection['head'],sigma=selection['sigma'],gt_outcome=name,probability='P('+label+')',relative_frame=int(frame),relative_time_seconds=float(frame/hz),interaction_count=int(count[j]),total_interactions=stats[name]['total'],mean=float(mean[j,c]),variance=float(variance[j,c]),std=float(np.sqrt(variance[j,c])),seed_mean_variance=float(seed_variance[j,c])))
                statistics[(group,split,configuration)]=(grid,stats)
                save(output/'aligned'/group/(configuration+'_'+split+'.npz'),dict(relative_frames=grid,relative_time_seconds=grid/hz,interaction_ids=np.array([s['interaction_id'] for s in ss]),outcomes=outcomes,seed_mean_interaction_probabilities=per_interaction,per_seed_interaction_probabilities=tables))
            print('CURVES',group,configuration,'all val/test,5 seeds',flush=True)
    write_csv(output/'mean_variance.csv',all_rows);write_csv(output/'population.csv',population);dump(output/'validation_replay.json',replay_errors)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=('tab:blue','tab:orange');configurations=[s['input']+'_'+s['head'] for s in selections if s['group']==GROUPS[0]]
    for group in GROUPS:
        for split in ('val','test'):
            for band in ('std','variance'):
                fig=plt.figure(figsize=(17,10));outer=fig.add_gridspec(2,3,hspace=.38,wspace=.22);axes=[]
                for k,configuration in enumerate(configurations):
                    inner=outer[k//3,k%3].subgridspec(2,1,height_ratios=[4,1],hspace=.08);ax=fig.add_subplot(inner[0]);support=fig.add_subplot(inner[1],sharex=ax);axes.append(ax)
                    grid,stats=statistics[(group,split,configuration)];t=grid/hz
                    for outcome,style in (('success','-'),('failure','--')):
                        entry=stats[outcome]
                        for c,label in enumerate(('success','failure')):
                            m=entry['mean'][:,c];spread=entry[band][:,c];ax.plot(t,m,color=colors[c],linestyle=style,label=f'{outcome.title()} GT: P({label})',linewidth=1.6)
                            ax.fill_between(t,np.clip(m-spread,0,1),np.clip(m+spread,0,1),color=colors[c],alpha=.10 if outcome=='success' else .07)
                        support.plot(t,entry['count'],linestyle=style,color='black' if outcome=='success' else 'gray',label=outcome)
                    for axis in (ax,support):axis.axvline(0,color='red',linestyle=':',linewidth=1);axis.grid(alpha=.15)
                    ax.set_ylim(0,1);ax.tick_params(labelbottom=False);ax.set_ylabel('Probability');support.set_ylim(bottom=0);support.set_ylabel('N');support.set_xlabel('Time relative to Key (s)')
                    selected=next(s for s in selections if s['group']==group and s['input']+'_'+s['head']==configuration)
                    ax.set_title(configuration.replace('f6_deform','Fusion')+f" | sigma={selected['sigma']}\nSuccess N={stats['success']['total']}; Failure N={stats['failure']['total']}")
                handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.97),ncol=4,fontsize=10)
                fig.suptitle(f'{group} | {split.upper()} | all interactions | mean ± '+('SD (sqrt interaction variance)' if band=='std' else 'interaction variance'),y=.995,fontsize=15)
                fig.subplots_adjust(top=.90,bottom=.06);fig.savefig(output/f'{group}_{split}_mean_{band}.png',dpi=170);fig.savefig(output/f'{group}_{split}_mean_{band}.pdf');plt.close(fig)
    # Focused verification of aggregation: direct per-frame mean and ddof1 variance agree with exported rows.
    lookup={(r['group'],r['split'],r['input']+'_'+r['head'],r['gt_outcome'],r['relative_frame'],r['probability']):r for r in all_rows}
    for key,(grid,stats) in statistics.items():
        aligned=load(output/'aligned'/key[0]/(key[2]+'_'+key[1]+'.npz'));v=aligned['seed_mean_interaction_probabilities']
        for outcome,name in ((1,'success'),(2,'failure')):
            for j,frame in enumerate(grid):
                present=v[aligned['outcomes']==outcome,j];present=present[np.isfinite(present[:,0])]
                for c,label in enumerate(('success','failure')):
                    r=lookup[(*key,name,int(frame),'P('+label+')')];assert r['interaction_count']==len(present)
                    if len(present):np.testing.assert_allclose(r['mean'],present[:,c].mean(),atol=1e-12)
                    if len(present)>1:np.testing.assert_allclose(r['variance'],present[:,c].var(ddof=1),atol=1e-12)
                    else:assert np.isnan(r['variance'])
    assert len(statistics)==24 and len(population)==4
    assert all(sha(ROOT/p)==v for p,v in hashes.items())
    dump(output/'verification.json',dict(status='PASS',all_interactions_included=True,plots_per_band=4,model_panels=24,seeds_per_checkpoint_config=5,no_training=True,no_extrapolation_or_zero_padding=True,one_vote_per_interaction_and_time=True,means_variances_recomputed=True,input_hashes_unchanged=True))
    dump(output/'manifest.json',dict(created_at=datetime.datetime.now().astimezone().isoformat(),source=str(source.relative_to(ROOT)),population=population,seeds=list(range(42,47)),selected_sigmas=selections,time='(last camera frame - annotation Key frame)/30 seconds',within_interaction='mean duplicate camera-frame ticks then mean5seeds',across_interactions='equal weight at each available relative frame; mean and sample variance(ddof1); no extrapolation; variance NaN for N<2',main_band='mean ± standard deviation(sqrt variance); clipped0..1 for display only',additional_band='literal mean ± variance as requested; CSV retains unclipped mean/variance',validation_predictions='CPU inference from existing selected checkpoints',test_predictions='unchanged stored GPU probabilities',input_hashes=hashes))
    lines=['# 全部 validation/test interaction：Key 对齐平均概率曲线','',f'源实验：`{source.relative_to(ROOT)}`。没有重新训练，没有变更σ、checkpoint、标签或split。','',
    '每组每个模型按原验证集所选σ使用seeds42–46的5个checkpoint。validation在CPU上补算完整概率，test使用原保存概率。所有success/failure interaction均参与，val/test分别画图，不只选示例；GT分组按interaction最终outcome，而非预测或当前硬GT。每个模型最终四条曲线：Success GT的P(success)/P(failure)，Failure GT的P(success)/P(failure)。不展示P(in_progress)，但预测仍保持三类softmax。','',
    '横轴time relative to Key=(当前last相机帧−该interaction标注Key帧)/fps，全部源录像fps=30。0为原标注Key，而非将首次有效tick强行移到0；单位秒，不按interval长度归一化、不按有效tick序号压缩缺帧。','',
    '同一个interaction在同一相机帧有重复tactile tick时先对预测取均值；再对5seeds取均值，每条interaction在同一相对时间只贡献一个值。按最终success/failure分别在当时有数据的interaction上等权平均。缺失或超出interaction观察范围的时间不插值、不补0、不延伸尾段；因此边缘N较小，mean不是固定成员全时段平均。下方N曲线标明每个相对时间参与的interaction数，实线Success、虚线Failure。','',
    '主要图阴影=mean±SD，SD=√interaction variance，variance使用样本方差(ddof=1)；这是interaction间离散程度，不是置信区间或seed方差。另提供严格mean±variance图，满足字面要求。显示时上下界裁到[0,1]，CSV保存未裁剪mean/variance/SD。N=1仍显示该interaction的均值，但方差/SD未定义，记NaN并不画阴影；N=0均值也记NaN。mean_variance.csv另外保留seed_mean_variance（5个seed在当前可用interaction上的均值之间的方差），两种方差不混在一起。','',
    '| 组 | split | Success interactions | Failure interactions |','|---|---|---:|---:|']
    for p in population:lines.append(f"| {p['group']} | {p['split']} | {p['success']} | {p['failure']} |")
    for group in GROUPS:
        for split in ('val','test'):
            lines+=['',f'## {group} / {split}','',f'![Mean ± SD]({group}_{split}_mean_std.png)','',f'[PDF]({group}_{split}_mean_std.pdf) · [严格 mean ± variance 图]({group}_{split}_mean_variance.png) · [variance PDF]({group}_{split}_mean_variance.pdf)']
    lines+=['','[全部均值/方差/SD与N](mean_variance.csv) · [人数统计](population.csv) · [验证](verification.json)。aligned目录保留每条interaction的对齐数组，predictions目录保留完整概率与interaction offsets，可重新计算所有曲线。','']
    (output/'README.md').write_text('\n'.join(lines));snapshot=output/'code_snapshot';snapshot.mkdir();shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    target=ROOT/'WeeklySummary/10.5/gaussian_online'/output.name;target.mkdir(parents=True,exist_ok=False);copies=[]
    for p in output.iterdir():
        if not p.is_file():continue
        q=target/p.name;shutil.copy2(p,q);copies.append(dict(source=str(p.relative_to(ROOT)),destination=str(q.relative_to(ROOT)),sha256=sha(p)))
    dump(target/'copy_manifest.json',copies);assert all(sha(ROOT/r['source'])==sha(ROOT/r['destination'])==r['sha256'] for r in copies)
    for base in (source,ROOT/'WeeklySummary/10.5/gaussian_online'):
        with (base/'README.md').open('a') as f:f.write(f'\n\n## 全部interaction的Key对齐统计\n\n已增加validation/test全部success/failure interaction的四条平均概率曲线，5seeds先在interaction内平均，再对interaction等权求均值及方差，横轴秒。主图mean±SD，另有严格mean±variance图与逐时间N。见[完整统计报告]({output.name}/README.md)。\n')
    # Existing copied README changed together: update its hash and preserve all previous copy entries.
    oldcopy=ROOT/'WeeklySummary/10.5/gaussian_online/copy_manifest.json';previous=json.loads(oldcopy.read_text())
    for r in previous:
        if r['source']==str((source/'README.md').relative_to(ROOT)):r['sha256']=sha(source/'README.md')
    dump(oldcopy,previous);assert all(sha(ROOT/r['source'])==sha(ROOT/r['destination'])==r['sha256'] for r in previous)
    weekly=ROOT/'WeeklySummary/10.5/10.5.md'
    with weekly.open('a') as f:
        f.write(f'\n\n## 全部validation/test interaction的Key对齐均值与方差\n\n[完整说明与四组图](gaussian_online/{output.name}/README.md)。每模型四条曲线：Success GT的P(success)/P(failure)，Failure GT的P(success)/P(failure)。横轴实际相机时间相对Key（秒），5seeds先在interaction内平均，再对当前有数据的interaction等权平均；主图阴影为SD=√variance，提供literal variance图。缺失位置不补值，下方N显示支持数量，val/test保持分开。\n')
        for group in GROUPS:
            for split in ('val','test'):f.write(f'\n![{group} {split}全部interaction均值](gaussian_online/{output.name}/{group}_{split}_mean_std.png)\n')
    print('COMPLETE',output,len(all_rows),'CSV rows',flush=True)

if __name__=='__main__':main()
