"""Record the existing unaugmented baseline before defining a new step ablation."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import numpy as np
from .common import ROOT,dump,sha
from .merged_online import load


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'outputs/sharpa_gaussian_online/20261005_182000';dataset=ROOT/'outputs/sharpa_gaussian_online_data/20261005_182000';aligned=source/'key_aligned_20261005_184000'
    protocol=json.loads((source/'protocol.json').read_text());dm=json.loads((dataset/'dataset_manifest.json').read_text());selected=json.loads((source/'sigma_selection.json').read_text())
    config=next(r for r in selected if r['group']=='original_align' and r['input']=='deform' and r['head']=='gru');assert config['sigma']==8
    assert 'no augmentation' in protocol['loss'] and 'no augmentation' in dm['loss']
    directory=source/'runs/original_align/deform_gru/sigma_8/seed_42';metrics=json.loads((directory/'metrics.json').read_text());dump(out/'baseline_metrics.json',metrics)
    inputs=[source/'protocol.json',source/'sigma_selection.json',dataset/'dataset_manifest.json',directory/'best.pt',directory/'metrics.json',Path(__file__)];rows=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
    for si,split in enumerate(('val','test')):
        samples=[r for r in dm['records'] if r['group']=='original_align' and r['split']==split];predpath=aligned/'predictions/original_align/deform_gru'/f'{split}_seed_42.npz';pred=load(predpath);inputs.append(predpath)
        observations=[]
        for i,r in enumerate(samples):
            path=dataset/r['feature_path'];data=load(path);inputs.append(path);a,b=pred['offsets'][i:i+2];p=pred['probabilities'][a:b];assert len(p)==len(data['video_frames'])
            relative=data['video_frames']-r['key'];observations.append((r['outcome'],{int(f):p[relative==f].mean(0) for f in np.unique(relative)}))
        lo=min(min(o[1]) for o in observations);hi=max(max(o[1]) for o in observations);grid=np.arange(lo,hi+1)
        for oi,outcome in enumerate((1,2)):
            name='success' if outcome==1 else 'failure';members=[v for o,v in observations if o==outcome];means=np.full((len(grid),3),np.nan);std=means.copy();count=[]
            for j,frame in enumerate(grid):
                values=np.array([v[int(frame)] for v in members if int(frame) in v]);n=len(values);count.append(n)
                if n:means[j]=values.mean(0);np.testing.assert_allclose(means[j].sum(),1,atol=1e-6)
                if n>1:std[j]=values.std(0,ddof=1)
                for c,label in enumerate(('in_progress','success','failure')):rows.append(dict(split=split,gt_outcome=name,relative_time_seconds=float(frame/30),relative_frame=int(frame),probability=label,mean=float(means[j,c]),std=float(std[j,c]),interaction_count=n,total_interactions=len(members)))
            ax=axes[si,oi]
            for c,(label,color) in enumerate(zip(('in_progress','success','failure'),('gray','tab:blue','tab:orange'))):
                ax.plot(grid/30,means[:,c],label='P('+label+')',color=color);ax.fill_between(grid/30,np.clip(means[:,c]-std[:,c],0,1),np.clip(means[:,c]+std[:,c],0,1),color=color,alpha=.12)
            ax.axvline(0,color='red',linestyle=':');ax.set_ylim(0,1);ax.set_title(f'{split.upper()} | {name.title()} GT | all {len(members)} interactions');ax.set_xlabel('Time relative to Key (s)');ax.set_ylabel('Probability');ax.grid(alpha=.15)
    axes[0,0].legend();fig.suptitle('Existing NO-augmentation baseline | original Align / Deform GRU / sigma8 / seed42\nMean ± interaction SD; no new training');fig.tight_layout();fig.savefig(out/'baseline_key_relative_three_class.png',dpi=170);fig.savefig(out/'baseline_key_relative_three_class.pdf');plt.close(fig)
    with (out/'baseline_curves.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    dump(out/'audit.json',dict(status='AWAITING_AUGMENTATION_DEFINITION',created_at=datetime.datetime.now().astimezone().isoformat(),current_gaussian_augmentation=False,current_sigma=8,group='original_align',input='deform',head='gru',seed=42,existing_baseline_reusable=True,no_new_training=True,reason='Current Gaussian code uses dense sequences with no step augmentation. Historical sampling step and Key jitter are different methods; need explicit definition and values for new augmented arm.',input_hashes=hashes))
    lines=['# Step augmentation ablation：当前设置核对','', '当前Gaussian实验无step augmentation：protocol.loss及dataset_manifest.loss均明确记录no augmentation；gaussian_online.py直接遍历完整interaction，batch()按原positions/key_index生成soft target，没有step、stride、Key jitter或输入位移分支。历史Key-window实验有采样间隔及Key jitter，但不属于当前Gaussian baseline。','', '固定original_align / Deform GRU / σ=8 / seed42，原rollout split与已有最佳epoch15不变。当前已有结果可直接作为“不增强”组；这里仅整理原结果和三类概率曲线，不重复训练，也不将不存在的augmentation伪装成第二组。新增组的step定义及取值待确认，不能计算两组差异或触发补seed规则。','', '| split | BA % | Macro F1 % | Failure precision % | Failure recall % | Failure FPR % |','|---|---:|---:|---:|---:|---:|']
    for split,key in (('val','validation'),('test','test')):
        m=metrics[key];vals=[m['balanced_accuracy'],m['macro_f1'],m['per_class']['failure']['precision'],m['per_class']['failure']['recall'],m['failure_false_positive_rate']];lines.append('| '+split+' | '+' | '.join(f'{100*v:.2f}' for v in vals)+' |')
    lines+=['','## 所有val/test interaction的三类曲线','', '每个GT outcome下画P(in_progress)、P(success)、P(failure)，不是只挑一个interaction。Key按相机帧对齐，秒=(lastframe−Key)/30，重复相机帧先平均，每条interaction等权，缺失不补值，mean±interaction SD。这里只使用seed42，不是五seed平均；N与均值在baseline_curves.csv。Validation复用先前CPU回放概率，test为原保存GPU概率；上表为原GPU评估指标。','', '![三类baseline曲线](baseline_key_relative_three_class.png)','', '[原指标](baseline_metrics.json) · [曲线数据及N](baseline_curves.csv) · [配置审计](audit.json)','']
    (out/'README.md').write_text('\n'.join(lines));assert all(sha(ROOT/p)==h for p,h in hashes.items());print('AUDIT_COMPLETE',out,flush=True)

if __name__=='__main__':main()
