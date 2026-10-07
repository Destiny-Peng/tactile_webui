"""Plot only failed validation rollouts using exact best-PR checkpoint epoch scores."""
from .common import project_path, relative_path
import argparse
import json
import re
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .common import ROOT,sha,dump
from .early_warning import write_csv


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve()
    manifest=json.loads((out/'H0/source_dataset_manifest.json').read_text())
    records={r['rollout_id']:r for r in manifest['records'] if r['group']=='merged_align' and r['split']=='val'}
    failids=[rid for rid,r in records.items() if r['outcome']==2];assert len(failids)==3
    short=lambda rid:(re.search(r'_(\d{4})-',rid).group(1) if re.search(r'_(\d{4})-',rid) else rid[-16:])
    banks={};frame_rows=[];summaries=[];provenance=[]
    for h in (0,8,15):
        directory=out/f'H{h}';best=json.loads((directory/'best_pr_auc.json').read_text());epoch=best['epoch']
        assert sha(directory/'best_pr_auc.pt')==best['sha256']==sha(directory/best['checkpoint'])
        scorefile=directory/'validation_scores'/f'epoch_{epoch:03d}.npz'
        with np.load(scorefile) as z:
            for i,rid in enumerate(z['rollout_ids'].tolist()):
                if rid not in failids:continue
                a,b=z['offsets'][i:i+2];frames=z['frames'][a:b].copy();logits=z['logits'][a:b].astype(float)
                probability=np.exp(-np.logaddexp(0,-logits));valid=z['valid'][a:b].copy();gt=z['labels'][a:b].copy();key=records[rid]['key']
                banks[h,rid]=(frames,probability,valid,gt,key,epoch)
                band=valid & (gt==1);hit=valid & (probability>=.5)
                summaries.append(dict(horizon_frames=h,rollout_id=rid,key_frame=key,checkpoint_epoch=epoch,
                    valid_ticks=int(valid.sum()),positive_band_ticks=int(band.sum()),max_valid_probability=float(probability[valid].max()),
                    max_positive_band_probability=float(probability[band].max()),positive_band_detected_at_0_5=bool(np.any(hit&band)),
                    first_alarm_frame=int(frames[np.flatnonzero(hit)[0]]) if hit.any() else None))
                for j in range(len(frames)):
                    frame_rows.append(dict(horizon_frames=h,rollout_id=rid,checkpoint_epoch=epoch,frame=int(frames[j]),key_frame=key,relative_seconds=float((frames[j]-key)/30),
                        logit=float(logits[j]),risk=float(probability[j]),valid_for_original_evaluation=bool(valid[j]),gt=int(gt[j]) if valid[j] else None))
        provenance.append(dict(horizon_frames=h,checkpoint_epoch=epoch,checkpoint_sha256=best['sha256'],score_file=str(relative_path(scorefile)),score_sha256=sha(scorefile)))
    fig,axes=plt.subplots(3,3,figsize=(18,11),sharey=True)
    def draw(ax,h,rid):
        frames,probability,valid,gt,key,epoch=banks[h,rid];x=(frames-key)/30
        ax.axvspan(max(x.min(),-h/30) if h else 0,0 if h else x.max(),color='tab:orange',alpha=.2)
        if h>0 and x.max()>=0:
            ax.axvspan(0,x.max(),color='gray',alpha=.12)
        ax.plot(x,np.where(valid,probability,np.nan),color='tab:blue',lw=1.6)
        if (~valid).any():ax.plot(x,np.where(~valid,probability,np.nan),color='gray',ls='--',lw=1.2)
        ax.axvline(0,color='tab:red',ls='--',lw=1.2);ax.axhline(.5,color='black',ls=':',lw=1)
        ax.set_ylim(-.03,1.03);ax.set_xlim(x.min(),x.max());ax.grid(alpha=.18)
        ax.set_title(f'Rollout {short(rid)} | H={h} | best AP epoch {epoch}')
        ax.set_xlabel('Time relative to Key (s)');ax.set_ylabel('Risk = sigmoid(logit)')
    for ri,rid in enumerate(failids):
        for hi,h in enumerate((0,8,15)):draw(axes[ri,hi],h,rid)
    handles=[Line2D([],[],color='tab:blue',label='Risk (valid region)'),Patch(facecolor='tab:orange',alpha=.2,label='Positive GT band'),
        Line2D([],[],color='tab:red',ls='--',label='Final failure Key'),Line2D([],[],color='black',ls=':',label='0.5 diagnostic threshold'),
        Line2D([],[],color='gray',ls='--',label='Post-Key: excluded for H>0')]
    fig.suptitle('Best PR-AUC checkpoints | validation failures only | seed 42',fontsize=16)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.955),ncol=5,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(out/'val_fail_best_pr_curves.png',dpi=180);fig.savefig(out/'val_fail_best_pr_curves.pdf');plt.close(fig)
    for rid in failids:
        fig,axes=plt.subplots(1,3,figsize=(17,4.5),sharey=True)
        for hi,h in enumerate((0,8,15)):draw(axes[hi],h,rid)
        fig.tight_layout();fig.savefig(out/f'val_fail_{short(rid)}.png',dpi=170);fig.savefig(out/f'val_fail_{short(rid)}.pdf');plt.close(fig)
    write_csv(out/'val_fail_best_pr_scores.csv',frame_rows);write_csv(out/'val_fail_best_pr_summary.csv',summaries)
    dump(out/'val_fail_plot_provenance.json',dict(checkpoints=provenance,rollouts=failids,note='Exact saved validation scores from selected epochs, no retraining or new inference; no success/test plot or calibration.'))
    paragraph='''\n\n## 新 best PR-AUC checkpoint：validation failure 单条曲线\n\n仅画validation中的三条failure rollout，行对应rollout、列对应H0/8/15。使用与best_pr_auc.pt相同epoch的已保存连续logit，checkpoint与epoch文件SHA逐一匹配，不重新训练或校准阈值。横轴为相对原Key的秒数，纵轴为sigmoid risk。橙色为正例GT带，红线为Key，黑点线为0.5诊断阈值。H>0的Key后尾段以灰色虚线显示，仅供观察，不参与原训练/validation指标；H0的Key后尾段是正例。无曲线平滑。\n\n![Validation failures](val_fail_best_pr_curves.png)\n\n逐帧结果见val_fail_best_pr_scores.csv；各rollout正带最大分数与检出情况见val_fail_best_pr_summary.csv；单rollout图见val_fail_*.png/PDF。\n'''
    readme=out/'README.md';text=readme.read_text();marker='\n\n## 新 best PR-AUC checkpoint：validation failure 单条曲线'
    if marker in text:text=text.split(marker)[0]
    readme.write_text(text+paragraph)
    weekly=project_path('WeeklySummary/10.5');dest=weekly/'early_warning_pr_diagnostic'/out.name
    for f in out.glob('val_fail*'):shutil.copy2(f,dest/f.name)
    shutil.copy2(readme,dest/'README.md')
    doc=weekly/'10.5early_warning_pr_diagnostic.md';text=doc.read_text()
    if marker in text:text=text.split(marker)[0]
    doc.write_text(text+paragraph.replace('(val_fail_best_pr_curves.png)',f'(early_warning_pr_diagnostic/{out.name}/val_fail_best_pr_curves.png)'))
    shutil.copy2(Path(__file__),out/'code_snapshot'/Path(__file__).name)
    print(json.dumps(summaries,indent=2));print('PLOT_COMPLETE')


if __name__=='__main__':main()
