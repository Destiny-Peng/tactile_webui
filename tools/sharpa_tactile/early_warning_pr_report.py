"""Validation-only curves and checkpoint audit for PR-AUC selected diagnostics."""
from .common import project_path, relative_path
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,dump,sha
from .early_warning import metrics,write_csv

HORIZONS=(0,8,15)
KEYS=('train_balanced_BCE','val_natural_BCE','val_train_weighted_BCE','val_PR_AUC','val_ROC_AUC','val_BA_at_0_5')
LABELS=('Train balanced BCE','Val natural BCE','Val train-weighted BCE','Val PR-AUC (AP)','Val ROC-AUC','Val BA @0.5 (diagnostic)')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve()
    allrows=[];summary=[];audit=[];comparisons=[]
    fig,axes=plt.subplots(6,3,figsize=(16,19),sharex='col')
    for hi,h in enumerate(HORIZONS):
        directory=out/f'H{h}';hist=list(csv.DictReader((directory/'epoch_metrics.csv').open()))
        history=[{k:int(v) if k=='epoch' else float(v) for k,v in r.items()} for r in hist]
        result=json.loads((directory/'result.json').read_text());best=result['best_epoch'];n=len(history)
        maxscore=-1;expected=0
        for r in history:
            if r['val_PR_AUC']>maxscore+1e-8:maxscore=r['val_PR_AUC'];expected=r['epoch']
        assert best==expected and n<=30 and (n==30 or n-best==8) and not result['test_evaluated']
        index=list(csv.DictReader((directory/'checkpoint_index.csv').open()))
        assert [int(x['epoch']) for x in index]==list(range(n+1))
        for row in index:assert sha(directory/row['path'])==row['sha256']
        assert sha(directory/'best_pr_auc.pt')==sha(directory/'checkpoints'/f'epoch_{best:03d}.pt')
        # Replay EVERY saved val epoch score without any model/test inference.
        for row in history:
            epoch=row['epoch']
            with np.load(directory/'validation_scores'/f'epoch_{epoch:03d}.npz') as z:
                valid=z['valid'];gt=z['labels'][valid];logit=z['logits'][valid].astype(np.float64)
            rank=metrics(gt,logit);probability=np.exp(-np.logaddexp(0,-logit));frame=metrics(gt,probability)
            natural=float((np.logaddexp(0,logit)-gt*logit).mean())
            assert np.isclose(rank['pr_auc'],row['val_PR_AUC']) and np.isclose(rank['roc_auc'],row['val_ROC_AUC'])
            assert np.isclose(frame['balanced_accuracy'],row['val_BA_at_0_5']) and np.isclose(natural,row['val_natural_BCE'])
            allrows.append(dict(horizon_frames=h,seed=42,**row))
        old=project_path('outputs/sharpa_early_warning/20261006_001500/runs', f'H{h}', 'seed_42')
        oldhist=json.loads((old/'history.json').read_text());overlap=min(n,len(oldhist))
        delta_loss=max(abs(history[i]['train_balanced_BCE']-oldhist[i]['train_balanced_BCE']) for i in range(overlap))
        delta_ba=max(abs(history[i]['val_BA_at_0_5']-oldhist[i]['validation_BA']) for i in range(overlap))
        audit.append(dict(horizon_frames=h,epochs=n,best_PR_AUC_epoch=best,patience_actual=n-best,epoch_checkpoints=n+1,
            best_checkpoint_SHA_matches=True,val_all_epochs_replayed=True,prefix_max_abs_train_BCE_delta=delta_loss,prefix_max_abs_val_BA_delta=delta_ba))
        original=json.loads((old/'metrics.json').read_text());oldbest=original['best_epoch']
        bestrow=history[best-1];oldrow=history[oldbest-1];last=history[-1]
        summary.append(dict(horizon_frames=h,seed=42,epochs=n,best_epoch=best,old_BA_best_epoch=oldbest,**{k:bestrow[k] for k in KEYS}))
        for name,row in (('old_BA_selected',oldrow),('PR_AUC_selected',bestrow),('last_trained',last)):
            comparisons.append(dict(horizon_frames=h,checkpoint=name,**row))
        x=np.array([r['epoch'] for r in history]);individual,iaxes=plt.subplots(3,2,figsize=(12,10))
        for ki,(key,label) in enumerate(zip(KEYS,LABELS)):
            y=[r[key] for r in history]
            for ax in (axes[ki,hi],iaxes.ravel()[ki]):
                ax.plot(x,y,marker='.',ms=3);ax.axvline(best,color='tab:red',ls='--',label=f'best AP epoch{best}')
                ax.axvline(oldbest,color='gray',ls=':',label=f'old best BA epoch{oldbest}');ax.set_title(f'H{h}: {label}');ax.grid(alpha=.2);ax.set_xlabel('epoch')
                if 'AUC' in key or 'BA_at' in key:ax.set_ylim(-.03,1.03)
                ax.legend(fontsize=7)
        individual.tight_layout();individual.savefig(out/f'H{h}_epoch_curves.png',dpi=170);individual.savefig(out/f'H{h}_epoch_curves.pdf');plt.close(individual)
    fig.tight_layout();fig.savefig(out/'epoch_curves.png',dpi=150);fig.savefig(out/'epoch_curves.pdf');plt.close(fig)
    write_csv(out/'epoch_metrics.csv',allrows);write_csv(out/'validation_summary.csv',summary);write_csv(out/'checkpoint_comparison.csv',comparisons);write_csv(out/'checkpoint_audit.csv',audit)
    dump(out/'results.json',summary)
    dump(out/'verification.json',dict(status='PASS',test_evaluated=False,threshold_calibrated=False,all_epoch_checkpoints=sum(r['epoch_checkpoints'] for r in audit),audit=audit))
    totalbytes=sum(p.stat().st_size for h in HORIZONS for p in (out/f'H{h}'/'checkpoints').glob('*.pt'))
    text=['# Early warning：PR-AUC 选 epoch，validation-only 诊断\n',
        '只跑 Deform+GRU、H={0,8,15}、seed42，max epoch30、patience8。按 validation frame PR-AUC（average precision）保存 best_pr_auc.pt；BA@0.5 仅诊断，不影响 checkpoint 或 early stopping。H30/45 未运行；本轮没有 test 推理/指标或 threshold calibration。\n',
        '## 输入、标签与训练\n',
        '保持历史直接分类 baseline：冻结当前 Deform2560D→Linear128→单层单向GRU128→Linear1→sigmoid；没有额外ReLU或三阶段value模型。保持原rollout train80（64success/16failure）/val17（14/3）划分，历史test split保留不用。使用原final Align Key，不使用新3/4标注。仅最终Align interval为failure时，其end作anchor；success与早期Key不建立正标签。H>0时[anchor−H,anchor)为1，更早为0，failure t≥anchor不参与loss/主val指标；H0时failure t≥anchor为1，之前为0，success全0。H按30fps相机frame计。\n',
        'Train-only标准化（std≥.01）；balanced BCE：w0=N/(2N0)、w1=N/(2N1)，只用对应H train有效tick统计。AdamW lr=.001、wd=.0001、batch8 rollout、clip1。无augmentation、无Gaussian。回归状态和padding处理保持旧baseline。每H独立初始化seed42、shuffle同旧实现。\n',
        '## 每 epoch 六项指标\n',
        '- train balanced BCE：训练过程中各batch有效tick的加权BCE累积平均，参数在epoch中更新，不是额外epoch-end train推理。\n- val natural BCE：epoch结束eval模式，对全部有效val tick求稳定的未加权BCE(logits,GT)。\n- val train-weighted BCE：同一val BCE乘train固定w0/w1后求均值；不重算val权重，仅诊断。\n- val PR-AUC：有效val tick汇总的average precision，作为唯一model-selection和early-stopping criterion。\n- val ROC-AUC：有效val tick的ROC曲线面积，仅诊断。\n- val BA@0.5：两类recall平均，仅诊断。\n',
        'PR-AUC/ROC-AUC按原始logit排序，数学上与sigmoid排序等价，避免sigmoid数值饱和导致假tie。GT、AP tie分组/积分函数与旧baseline相同。early stop：AP改善需>1e−8，平手保留最早，连续8 epoch无改善停止，最多30。weighted val BCE不参与选epoch。\n',
        '## Validation 最佳 PR-AUC checkpoint\n','|H|Best AP epoch|Stopped epoch|Val PR-AUC|Val ROC-AUC|Val BA@0.5|Val natural BCE|Val train-weighted BCE|\n|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:text.append(f"|{r['horizon_frames']}|{r['best_epoch']}|{r['epochs']}|{r['val_PR_AUC']:.4f}|{r['val_ROC_AUC']:.4f}|{r['val_BA_at_0_5']:.2%}|{r['val_natural_BCE']:.4f}|{r['val_train_weighted_BCE']:.4f}|")
    text.extend(['\n## 曲线与诊断\n','![All epoch metrics](epoch_curves.png)\n',
        '六行依次为train balanced BCE、val natural BCE、val train-weighted BCE、val PR-AUC、val ROC-AUC、val BA@0.5。红虚线是本轮best AP epoch；灰点线是旧best BA epoch。没有对曲线做平滑。\n',
        '不能把本轮一概归为纯threshold问题或纯过拟合：不同指标在不同阶段有不同变化。下表直接比较旧BA选点、本轮AP选点和最后epoch。\n',
        '|H|Checkpoint|Epoch|Train BCE|Val natural BCE|Val weighted BCE|Val AP|Val ROC-AUC|Val BA@0.5|\n|---:|---|---:|---:|---:|---:|---:|---:|---:|'])
    for r in comparisons:text.append(f"|{r['horizon_frames']}|{r['checkpoint']}|{r['epoch']}|{r['train_balanced_BCE']:.4f}|{r['val_natural_BCE']:.4f}|{r['val_train_weighted_BCE']:.4f}|{r['val_PR_AUC']:.4f}|{r['val_ROC_AUC']:.4f}|{r['val_BA_at_0_5']:.2%}|")
    for h in HORIZONS:
        rows=[r for r in comparisons if r['horizon_frames']==h];old,best,last=rows
        text.append(f"\nH={h}：epoch{old['epoch']}→AP最佳epoch{best['epoch']}，BA {old['val_BA_at_0_5']:.2%}→{best['val_BA_at_0_5']:.2%}，AP {old['val_PR_AUC']:.4f}→{best['val_PR_AUC']:.4f}，ROC-AUC {old['val_ROC_AUC']:.4f}→{best['val_ROC_AUC']:.4f}。之后到epoch{last['epoch']}，AP为{last['val_PR_AUC']:.4f}、ROC-AUC为{last['val_ROC_AUC']:.4f}；应同时看排序、固定阈值行为和BCE，不能仅凭BA下降判断信号消失。")
    text.extend(['\nValidation只有3条failure，H8/H15分别仅24/45个positive tick；帧之间高度相关。这些曲线可用于诊断和选epoch，不能当作独立泛化测试。按用户要求暂不查看test。\n',
        '## 全 epoch checkpoint 与独立加载\n',
        f'保留全部已运行epoch（含初始化epoch000），共{sum(r["epoch_checkpoints"] for r in audit)}个state_dict，合计{totalbytes/1024**2:.1f}MiB；没有承诺保存早停后的未运行epoch。每H目录结构：\n',
        '```text\nH0/  # H8、H15 同样\n  epoch_metrics.csv\n  checkpoint_index.csv\n  checkpoints/epoch_000.pt ... epoch_NNN.pt\n  best_pr_auc.pt\n  best_pr_auc.json\n  validation_scores/epoch_001.npz ...\n```\n',
        '每个.pt直接保存raw state_dict，含train mean/std buffers；best_pr_auc.pt与选中epoch文件SHA一致。epoch000为训练前，epoch001为完成第一次epoch后。validation_scores保留各epoch连续logit、GT、valid mask、rollout offsets与frame坐标，之后可直接重算指标，无需重训。优化器状态未保存，本轮checkpoint用于独立推理评估，非完整训练恢复。\n',
        '按 [PyTorch 官方 state_dict 保存/加载方式](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) 实现。也可用项目helper：\n',
        '```python\nimport torch\nfrom sharpa_tactile.early_warning_pr_diagnostic import load_model\nmodel = load_model(".../H0/best_pr_auc.pt", device="cuda:0")\n# 输入原始 frozen Deform features [B,T,2560]；模型内部执行train mean/std归一化\nmodel.eval()\nwith torch.inference_mode():\n    risk = model(features).sigmoid()\n```\n',
        '全epoch checkpoint SHA、best选点、patience、所有epoch的val AP/ROC-AUC/BA/natural BCE均已复核；checkpoint_audit.csv包含与旧训练重叠前缀的train BCE/BA差值。完整weights/scores仅留原outputs，周报复制指标、图和文档。原标注与旧实验不改。\n'])
    text.insert(1,'诊断结论：H8/H15从epoch2到epoch17，BA@0.5降至约50%，但AP分别从0.0344→0.2467、0.0556→0.1472，ROC-AUC也提升，支持这一阶段存在明显的score尺度/固定阈值问题。最佳AP时两组正例score最大值仅0.4293/0.3172，因此0.5阈值没有检出正例。继续到epoch25，train BCE继续下降，而val natural BCE上升、AP与ROC-AUC下降，说明后续也出现真实的validation泛化退化；两种现象可以先后存在。H0的AP提升同时ROC-AUC略降，表现不是完全一致的排序改善。当前仅validation诊断，不做阈值搜索或test验证。')
    (out/'README.md').write_text('\n'.join(text))
    code=out/'code_snapshot';code.mkdir(exist_ok=True)
    for name in ('early_warning_pr_diagnostic.py','early_warning_pr_report.py','early_warning.py','common.py'):shutil.copy2(project_path('tools/sharpa_tactile', name),code/name)
    shutil.copy2(project_path('tools/run_early_warning_pr_diagnostic.sh'),code/'run_early_warning_pr_diagnostic.sh')
    weekly=project_path('WeeklySummary/10.5');dest=weekly/'early_warning_pr_diagnostic'/out.name;dest.mkdir(parents=True,exist_ok=True)
    for f in out.iterdir():
        if f.is_file() and f.suffix in ('.md','.csv','.json','.png','.pdf'):shutil.copy2(f,dest/f.name)
    doc=(out/'README.md').read_text().replace('(epoch_curves.png)',f'(early_warning_pr_diagnostic/{out.name}/epoch_curves.png)')
    doc+='\n原始完整目录：`'+str(relative_path(out))+'`。\n'
    (weekly/'10.5early_warning_pr_diagnostic.md').write_text(doc)
    print(json.dumps(summary,indent=2));print('REPORT_COMPLETE')


if __name__=='__main__':main()
