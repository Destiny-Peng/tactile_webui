"""Add GT-row-normalized confusion matrices to an existing probe report."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from .common import ROOT, dump, sha


def normalized_confusions(output, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[]
    for result in results:
        counts=np.asarray(result['test']['confusion_matrix'],dtype=np.float64)
        support=counts.sum(axis=1,keepdims=True)
        normalized=np.divide(counts,support,out=np.zeros_like(counts),where=support!=0)
        np.testing.assert_allclose(normalized.sum(axis=1)[support[:,0]>0],1)
        for i,label in enumerate(('in_progress','success','failure')):
            assert np.isclose(normalized[i,i],result['test']['per_class'][label]['recall'])
        records.append(dict(group=result['group'],input=result['input'],head=result['head'],seed=result['seed'],class_order=['in_progress','success','failure'],row_support=support[:,0].astype(int).tolist(),normalized_by_gt_row=normalized.tolist()))
    dump(output/'confusion_normalized.json',records)
    groups=list(dict.fromkeys(r['group'] for r in results))
    fig,axes=plt.subplots(len(groups),6,figsize=(19,7),squeeze=False)
    for i,group in enumerate(groups):
        selected=[r for r in records if r['group']==group and r['seed']==42]
        assert len(selected)==6
        for ax,result in zip(axes[i],selected):
            cm=np.asarray(result['normalized_by_gt_row'])
            img=ax.imshow(cm,cmap='Blues',vmin=0,vmax=1)
            for y in range(3):
                for x in range(3):
                    ax.text(x,y,f'{100*cm[y,x]:.1f}%',ha='center',va='center',fontsize=9,color='white' if cm[y,x]>.55 else 'black')
            ax.set_xticks(range(3),['Prog','Succ','Fail'])
            ax.set_yticks(range(3),[f'{name}\nN={n}' for name,n in zip(('Prog','Succ','Fail'),result['row_support'])])
            ax.set_xlabel('Prediction'); ax.set_ylabel('GT')
            ax.set_title(result['input'].replace('f6_deform','Fusion')+' '+result['head'].upper(),fontsize=10)
    fig.suptitle('Seed 42 | GT-row normalized | top: symmetric; bottom: asymmetric + merged',fontsize=13)
    fig.tight_layout(rect=(0,0,.95,.94))
    fig.colorbar(img,ax=axes.ravel().tolist(),fraction=.018,pad=.025,label='Fraction within GT class')
    fig.savefig(output/'confusion_seed42_normalized.png',dpi=170); plt.close(fig)
    return records


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); output=args.output.resolve()
    results=json.loads((output/'results.json').read_text())
    normalized_confusions(output,results)
    readme=output/'README.md'
    addition='''## 按 GT 行归一化的混淆矩阵

每个格子 = 该格子的窗口数 / 该 GT 类别的总窗口数。行是 GT、列是预测，类别顺序为 in_progress / success / failure；每行总和为 100%（显示值四舍五入可能有微小差异），对角线就是各类 recall。图中 N 为该类真实测试窗口数。归一化便于比较各类识别比例，没有重新采样或改变测试分布，也没有改变 ACC、BA、F1。没有样本的 GT 行设为 0；本实验各类均有样本。

下图为 seed 42 的 12 个模型；原始计数图继续保留。全部 60 次实验的归一化矩阵与各行 support 见 [confusion_normalized.json](confusion_normalized.json)，不是把五个 seed 的计数混合。

![Seed 42 按 GT 行归一化](confusion_seed42_normalized.png)
'''
    if '## 按 GT 行归一化的混淆矩阵' not in readme.read_text():
        readme.write_text(readme.read_text()+'\n'+addition)
    target=ROOT/'WeeklySummary/10.5/align_online_training'
    copies=json.loads((target/'copy_manifest.json').read_text())
    for name in ('README.md','confusion_normalized.json','confusion_seed42_normalized.png'):
        shutil.copy2(output/name,target/name)
        entry=dict(source=str((output/name).relative_to(ROOT)),destination=str((target/name).relative_to(ROOT)),sha256=sha(output/name))
        copies=[row for row in copies if row['destination']!=entry['destination']]; copies.append(entry)
    dump(target/'copy_manifest.json',copies)
    weekly=ROOT/'WeeklySummary/10.5/10.5.md'; text=weekly.read_text()
    if 'align_online_training/confusion_seed42_normalized.png' not in text:
        marker='![修正数据集五 seed 结果](align_online_training/comparison.png)'
        assert marker in text
        text=text.replace(marker,marker+'\n\n按 GT 行归一化后，每行总和为 100%，对角线为对应类别 recall；N 是该类真实测试窗口数。图为 seed 42，原计数图保留；[全部 60 次归一化矩阵](align_online_training/confusion_normalized.json)。\n\n![修正数据集按 GT 行归一化混淆矩阵](align_online_training/confusion_seed42_normalized.png)')
        weekly.write_text(text)
    snapshot=output/'code_snapshot'/Path(__file__).name
    shutil.copy2(Path(__file__),snapshot)
    for row in copies:
        assert sha(ROOT/row['source'])==sha(ROOT/row['destination'])==row['sha256']
    dump(output/'normalization_verification.json',dict(status='PASS',runs=len(results),row_sums_checked=True,diagonal_matches_recall=True,copies_checked=len(copies),source_metrics_sha256=sha(output/'results.json'),code_sha256=sha(Path(__file__))))
    print(f'PASS: normalized {len(results)} matrices; plotted 12 seed-42 models; copied {len(copies)} files verified.')

if __name__=='__main__':
    main()
