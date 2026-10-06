"""Fixed-cache class-weight ablation followed by five paired training seeds."""
from __future__ import annotations
import argparse
import csv
import datetime
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import numpy as np
import torch
from .common import ROOT, INPUTS, HEADS, dump, sha
from .train import load_sequences
from .train_three import CLASSES, WEIGHT_MODES, class_weights, train_group

SEEDS = (42,43,44,45,46)
MEASURES = ('balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_f1',
            'failure_false_positive_rate','background_to_failure_rate','success_to_failure_rate')


def flat_metrics(m):
    cm = np.asarray(m['confusion_matrix']); fp = int(cm[:2,2].sum())
    return {'balanced_accuracy':m['balanced_accuracy'],'macro_f1':m['macro_f1'],
        **{'failure_'+key:m['per_class']['failure'][key] for key in ('precision','recall','f1')},
        'failure_false_positives':fp,'failure_false_positive_rate':float(fp/cm[:2].sum()),
        'background_to_failure_rate':float(cm[0,2]/cm[0].sum()),
        'success_to_failure_rate':float(cm[1,2]/cm[1].sum())}


def summarize(out, results, config):
    rows = []
    for r in results:
        rows.append({'weight_mode':r['weight_mode'],'seed':r['seed'],'group':r['name'],
            'best_epoch':r['best_epoch'],'validation_balanced_accuracy':r['validation']['balanced_accuracy'],
            **flat_metrics(r['test'])})
    with (out/'per_seed.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    groups = []
    for mode in WEIGHT_MODES:
        for name in [i+'_'+h for i in INPUTS for h in HEADS]:
            chosen = [r for r in rows if r['weight_mode']==mode and r['group']==name]
            if not chosen: continue
            summary = {'weight_mode':mode,'group':name,'n_seeds':len(chosen),'seeds':[r['seed'] for r in chosen]}
            for key in MEASURES:
                values = np.array([r[key] for r in chosen])
                summary[key+'_mean'] = float(values.mean())
                summary[key+'_std'] = float(values.std(ddof=1)) if len(values)>1 else None
            groups.append(summary)
    dump(out/'summary.json',groups)
    with (out/'summary.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(groups[0])); writer.writeheader(); writer.writerows(groups)
    pairs = []
    from scipy.stats import t
    for mode in WEIGHT_MODES:
        fusion = {r['seed']:r for r in rows if r['weight_mode']==mode and r['group']=='f6_deform_lstm'}
        for comparison in ('f6_lstm','deform_lstm','deform_mlp'):
            other = {r['seed']:r for r in rows if r['weight_mode']==mode and r['group']==comparison}
            seeds = sorted(set(fusion)&set(other))
            if not seeds: continue
            for metric in ('balanced_accuracy','macro_f1'):
                differences = np.array([fusion[s][metric]-other[s][metric] for s in seeds])
                std = float(differences.std(ddof=1)) if len(seeds)>1 else None
                half = float(t.ppf(.975,len(seeds)-1)*std/np.sqrt(len(seeds))) if std is not None else None
                mean = float(differences.mean())
                pairs.append({'weight_mode':mode,'comparison':'f6_deform_lstm minus '+comparison,
                    'metric':metric,'seeds':seeds,'n':len(seeds),'mean_delta':mean,'std_delta':std,
                    'wins':int((differences>0).sum()),'ties':int((differences==0).sum()),
                    'ci95_low':mean-half if half is not None else None,'ci95_high':mean+half if half is not None else None,
                    'scope':'paired training-seed t interval on one fixed rollout split; not dataset uncertainty'})
    dump(out/'paired_differences.json',pairs)
    dump(out/'results.json',results)
    complete = len(results)==90 and all(g['n_seeds']==5 for g in groups)
    dump(out/'run_status.json',{'status':'complete' if complete else 'running','completed':len(results),'planned':90,
        'new_trainings':84,'imported_baseline':6,'updated_at':datetime.datetime.now().astimezone().isoformat()})
    if not complete: return
    make_report(out,groups,pairs,config,rows)


def make_report(out,groups,pairs,config,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(3,3,figsize=(16,11))
    names = [i+'_'+h for i in INPUTS for h in HEADS]
    for i,mode in enumerate(WEIGHT_MODES):
        for ax,key,title in zip(axes[i],('balanced_accuracy','failure_precision','failure_false_positive_rate'),
                               ('Balanced accuracy / macro recall','Failure precision','Failure false-positive rate')):
            chosen = [next(g for g in groups if g['weight_mode']==mode and g['group']==name) for name in names]
            means = [g[key+'_mean'] for g in chosen]; stds = [g[key+'_std'] for g in chosen]
            ax.bar(range(6),means,yerr=stds,capsize=3,color=['#2563eb','#60a5fa','#d97706','#fbbf24','#15803d','#86efac'])
            ax.set_xticks(range(6),names,rotation=35,ha='right',fontsize=8); ax.set_ylim(0,1)
            ax.set_title(mode+' | '+title,fontsize=10); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Fixed rollout split | five training seeds 42–46 | mean ± sample SD')
    fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=180); fig.savefig(out/'comparison.pdf'); plt.close(fig)
    table = ['| Weight mode | Group | BA / macro recall | Macro F1 | Failure precision | Failure recall | Failure FPR |',
             '|---|---|---:|---:|---:|---:|---:|']
    def fmt(g,key): return f"{100*g[key+'_mean']:.2f} ± {100*g[key+'_std']:.2f}"
    for g in groups:
        table.append('| '+g['weight_mode']+' | '+g['group']+' | '+' | '.join(fmt(g,k) for k in
            ('balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_false_positive_rate'))+' |')
    pair_table = ['| Loss | Fusion LSTM minus | Metric | Mean delta (pp) | Paired seed 95% t interval (pp) | Wins / 5 |',
                  '|---|---|---|---:|---:|---:|']
    for p in pairs:
        pair_table.append(f"| {p['weight_mode']} | {p['comparison'].split(' minus ')[1]} | {p['metric']} | {100*p['mean_delta']:.2f} | [{100*p['ci95_low']:.2f}, {100*p['ci95_high']:.2f}] | {p['wins']} |")
    seed_table = ['| Mode | Group | BA | Macro F1 | Failure precision | Failure recall | Failure FPR |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['seed']==42:
            seed_table.append('| '+r['weight_mode']+' | '+r['group']+' | '+' | '.join(f'{100*r[k]:.2f}' for k in
                ('balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_false_positive_rate'))+' |')
    weights = '\n'.join(f"- {mode}: {config['class_weights'][mode]}" for mode in WEIGHT_MODES)
    readme = f'''# Class-weight ablation and five repeated seeds

固定三类标签、pretrained encoder、architecture、feature cache 和原始 rollout split。完成三种 loss × 六组模型 × 五个 seeds = 90 组结果，其中 6 组 inverse-frequency seed 42 为已有 baseline 导入，新增 84 个小型 frozen-encoder probe 训练。

## Protocol

- 数据源 `{config['source']}`；split 文件 SHA256 `{config['split_sha256']}`。split seed 始终 42；训练 seeds 42/43/44/45/46 仅改变 trainable 参数初始化、dropout 和训练序列 shuffle，不重新划分数据。
- train/val/test 仍为 80/17/17 rollouts，共 47,915 / 9,794 / 10,207 个有效同步帧。Background=0、Success=1、Failure=2；hard labels，无 Gaussian/soft label/ignore target；padding 按长度排除。
- inverse-frequency：保留原权重 `N/(3*n_c)`；unweighted：全部 1（标准 CE）；sqrt-inverse：`1/sqrt(n_c)` 后除以三类权重算术均值，使 mean=1。权重均仅来自 train counts。PyTorch weighted CE 的 mean 按目标权重总和归一化，整体乘常数不改变 loss。
- 三种 loss 的实际权重（background/success/failure）：
{weights}
- 顺序先 seed 42 的两种新权重，再重复 seeds 43–46；不根据 test 选择 loss 或修改模型。三种 loss 均跑五 seeds，保留 recall/precision/FPR tradeoff。
- 其他设置与 baseline 相同：hidden=128、LSTM 单层单向、simple concat、AdamW lr=.001/weight_decay=.0001、batch 8、clip 1、max 30 epochs/patience 8、FP32/TF32 关闭、train-only normalization/std floor .01。最佳 epoch 由 validation 三类 macro recall 选取；test 仅用于评估。
- GPU jobs 顺序执行；features 仅载入一次，不重新训练或抽取 encoder，不添加依赖。

## Five-seed test results

单位为百分比；`mean ± sample SD`，SD 使用 ddof=1。全部三类的 per-seed precision/recall/F1 和 3×3 confusion matrices 见各组 metrics.json。

{chr(10).join(table)}

Failure FPR = `(background→failure + success→failure)/(全部真实 non-failure 帧)`；failure precision = 真 failure / 全部预测 failure。二者分母不同，均报告。summary.csv 另列 background→failure 与 success→failure 的条件误报率。

[汇总 CSV](summary.csv) · [逐 seed CSV](per_seed.csv) · [对照图](comparison.png) · [PDF](comparison.pdf) · [全部结果](results.json)

## Paired fusion comparisons

在同一 loss、同一 training seed 下做差。Fusion LSTM 与单模态 LSTM 是相同 head 的比较；额外与用户关注的 Deform MLP 作跨 head 比较。差值以百分点计。

{chr(10).join(pair_table)}

这些区间只描述固定 split 下训练随机性；只有 5 seeds，t interval 依赖近似正态假设，不能视为对新数据、跨任务或不同 split 泛化的置信区间，也不作多重比较显著性宣称。逐 seed 胜负和差值见 paired_differences.json。是否存在互补性应同时查看 BA、macro F1 和 failure precision/recall/FPR，不能只挑某个 seed 的最大值。

## Seed 42 class-weight ablation

{chr(10).join(seed_table)}

## Artifacts and reproducibility

`<mode>/seed_<seed>/<group>/` 保存 best.pt、metrics.json、history.json、test_predictions.csv、curves.png。导入的 inverse seed 42 额外含 IMPORT.json，指向未改写的原始 baseline，权重与训练配置校验通过。每个 seed 保存 training_config.json；suite_config.json、verification.json 记录实验设置和检查。

```bash
bash tools/run_sharpa_tactile_ablation.sh weight_seed_ablation --source {config['source']} --output {config['output']} --device cuda:1
```

完整运行日志：`{config['log']}`。入口支持在同一配置下恢复已完成组，配置变化需新输出目录。固定 label、split、encoder identities 和 weight/CE 公式检查通过；前一轮真实在线因果验证继续适用于未改 architecture 的探针，未把本轮统计分析当作在线延迟测试。
'''
    (out/'README.md').write_text(readme)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    args = parser.parse_args(); source = args.source.resolve(); out = args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2); torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    data = json.loads((source/'data_manifest.json').read_text()); assert data['status']=='complete' and data['signature']['num_classes']==3
    split = json.loads((source/'split_manifest.json').read_text()); assert split['seed']==42
    verification = json.loads((source/'verification.json').read_text()); assert all(v=='PASS' for v in verification.values())
    for kind,filename in [('f6','f6_tactile_vqvae.pt'),('deform','sharpa_wave_deform_encoder.pth')]:
        assert sha(ROOT/'checkpoints/T-Rex/encoders'/filename)==data['signature'][kind+'_sha256']
    baseline_config = json.loads((source/'training_config.json').read_text())
    parameters = {key:baseline_config[key] for key in ('epochs','patience','batch_size','hidden','layers','lr','weight_decay')}
    assert parameters=={'epochs':30,'patience':8,'batch_size':8,'hidden':128,'layers':1,'lr':.001,'weight_decay':.0001}
    sequences = {name:load_sequences(source,split[name]) for name in ('train','val','test')}
    counts = np.bincount(np.concatenate([s['labels'] for s in sequences['train']]),minlength=3)
    assert counts.tolist()==[data['split_counts']['train'][key] for key in ('background_rows','success_rows','failure_rows')]
    # Directly validate the formulas and weighted-mean CE normalization.
    np.testing.assert_allclose(class_weights([16,4,1],'sqrt_inverse'),np.array([.25,.5,1])/(1.75/3))
    assert np.array_equal(class_weights(counts,'unweighted'),np.ones(3))
    np.testing.assert_allclose(class_weights(counts),counts.sum()/(3*counts))
    logits = torch.tensor([[1.,2.,3.],[3.,2.,1.],[0.,3.,1.],[0.,1.,2.]])
    target = torch.tensor([0,2,1,0]); weights = torch.tensor(class_weights(counts,'sqrt_inverse'),dtype=torch.float32)
    fn = torch.nn.functional.cross_entropy
    torch.testing.assert_close(fn(logits,target,weight=weights),fn(logits,target,weight=weights*7))
    torch.testing.assert_close(fn(logits,target),fn(logits,target,weight=torch.ones(3)))
    config = {'source':str(source.relative_to(ROOT)),'output':str(out.relative_to(ROOT)),'device':args.device,
        'seeds':list(SEEDS),'split_seed':42,'split_sha256':sha(source/'split_manifest.json'),
        'data_manifest_sha256':sha(source/'data_manifest.json'),'encoder_sha256':{k:data['signature'][k+'_sha256'] for k in ('f6','deform')},
        'parameters':parameters,'class_counts':counts.tolist(),'class_weights':{m:class_weights(counts,m).tolist() for m in WEIGHT_MODES},
        'log':f'logs/sharpa_weight_seeds_{out.name}.log',
        'source_file_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tools/sharpa_tactile').glob('*.py'))}}
    old_config = out/'suite_config.json'
    if old_config.exists() and json.loads(old_config.read_text())!=config: raise ValueError('Changed settings; use new output')
    dump(old_config,config); shutil.copyfile(source/'split_manifest.json',out/'split_manifest.json')
    dump(out/'verification.json',{'source_label_causal_and_padding_checks':'PASS','encoder_weights_unchanged':'PASS',
        'class_counts_and_weight_formulas':'PASS','unweighted_CE_and_weight_scale_equivalence':'PASS',
        'fixed_split_independent_training_seeds':'PASS'})
    results = []
    baseline = json.loads((source/'results.json').read_text()); assert len(baseline)==6
    destination = out/'inverse_frequency/seed_42'; destination.mkdir(parents=True,exist_ok=True)
    for result in baseline:
        assert result['seed']==42 and result['model_config']['num_classes']==3
        np.testing.assert_allclose(result['training_class_weights'],class_weights(counts))
        assert result['training_class_counts']==counts.tolist()
        target_dir = destination/result['name']
        if not target_dir.exists(): shutil.copytree(source/result['name'],target_dir)
        r = {**result,'weight_mode':'inverse_frequency','imported_from':str((source/result['name']).relative_to(ROOT))}
        dump(target_dir/'IMPORT.json',{'source':r['imported_from'],'checkpoint_sha256':sha(source/result['name']/'best.pt'),
            'config_checked':True,'split_sha256':config['split_sha256']})
        dump(target_dir/'metrics.json',r); results.append(r)
    dump(destination/'training_config.json',{**parameters,'seed':42,'weight_mode':'inverse_frequency','imported_baseline':True})
    summarize(out,results,config)
    phases = [(m,42) for m in ('unweighted','sqrt_inverse')]+[(m,s) for m in WEIGHT_MODES for s in SEEDS[1:]]
    for mode,seed in phases:
        run = out/mode/f'seed_{seed}'; run.mkdir(parents=True,exist_ok=True)
        options = SimpleNamespace(**parameters,output=run,seed=seed,device=args.device,weight_mode=mode)
        dump(run/'training_config.json',{**parameters,'seed':seed,'weight_mode':mode,'device':args.device,
            'source':config['source'],'split_sha256':config['split_sha256'],'class_weights':config['class_weights'][mode]})
        for input_kind in INPUTS:
            for head_kind in HEADS:
                name = input_kind+'_'+head_kind; saved = run/name/'metrics.json'
                if saved.exists():
                    result = json.loads(saved.read_text())
                    assert result['seed']==seed and result['weight_mode']==mode and (run/name/'best.pt').exists()
                    print('RESUME',mode,seed,name,flush=True)
                else:
                    print('START',mode,seed,name,flush=True)
                    result = train_group(options,input_kind,head_kind,sequences,data)
                results.append(result); summarize(out,results,config)
        print('LOSS_SEED_COMPLETE',mode,seed,'groups',len(results),flush=True)
    # Check immutable source artifacts after the full sweep.
    assert sha(source/'split_manifest.json')==config['split_sha256']
    assert sha(source/'data_manifest.json')==config['data_manifest_sha256']
    for kind,filename in [('f6','f6_tactile_vqvae.pt'),('deform','sharpa_wave_deform_encoder.pth')]:
        assert sha(ROOT/'checkpoints/T-Rex/encoders'/filename)==config['encoder_sha256'][kind]
    print('WEIGHT_SEED_ABLATION_COMPLETE',len(results),flush=True)


if __name__=='__main__': main()
