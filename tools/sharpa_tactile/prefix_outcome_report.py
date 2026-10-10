"""Verify and publish five-modality causal outcome prediction experiments."""
import argparse
import collections
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .common import ROOT,dump,sha
from .prefix_outcome_train import MODELS,SEEDS,evaluate,calibrate


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def configure(out):
    # One immutable shared configuration used by all modality/seed fits.
    dump(out/'training_protocol.json',dict(models={k:list(v)for k,v in MODELS.items()},seeds=list(SEEDS),
         architecture='Tactile2560->Linear128;RGB384->Linear128;Pose34->MLP(128,ReLU,128);concat->Linear128->single unidirectional GRU128->Linear1->sigmoid',
         encoders='Deform and DINOv2 pretrained encoders frozen, precomputed; modality projections/Pose MLP/fusion/GRU/head trained',
         normalization='train only mean and second moment, each rollout equal weight; std floor.01',
         loss='unweighted BCEWithLogits at every valid tick; average within rollout first, then across rollout minibatch; no class weights or class-balanced sampler',
         optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,batch_size=8,gradient_clip=1),
         earlystop=dict(max_epochs=300,patience=50,criterion='natural validation rollout-mean BCE',min_delta=1e-8,tie='earliest epoch'),
         threshold='one constant per model/seed; maximize validation rollout-equal frame BA over exact unique scores; ties higher threshold; fixed for train/test/stages',
         metrics='primary rollout-equal frame confusion/macro metrics; also pooled frame and last-timestep rollout metrics; every raw curve retained',
         prefix='same causal model emits at every tick; evaluation20/40/60/80/100% positions and20%-width stages; relative progress never enters model',
         split_manifest_sha256=sha(out/'split_manifest.json'),data_manifest_sha256=sha(out/'dataset_manifest.json')))


def report(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                                 recall_score, confusion_matrix, roc_auc_score,
                                 average_precision_score, roc_curve)
    data=json.loads((out/'dataset_manifest.json').read_text());records=data['records']
    from .prefix_outcome_data import DEFORM,DINO
    encoder_signature=json.loads((out/'data_protocol.json').read_text())['encoders']
    assert encoder_signature['deform_sha256']==sha(DEFORM)
    assert encoder_signature['dino_model_sha256']==sha(DINO/'model.safetensors')
    split=json.loads((out/'split_manifest.json').read_text());protocol=json.loads((out/'training_protocol.json').read_text())
    assert protocol['split_manifest_sha256']==sha(out/'split_manifest.json')
    assert protocol['data_manifest_sha256']==sha(out/'dataset_manifest.json')
    parts=[set(split[k]) for k in ('train','val','test')]
    assert not any(parts[i]&parts[j] for i in range(3) for j in range(i+1,3))
    assert set.union(*parts)=={r['id']for r in records}
    dist=[]
    for phase in ('train','val','test'):
        for label in (0,1):
            rr=[r for r in records if r['split']==phase and r['label']==label]
            dist.append(dict(split=phase,class_name='failure' if label else 'success',rollouts=len(rr),
                             timesteps=sum(r['alignment_audit']['usable_ticks']for r in rr)))
    write_csv(out/'data_distribution.csv',dist)
    features_audit=[]
    for r in records:
        with np.load(out/r['alignment_path'])as z:
            assert z['pose'].shape==(len(z['ticks']),34) and np.isfinite(z['pose']).all()
            assert np.all(z['receive_mono_ns']<=z['tick_mono_ns'][:,None])
            for key,dim in (('tactile',2560),('rgb',384)):
                path=out/'features'/key/(r['id']+'.npz')
                with np.load(path)as f:
                    assert np.array_equal(z['ticks'],f['ticks'])
                    assert f['features'].shape==(len(z['ticks']),dim) and np.isfinite(f['features']).all()
                features_audit.append(dict(rollout_id=r['id'],modality=key,timesteps=len(z['ticks']),sha256=sha(path)))
    dump(out/'feature_audit.json',features_audit)
    result_rows=[];metric_rows=[];stage_rows=[];prefix_rows=[];checks=[];banks={}
    for kind in MODELS:
        for seed in SEEDS:
            directory=out/'runs'/kind/f'seed_{seed}'
            result=json.loads((directory/'metrics.json').read_text());history=json.loads((directory/'history.json').read_text())
            assert len(history)==result['epochs']<=300
            vals=np.array([h['val_rollout_mean_BCE']for h in history])
            assert abs(vals[result['best_epoch']-1]-vals.min())<1e-8
            assert result['epochs']==300 or result['epochs']-result['best_epoch']==50
            assert result.get('causal_prefix_consistency_max_probability_delta',0)<2e-5
            assert abs(result['metrics']['val']['rollout_mean_BCE']-result['best_val_loss'])<1e-7
            for phase in ('train','val','test'):
                ss=json.loads((directory/(phase+'_samples.json')).read_text())
                samples=[dict(id=s['rollout_id'],label=s['label'])for s in ss]
                with np.load(directory/(phase+'_predictions.npz'))as z:
                    offsets=z['offsets'];p=z['probabilities'];time=z['time_s']
                    assert set(s['id']for s in samples)==set(split[phase])
                    ps=[p[a:b] for a,b in zip(offsets[:-1],offsets[1:])]
                    for i,s in enumerate(samples):
                        assert s['label']==next(r['label']for r in records if r['id']==s['id'])
                        banks[(kind,seed,s['id'])]=(ps[i].copy(),time[offsets[i]:offsets[i+1]].copy(),result['threshold'])
                replay=evaluate(samples,ps,result['threshold'])
                # Independent metric implementation, in addition to replaying
                # saved outputs through the training evaluation function.
                yy=np.concatenate([np.full(len(local),s['label'])for s,local in zip(samples,ps)])
                pp=np.concatenate(ps);ww=np.concatenate([np.full(len(local),1/len(local))for local in ps])
                pred=pp>=result['threshold'];primary=replay['rollout_equal_frames']
                independent=dict(balanced_accuracy=balanced_accuracy_score(yy,pred,sample_weight=ww),
                    macro_f1=f1_score(yy,pred,average='macro',sample_weight=ww,zero_division=0),
                    failure_precision=precision_score(yy,pred,sample_weight=ww,zero_division=0),
                    failure_recall=recall_score(yy,pred,sample_weight=ww,zero_division=0),
                    roc_auc=roc_auc_score(yy,pp,sample_weight=ww),
                    pr_auc=average_precision_score(yy,pp,sample_weight=ww))
                for key,value in independent.items():assert abs(value-primary[key])<1e-8
                assert np.allclose(confusion_matrix(yy,pred,sample_weight=ww),primary['confusion_matrix'])
                for metric in ('pooled_frames','rollout_equal_frames','final_timestep'):
                    actual=result['metrics'][phase][metric]
                    for k in ('accuracy','balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_fpr','roc_auc','pr_auc'):
                        assert abs(replay[metric][k]-actual[k])<1e-8
                    metric_rows.append(dict(model=kind,seed=seed,split=phase,unit=metric,
                        **{k:actual[k]for k in ('accuracy','balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_fpr','roc_auc','pr_auc')},
                        threshold=result['threshold'],confusion_matrix=json.dumps(actual['confusion_matrix'])))
                for name,m in replay['stages'].items():
                    stage_rows.append(dict(model=kind,seed=seed,split=phase,stage=name,
                        **{k:m[k]for k in ('balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_fpr','roc_auc')}))
                for name,m in replay['prefix_endpoints'].items():
                    prefix_rows.append(dict(model=kind,seed=seed,split=phase,prefix_percent=int(name),
                        **{k:m[k]for k in ('balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_fpr','roc_auc')}))
                if phase=='val':
                    selected,calibration=calibrate(samples,ps)
                    assert abs(selected-result['threshold'])<1e-12
                    fpr,tpr,_=roc_curve(yy,pp,sample_weight=ww,drop_intermediate=False)
                    assert abs(np.max((tpr+1-fpr)/2)-primary['balanced_accuracy'])<1e-8
            result_rows.append(result)
            checks.append(dict(model=kind,seed=seed,metric_replay=True,threshold_replay=True,
                               earlystop_verified=True,checkpoint_sha256=sha(directory/'best.pt')))
    dump(out/'results.json',result_rows);write_csv(out/'metrics_by_seed.csv',metric_rows)
    write_csv(out/'stage_metrics_by_seed.csv',stage_rows);write_csv(out/'prefix_metrics_by_seed.csv',prefix_rows)
    summary=[]
    for kind in MODELS:
        for phase in ('val','test'):
            for unit in ('rollout_equal_frames','pooled_frames','final_timestep'):
                rr=[r for r in metric_rows if r['model']==kind and r['split']==phase and r['unit']==unit]
                row=dict(model=kind,split=phase,unit=unit,seeds=len(rr))
                for key in ('accuracy','balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_fpr','roc_auc','pr_auc','threshold'):
                    row[key+'_mean']=float(np.mean([r[key]for r in rr]));row[key+'_sd']=float(np.std([r[key]for r in rr],ddof=1))
                summary.append(row)
    write_csv(out/'summary.csv',summary)
    figures=out/'figures';figures.mkdir(exist_ok=True);curves=out/'rollout_curves';curves.mkdir(exist_ok=True)
    for phase in ('val','test'):
        fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        for ax,metric in zip(axs,('balanced_accuracy','roc_auc')):
            for kind in MODELS:
                mean=[];std=[]
                for percent in (20,40,60,80,100):
                    rr=[r[metric]for r in prefix_rows if r['model']==kind and r['split']==phase and r['prefix_percent']==percent]
                    mean.append(np.mean(rr));std.append(np.std(rr,ddof=1))
                ax.errorbar((20,40,60,80,100),mean,yerr=std,marker='o',label=kind,capsize=2)
            ax.set(xlabel='Observed rollout (%)',ylabel=metric,ylim=(0,1.04));ax.axhline(.5,color='gray',ls=':');ax.legend(fontsize=7)
        fig.suptitle(phase+' prefix endpoints: same causal model, 5 seeds mean ± SD')
        for ext in ('png','pdf'):fig.savefig(figures/(phase+'_prefix_curves.'+ext),dpi=180)
        plt.close(fig)
        fig,axs=plt.subplots(1,5,figsize=(17,3.5),layout='constrained')
        for ax,kind in zip(axs,MODELS):
            cm=np.mean([np.array(r['metrics'][phase]['pooled_frames']['confusion_matrix'])for r in result_rows if r['model']==kind],axis=0)
            norm=cm/cm.sum(1,keepdims=True);ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
            for i in (0,1):
                for j in (0,1):ax.text(j,i,f'{norm[i,j]*100:.1f}%\n({cm[i,j]:.1f})',ha='center',va='center',fontsize=8)
            ax.set(title=kind,xticks=[0,1],yticks=[0,1],xticklabels=['Success','Failure'],yticklabels=['Success','Failure'],xlabel='Prediction',ylabel='GT')
        fig.suptitle(phase+' frame confusion: GT-row normalized, mean raw counts in parentheses')
        for ext in ('png','pdf'):fig.savefig(figures/(phase+'_confusions.'+ext),dpi=180)
        plt.close(fig)
    # Every known-GT rollout, including train, receives its own complete curve.
    curve_index=[]
    for r in records:
        fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
        for kind in MODELS:
            bank=[banks[(kind,seed,r['id'])]for seed in SEEDS]
            time=bank[0][1];pp=np.stack([v[0]for v in bank]);mean=pp.mean(0);std=pp.std(0,ddof=1)
            ax.plot(time,mean,label=kind);ax.fill_between(time,np.clip(mean-std,0,1),np.clip(mean+std,0,1),alpha=.12)
        ax.axhline(r['label'],color='black',ls=':',label='Final outcome GT')
        ax.set(xlabel='Recorded elapsed time (s)',ylabel='P(failure)',ylim=(-.04,1.04),title=f'{r["split"]}: {r["id"]}')
        ax.legend(fontsize=7)
        fig.savefig(curves/(r['id']+'.png'),dpi=150);plt.close(fig)
        curve_index.append(dict(rollout_id=r['id'],split=r['split'],label=r['label'],figure='rollout_curves/'+r['id']+'.png'))
    write_csv(out/'rollout_curve_index.csv',curve_index)
    fig,axs=plt.subplots(1,5,figsize=(18,3.5),layout='constrained')
    for ax,kind in zip(axs,MODELS):
        for seed in SEEDS:
            hh=json.loads((out/'runs'/kind/f'seed_{seed}'/'history.json').read_text())
            ax.plot([h['epoch']for h in hh],[h['train_rollout_mean_BCE']for h in hh],color='C0',alpha=.35)
            ax.plot([h['epoch']for h in hh],[h['val_rollout_mean_BCE']for h in hh],color='C1',alpha=.35)
        ax.set(title=kind,xlabel='Epoch',ylabel='Rollout-mean BCE')
    fig.suptitle('Training (blue) / validation (orange) loss; max300, validation loss patience50')
    for ext in ('png','pdf'):fig.savefig(figures/('loss_curves.'+ext),dpi=180)
    plt.close(fig)
    dump(out/'verification.json',dict(status='complete',models=5,seeds=5,runs=25,rollouts=len(records),
         all_rollouts_have_curves=True,curves=len(curve_index),cached_modalities_verified=len(features_audit),
         causal_alignment_verified=True,no_cross_split_overlap=True,gt_uses_annotations=False,
         all_metrics_and_thresholds_replayed=True,independent_sklearn_metric_and_threshold_check=True,
         earlystop_verified=True,retrained_runs=15 if json.loads((out/'data_protocol.json').read_text()).get('camera')=='cam_wrist' else 25,checks=checks))
    text=make_readme(out,data,dist,summary,result_rows)
    (out/'README.md').write_text(text)
    wrist=json.loads((out/'data_protocol.json').read_text()).get('camera')=='cam_wrist'
    week='10.10' if wrist else '10.9'
    category='multimodal_prefix_outcome_wrist' if wrist else 'multimodal_prefix_outcome'
    weekly=ROOT.parent/'LF3R/WeeklySummary'/week;weekly.mkdir(parents=True,exist_ok=True)
    target=weekly/category/out.name;target.mkdir(parents=True,exist_ok=True)
    for name in ('README.md','summary.csv','metrics_by_seed.csv','stage_metrics_by_seed.csv','prefix_metrics_by_seed.csv',
                 'data_distribution.csv','data_protocol.json','training_protocol.json','split_manifest.json','verification.json','rollout_curve_index.csv'):
        shutil.copy2(out/name,target/name)
    shutil.copytree(figures,target/'figures',dirs_exist_ok=True)
    (weekly/(week+category+'.md')).write_text(text.replace('(figures/','('+category+'/'+out.name+'/figures/'))
    print('REPORT_COMPLETE',out,flush=True)


def make_readme(out,data,dist,summary,results):
    lines=['# SHARPA Tactile / RGB / Pose：causal prefix outcome prediction','',
      '## 数据与GT','',
      f'来源为当前原始SQLite manifest。共{len(data["records"])}条可用且已有最终outcome的rollout；另外{len(data["excluded"])}条因缺少最终outcome或Discard排除（dataset_manifest.json逐条列出）。既有125条结果为92success/33failure；当前manifest缺失的结果只回查既有manifest，不读取人工interval annotation，不推断无结果的新数据。source outcome既有filename推断的来源会原样标注于gt_provenance，不能将它描述为全部经过人工成功审核。','',
      'Success=0、Failure=1，整条rollout所有有效timestep继承最终结果。完整记录从首个模态齐备的因果tick到记录末尾，没有标注区间裁切、Key或background分类。按task×final outcome固定seed42做约70/15/15 rollout split，所有模型及训练seed42–46共享；同一rollout不跨split。','',
      '| Split | Outcome | Rollouts | Timesteps |','|---|---|---:|---:|']
    for r in dist:lines.append(f'| {r["split"]} | {r["class_name"]} | {r["rollouts"]} | {r["timesteps"]} |')
    lines+=['','## 因果对齐与编码','',
      '使用recorder `tick_mono_ns` 与各source的 `receive_mono_ns`，避免混用设备SDK、相机或控制器时钟。每tick只取receive≤tick的最新valid快照；缺失或stale时延用过去已观察值，绝不向未来取样或插值。启动时还没有完整模态的tick单独统计并排除。最大/中位延迟在alignment_audit中报告；不能把过期held观测称为同步新观测。所有模型使用完全相同的tick集合。','',
      'Tactile：现有冻结DeformEncoder，5指各灰度[1,240,240]、float32数值0–255；encoder输出经每指2×2平均池化为512D，拼成2560D缓存。RGB：冻结DINOv2 ViT-S/14，cam_high当前已接收帧，官方Resize短边256、center crop224、ImageNet归一化，缓存CLS384D。缓存本身不含训练后128D投影；两者各自训练Linear→128D。已存在Deform缓存只在checkpoint SHA、原始tick时间与5指event IDs吻合时复用，其余tick重新编码，不继承F6的16tick warmup/interval裁剪。','',
      'Pose为34D：实测右TCP xyz+rotation vector6D、实际机械臂关节6D、实测右手关节角22D；不用operator mocap或target action。train-only normalization后MLP(34→128→ReLU→128)。Tactile/RGB latent同样仅由train估计mean/std（每rollout等权，std≥0.01）。原始传感器/视频不修改。','',
      '[DINOv2官方模型](https://github.com/facebookresearch/dinov2)；[官方ViT-S/14配置](https://huggingface.co/facebook/dinov2-small/blob/main/config.json)及[预处理配置](https://huggingface.co/facebook/dinov2-small/blob/main/preprocessor_config.json)。','',
      '## 模型、loss与阈值','',
      '五组分别为Tactile、Tactile+RGB、Tactile+Pose、Tactile+RGB+Pose、RGB+Pose。模态128D特征拼接→Linear128→单层单向GRU128→Linear1→sigmoid P(final failure | observed history)。每rollout重置hidden；全序列训练、逐时刻输出，右侧padding用mask从loss排除。没有未来观测、最终长度或相对progress输入。冻结图像encoder不参与下游反传，projection/Pose MLP/fusion/GRU/head训练。','',
      '`loss_i = mean_t BCEWithLogits(logit_it, final_outcome_i)`，`loss = mean_i loss_i`。普通unweighted BCE，没有按帧数量给长rollout更高权重，没有class weights，也没有balanced class sampler。AdamW lr0.001、wd0.0001、batch8、clip1；max300、patience50，按自然validation rollout-mean BCE选best epoch，min_delta1e-8。loss曲线和停止epoch逐run保存。','',
      '每个模型/seed在其best checkpoint的validation预测上搜索一个恒定分类threshold，最大化rollout-equal frame BA，平手取较高threshold；再固定到test及全部阶段，不在test调参。原始P(failure)和使用该threshold的binary预测都保存。seed42额外核对完整输入前缀输出与截断输入输出一致，验证GRU因果性。','',
      '## 主要结果：Test，每rollout等权的frame指标，5seeds mean±sample SD','',
      '| Model | BA | Macro F1 | Failure precision | Failure recall | Failure FPR | ROC-AUC |','|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['split']!='test' or r['unit']!='rollout_equal_frames':continue
        values=[f'{r[k+"_mean"]*100:.2f}±{r[k+"_sd"]*100:.2f}%'for k in ('balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_fpr','roc_auc')]
        lines.append('| '+r['model']+' | '+' | '.join(values)+' |')
    lines+=['','## Test：标准pooled frame指标','',
      '| Model | BA | Macro F1 | Failure precision | Failure recall | Failure FPR |','|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['split']!='test' or r['unit']!='pooled_frames':continue
        values=[f'{r[k+"_mean"]*100:.2f}±{r[k+"_sd"]*100:.2f}%'for k in ('balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_fpr')]
        lines.append('| '+r['model']+' | '+' | '.join(values)+' |')
    lines+=['','## Test：最后时刻的rollout最终判断','',
      '| Model | BA | Macro F1 | Failure precision | Failure recall | Failure FPR |','|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['split']!='test' or r['unit']!='final_timestep':continue
        values=[f'{r[k+"_mean"]*100:.2f}±{r[k+"_sd"]*100:.2f}%'for k in ('balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_fpr')]
        lines.append('| '+r['model']+' | '+' | '.join(values)+' |')
    lines+=['','主要frame指标先使每条rollout的frame权重为1/T，再累积2×2confusion matrix；因此每rollout总权重1。BA为两类recall均值，Macro F1为两类F1均值。Failure precision/recall/FPR均针对failure=1。另行保存直接汇总全部frame的pooled指标（长轨迹权重更高）及每rollout最后一帧的event指标，不能混用。ROC/PR-AUC使用相应权重和原始连续概率。图中confusion matrix为pooled frame按GT行归一化，括号为5seed平均原始计数；原始及rollout-equal矩阵均在metrics_by_seed.csv/results.json。','',
      '![Test confusion matrices](figures/test_confusions.png)','',
      '## 时序阶段与完整曲线','',
      '同一个causal模型在各tick都产生预测。统计20/40/60/80/100% prefix endpoint（每rollout一票），并统计五个连续20%阶段内的rollout-equal frame指标。相对长度只用于离线评估和作图，不作为模型输入，不用于在线决定何时预测。','',
      '![Test prefix](figures/test_prefix_curves.png)','',
      '![Validation prefix](figures/val_prefix_curves.png)','',
      'rollout_curves/覆盖全部可用rollout，包括train，并分别注明split；每图为五组P(failure)的5seeds mean±SD及最终GT。这些是完整causal probability curves，不是只保存几个prefix端点。逐seed曲线及最终threshold可由runs/*/*/{train,val,test}_predictions.npz和csv.gz重建；表中指标按各seed自身threshold计算，不能用mean curve重新阈值化替代。','',
      '![Loss curves](figures/loss_curves.png)','',
      '## 限制与可复现交付','',
      '部分outcome来源是原录制命名已有的success/mixedfail推断，而非独立人工审核。没有final outcome的数据不会被偷偷补成success/failure；若后续补齐其既有结果，应重建统一split并全部重训。failure集中于一个task/录制批次，RGB/Pose可以学到task、环境或录制批次差异；本实验不单独证明是触觉接触机制或异常因果证据。train/val/test按rollout隔离不能排除批次间相关。','',
      '检查points为25个runs/<model>/seed_<n>/best.pt，连同history/metrics和逐帧NPZ/CSV.GZ；dataset_manifest、split、data_protocol、training_protocol包含完整配置与GT/encoder来源。summary.csv为五组汇总，metrics_by_seed.csv为逐seed指标，stage_metrics_by_seed.csv/prefix_metrics_by_seed.csv为时序结果。verification.json覆盖全部25run的metric/threshold replay、earlystop、因果对齐和数据/特征完整性。','',
      f'原始结果：`{out}`。准备/提取/训练/报告分别使用tools/sharpa_tactile/prefix_outcome_data.py、prefix_outcome_train.py、prefix_outcome_report.py。']
    text='\n'.join(lines)+'\n'
    if json.loads((out/'data_protocol.json').read_text()).get('camera')=='cam_wrist':
        text=text.replace('cam_high当前已接收帧','cam_wrist（camera:wrist_right）当前已接收帧')
        text=text.replace('检查points为25个runs/', '包含15个重新训练的RGB run与10个复用的非RGB run，检查points为25个runs/')
    return text


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('configure','report'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();globals()[args.action](args.output.resolve())
