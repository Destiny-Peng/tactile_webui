"""Paired wrist-camera replacement of the existing high-camera experiment."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .common import ROOT, dump, sha
from .prefix_outcome_data import open_database
from .prefix_outcome_report import configure, report, write_csv

RGB_MODELS=('tactile_rgb','tactile_rgb_pose','rgb_pose')


def prepare(out,base):
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'dataset_manifest.json').exists(), 'Refuse to overwrite prepared data'
    (out/'alignment').mkdir(exist_ok=True)
    data=json.loads((base/'dataset_manifest.json').read_text())
    for r in data['records']:
        with np.load(base/r['alignment_path']) as old:
            a={key:old[key].copy() for key in old.files}
        # As-of receiver timestamp lookup; no device clock or future frame.
        with open_database(r) as conn:
            rows=conn.execute('SELECT receive_mono_ns,frame_index FROM camera_frames WHERE source=? ORDER BY receive_mono_ns,id',('camera:wrist_right',)).fetchall()
        if not rows:raise ValueError(f'Missing wrist video: {r["id"]}')
        camera=np.asarray(rows,np.int64)
        ii=np.searchsorted(camera[:,0],a['tick_mono_ns'],side='right')-1
        assert (ii>=0).all(),f'Wrist absent at original startup: {r["id"]}'
        a['video_frames']=camera[ii,1]
        a['receive_mono_ns'][:,5]=camera[ii,0]
        assert np.all(a['receive_mono_ns']<=a['tick_mono_ns'][:,None])
        partial=out/r['alignment_path']
        if partial.exists():
            with np.load(partial) as previous:
                for key in a:assert np.array_equal(a[key],previous[key]),f'Camera table differs from snapshot: {r["id"]}/{key}'
        age=(a['tick_mono_ns'][:,None]-a['receive_mono_ns'])/1e6
        audit=dict(r['alignment_audit'])
        for field,values in [('maximum_age_ms_by_source',age.max(0)),('median_age_ms_by_source',np.median(age,axis=0))]:
            audit[field]=dict(audit[field])
            audit[field].pop('camera:realsense_color')
            audit[field]['camera:wrist_right']=float(values[5])
        audit.update(camera_alignment='as-of camera_frames.receive_mono_ns on identical baseline ticks',
                     camera_snapshot_equivalence_verified=partial.exists(),
                     held_stale_source_ticks=None,causal_violations=0)
        r.update(rgb_camera_key='cam_wrist',alignment_audit=audit)
        np.savez_compressed(out/r['alignment_path'],**a)
        print('WRIST_ALIGN',r['id'],len(a['ticks']),flush=True)
    dump(out/'dataset_manifest.json',data)
    for name in ('split_manifest.json','input_manifest_snapshot.jsonl','prior_manifest_snapshot.jsonl'):
        shutil.copy2(base/name,out/name)
    protocol=json.loads((base/'data_protocol.json').read_text())
    protocol.update(rgb='cam_wrist / camera:wrist_right only; latest received camera_frames entry <= recorder tick; frozen DINOv2 ViT-S/14 CLS384, official resize256/center224/ImageNet normalization',
                    camera='cam_wrist',comparison_base=str(base),
                    paired_control='identical rollout IDs, split, tick times, tactile features and measured pose; only RGB camera changes')
    dump(out/'data_protocol.json',protocol)
    (out/'features').mkdir()
    (out/'features/tactile').symlink_to(base/'features/tactile',target_is_directory=True)
    (out/'runs').mkdir()
    for kind in ('tactile','tactile_pose'):
        (out/'runs'/kind).symlink_to(base/'runs'/kind,target_is_directory=True)
    configure(out)
    training=json.loads((out/'training_protocol.json').read_text())
    training.update(rgb_camera='cam_wrist',retrained_models=list(RGB_MODELS),reused_non_rgb_models=['tactile','tactile_pose'],
                    comparison_base=str(base),baseline_training_protocol_sha256=sha(base/'training_protocol.json'))
    dump(out/'training_protocol.json',training)
    dump(out/'camera_comparison_protocol.json',dict(base=str(base),replacement_camera='cam_wrist',
         retrained_runs=15,reused_non_rgb_runs=10,seeds=[42,43,44,45,46],
         identical_split_sha256=sha(out/'split_manifest.json'),identical_non_rgb_ticks_and_observations=True))
    print('WRIST_PREPARE_COMPLETE',len(data['records']),flush=True)


def publish(out,base):
    report(out)
    import csv
    def rows(path):
        with path.open() as f:return list(csv.DictReader(f))
    high=rows(base/'metrics_by_seed.csv');wrist=rows(out/'metrics_by_seed.csv')
    comparison=[]
    metrics=('balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_fpr','roc_auc','pr_auc')
    for model in RGB_MODELS:
        for split in ('val','test'):
            for unit in ('rollout_equal_frames','pooled_frames','final_timestep'):
                h=sorted([r for r in high if r['model']==model and r['split']==split and r['unit']==unit],key=lambda r:int(r['seed']))
                w=sorted([r for r in wrist if r['model']==model and r['split']==split and r['unit']==unit],key=lambda r:int(r['seed']))
                assert [r['seed']for r in h]==[r['seed']for r in w]
                for metric in metrics:
                    a=np.array([float(r[metric])for r in h]);b=np.array([float(r[metric])for r in w])
                    comparison.append(dict(model=model,split=split,unit=unit,metric=metric,
                        high_mean=float(a.mean()),high_sd=float(a.std(ddof=1)),wrist_mean=float(b.mean()),wrist_sd=float(b.std(ddof=1)),
                        paired_delta_mean=float((b-a).mean()),paired_delta_sd=float((b-a).std(ddof=1))))
    write_csv(out/'camera_comparison.csv',comparison)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hp=rows(base/'prefix_metrics_by_seed.csv');wp=rows(out/'prefix_metrics_by_seed.csv')
    for phase in ('val','test'):
        fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for ax,model in zip(axs,RGB_MODELS):
            for name,bank in [('High',hp),('Wrist',wp)]:
                means=[];sd=[]
                for percent in (20,40,60,80,100):
                    vals=[float(r['balanced_accuracy']) for r in bank if r['model']==model and r['split']==phase and int(r['prefix_percent'])==percent]
                    means.append(np.mean(vals));sd.append(np.std(vals,ddof=1))
                ax.errorbar((20,40,60,80,100),means,yerr=sd,marker='o',capsize=3,label=name)
            ax.axhline(.5,ls=':',color='gray')
            ax.set(title=model,xlabel='Observed rollout (%)',ylabel='Balanced accuracy',ylim=(0,1.04));ax.legend()
        fig.suptitle(phase+' camera comparison: prefix endpoints, 5 seeds mean ± SD')
        for ext in ('png','pdf'):fig.savefig(out/'figures'/(phase+'_camera_prefix_comparison.'+ext),dpi=180)
        plt.close(fig)
    lines=['# Wrist / High camera 对照','',
        '只替换 RGB 为 cam_wrist（camera:wrist_right）。3个RGB组合×seeds42–46全部重新初始化训练，共15个新模型；Tactile与Tactile+Pose两个不含RGB的组合复用原10个run，未重训。原high camera结果不覆盖。',
        '', '数据保持125条rollout（92success/33failure），train/val/test=87/19/19。逐rollout核对ticks、tactile event IDs、实测pose完全相同；wrist依据自身receive_mono_ns做因果对齐，不套用high相机frame index。其余编码器、GRU、loss、optimizer、earlystop与val阈值选择均沿用原设置。',
        '', '## Test：每rollout等权frame指标，5 seeds mean±SD','',
        '| Model | High BA | Wrist BA | ΔBA（百分点） | Wrist Macro F1 | Wrist failure P / R / FPR |',
        '|---|---:|---:|---:|---:|---:|']
    for model in RGB_MODELS:
        rr={r['metric']:r for r in comparison if r['model']==model and r['split']=='test' and r['unit']=='rollout_equal_frames'}
        ba=rr['balanced_accuracy'];f1=rr['macro_f1']
        prf=' / '.join(f'{rr[k]["wrist_mean"]*100:.2f}%' for k in ('failure_precision','failure_recall','failure_fpr'))
        lines.append(f'| {model} | {ba["high_mean"]*100:.2f}±{ba["high_sd"]*100:.2f}% | {ba["wrist_mean"]*100:.2f}±{ba["wrist_sd"]*100:.2f}% | {ba["paired_delta_mean"]*100:+.2f} | {f1["wrist_mean"]*100:.2f}±{f1["wrist_sd"]*100:.2f}% | {prf} |')
    lines+=['','BA为success/failure recall的平均，不是普通binary accuracy。表中frame权重为每rollout内1/T；完整pooled frame及最后时刻rollout指标另见summary.csv。各模型/seed只在val上选threshold，test固定使用。camera_comparison.csv同时保存val/test三种统计单位、paired seed差值及SD。','',
        '![Test high/wrist prefix comparison](figures/test_camera_prefix_comparison.png)','',
        '![Validation high/wrist prefix comparison](figures/val_camera_prefix_comparison.png)','',
        '不能仅凭这次相机对照消除task/录制批次混杂：failure仍集中于一个task/录制批次。完整loss、逐seed指标、分阶段曲线、归一化confusion matrix和125条完整概率曲线见下面的实验报告。','']
    text='\n'.join(lines)+'\n'+(out/'README.md').read_text()
    (out/'README.md').write_text(text)
    weekly=ROOT.parent/'LF3R/WeeklySummary/10.10';target=weekly/'multimodal_prefix_outcome_wrist'/out.name
    shutil.copy2(out/'README.md',target/'README.md');shutil.copy2(out/'camera_comparison.csv',target/'camera_comparison.csv')
    for name in ('val_camera_prefix_comparison','test_camera_prefix_comparison'):
        for ext in ('png','pdf'):shutil.copy2(out/'figures'/(name+'.'+ext),target/'figures'/(name+'.'+ext))
    (weekly/'10.10multimodal_prefix_outcome_wrist.md').write_text(text.replace('(figures/','(multimodal_prefix_outcome_wrist/'+out.name+'/figures/'))
    print('WRIST_REPORT_COMPLETE',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','publish'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base',type=Path,default=ROOT/'outputs/sharpa_multimodal_prefix_outcome/20261009_223000')
    args=p.parse_args()
    if args.action=='prepare':prepare(args.output.resolve(),args.base.resolve())
    else:publish(args.output.resolve(),args.base.resolve())
