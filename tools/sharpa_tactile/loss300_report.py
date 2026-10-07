"""Loss convergence, previous-run comparison and publication for loss-stop reruns."""
from .common import project_path, relative_path
import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from .common import ROOT,dump,sha
from .early_warning import write_csv


def report(args):
    out=args.output
    if args.experiment=='critical':
        from .report_critical_reward_final import main
        sys.argv=[sys.argv[0],'--output',str(out)];main()
        previous=project_path('outputs/sharpa_critical_reward_final/20261005_231500')
        target=project_path('WeeklySummary/10.5/critical_reward_final', out.name)
        weekly_name='10.5criticalreward_loss300.md'
    else:
        from .dynamic_threshold import report as base_report
        base_report(args)
        previous=project_path('outputs/sharpa_early_warning_dynamic_threshold/20261006_224500')
        target=project_path('WeeklySummary/10.5/early_warning_dynamic_threshold', out.name)
        weekly_name='10.5threshold_loss300.md'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=json.loads((out/'results.json').read_text());old=json.loads((previous/'results.json').read_text())
    histories=[];comparison=[];variation=[]
    if args.experiment=='critical':
        selected=json.loads((out/'selected_config.json').read_text());n,m=selected['n'],selected['m']
        fig,axes=plt.subplots(7,5,figsize=(20,23),squeeze=False)
        for r in results:
            directory=out/'runs'/f"n{r['n']}_m{r['m']}"/f"seed_{r['seed']}"
            history=json.loads((directory/'history.json').read_text());best=min(history,key=lambda v:v['val_BCE'])
            assert abs(next(v['val_BCE'] for v in history if v['epoch']==r['best_epoch'])-best['val_BCE'])<1e-8
            assert r['epochs']==len(history) and len(history)<=300
            if len(history)<300:assert len(history)-r['best_epoch']>=50
            histories.append(dict(n=r['n'],m=r['m'],seed=r['seed'],fit='critical',epochs=len(history),best_epoch=r['best_epoch'],best_val_loss=best['val_BCE'],hit_max300=len(history)==300))
            ax=axes[[0,5,10,15,20,25,30].index(r['n']),[0,3,5,10,15].index(r['m'])];color=f"C{r['seed']-42}"
            ax.plot([v['epoch'] for v in history],[v['train_BCE'] for v in history],color=color,ls='--',alpha=.5,lw=.8)
            ax.plot([v['epoch'] for v in history],[v['val_BCE'] for v in history],color=color,label=str(r['seed']),lw=1)
            ax.scatter(r['best_epoch'],best['val_BCE'],color=color,s=8);ax.set_title(f"n={r['n']},m={r['m']}");ax.set_xlabel('Epoch');ax.set_ylabel('Unweighted BCE');ax.grid(alpha=.15)
            if r['n']==0 and r['m']==0:ax.legend(fontsize=7,ncol=3)
            previous_r=next(v for v in old if (v['n'],v['m'],v['seed'])==(r['n'],r['m'],r['seed']))
            row=dict(n=r['n'],m=r['m'],seed=r['seed'],old_epochs=previous_r['epochs'],new_epochs=r['epochs'],old_best_epoch=previous_r['best_epoch'],new_best_epoch=r['best_epoch'])
            for phase in ('validation','test','validation_event','test_event'):
                for metric in ('balanced_accuracy','macro_f1','precision','recall','false_positive_rate'):
                    row['old_'+phase+'_'+metric]=previous_r[phase][metric];row['new_'+phase+'_'+metric]=r[phase][metric]
            comparison.append(row)
        fig.suptitle('Critical reward: dashed=train, solid=validation BCE; dot=selected minimum; 5 seeds');fig.tight_layout(rect=(0,0,1,.98));fig.savefig(out/'train_val_loss_all_configs.png',dpi=150);fig.savefig(out/'train_val_loss_all_configs.pdf');plt.close(fig)
        summary=f"Critical reward完整35×5重跑。n/m按原validation BA规则选中n={n},m={m}；epoch按validation BCE，max300/patience50。175runs实际epoch范围{min(r['epochs'] for r in results)}–{max(r['epochs'] for r in results)}，中位数{np.median([r['epochs'] for r in results]):g}；best epoch中位数{np.median([r['best_epoch'] for r in results]):g}。"
        image_names=['train_val_loss_all_configs.png']
    else:
        fig,axes=plt.subplots(5,2,figsize=(13,17),squeeze=False);refit_fig,refit_axes=plt.subplots(5,2,figsize=(13,17),squeeze=False)
        folds=json.loads((out/'folds.json').read_text())['nested_splits']
        sample_meta=json.loads((out/'samples.json').read_text());frame_coordinates={}
        for phase in ('val','test'):
            with np.load(out/(phase+'_coordinates.npz')) as z:frame_coordinates[phase]=z['frames'].copy()
        for split in folds:
            assert not(set(split['inner_train'])&set(split['inner_val']))
            assert not(set(split['outer_train'])&set(split['outer_holdout']))
            assert set(split['inner_train'])|set(split['inner_val'])==set(split['outer_train'])
        for r in results:
            if r['model']=='fixed':continue
            kind,h,seed=r['model'],r['horizon_frames'],r['seed'];directory=out/'runs'/kind/f'H{h}'/f'seed_{seed}'
            history=json.loads((directory/'history.json').read_text());i=[0,8,15,30,45].index(h);j=['mlp','gru'].index(kind);ax=axes[i,j];color=f'C{seed-42}'
            for k,inner in enumerate(history['inner']):
                info=r['fold_selection'][k]['inner'];best=min(inner,key=lambda v:v['val_balanced_BCE'])
                assert abs(next(v['val_balanced_BCE'] for v in inner if v['epoch']==info['best_epoch'])-best['val_balanced_BCE'])<1e-8
                assert len(inner)<=300
                if len(inner)<300:assert len(inner)-info['best_epoch']>=50
                histories.append(dict(horizon_frames=h,model=kind,seed=seed,fit=f'inner{k}',epochs=len(inner),best_epoch=info['best_epoch'],best_val_loss=info['best_val_loss'],hit_max300=len(inner)==300))
                ax.plot([v['epoch'] for v in inner],[v['train_balanced_BCE'] for v in inner],color=color,ls='--',alpha=.25,lw=.7)
                ax.plot([v['epoch'] for v in inner],[v['val_balanced_BCE'] for v in inner],color=color,alpha=.6,lw=.8,label=str(seed) if k==0 else None)
                ax.scatter(info['best_epoch'],info['best_val_loss'],color=color,s=7)
            ax.set_title(f'{kind} H{h} inner loss fits');ax.set_xlabel('Optimizer update');ax.set_ylabel('Balanced BCE');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=5)
            refit=history['refit'];assert len(refit)==int(np.median([v['inner']['best_epoch'] for v in r['fold_selection']]))==r['selected_epochs']
            ra=refit_axes[i,j];ra.plot([v['epoch'] for v in refit],[v['train_balanced_BCE'] for v in refit],color=color,label=str(seed));ra.set_title(f'{kind} H{h} final all-val refit');ra.set_xlabel('Update (duration selected by inner loss)');ra.set_ylabel('Train balanced BCE');ra.grid(alpha=.2);ra.legend(fontsize=7,ncol=5)
            previous_r=next(v for v in old if (v['model'],v['horizon_frames'],v['seed'])==(kind,h,seed));row=dict(horizon_frames=h,model=kind,seed=seed,old_updates=20,new_refit_updates=r['selected_epochs'])
            for phase in ('test','test_event'):
                for metric in ('balanced_accuracy','macro_f1','precision','recall','fpr','pr_auc','roc_auc','correct_first_alarm_rate','positive_band_detection_rate'):
                    if metric in r[phase]:row['old_'+phase+'_'+metric]=previous_r[phase][metric];row['new_'+phase+'_'+metric]=r[phase][metric]
            comparison.append(row)
            for phase in ('val','test'):
                with np.load(directory/(phase+'_predictions.npz')) as z:
                    tau=z['threshold'].copy();offsets=z['offsets'].copy()
                for ii,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
                    ss=sample_meta[phase][ii];local=tau[a:b];active=frame_coordinates[phase][a:b]<ss['key'];active&=np.arange(len(local))>=16
                    variation.append(dict(model=kind,horizon_frames=h,seed=seed,split=phase,rollout_index=ii,failure=int(ss['outcome']==2),range=float(np.ptp(local)),std=float(np.std(local)),prekey_after16_range=float(np.ptp(local[active])) if active.any() else None))
        for fig,name,title in ((fig,'inner_train_val_loss','Nested inner fits: dashed=train, solid=held-out loss; dots=selected epoch'),(refit_fig,'final_refit_train_loss','Final refit: training only; duration chosen by held-out inner losses')):
            fig.suptitle(title);fig.tight_layout(rect=(0,0,1,.98));fig.savefig(out/(name+'.png'),dpi=160);fig.savefig(out/(name+'.pdf'));plt.close(fig)
        write_csv(out/'threshold_variation.csv',variation)
        variation_summary=[]
        for h in (0,8,15,30,45):
            for kind in ('mlp','gru'):
                vv=[v for v in variation if v['split']=='test' and v['failure'] and v['horizon_frames']==h and v['model']==kind]
                variation_summary.append(dict(horizon_frames=h,model=kind,failure_sequences=len(vv),median_range=float(np.median([v['range'] for v in vv])),median_prekey_after16_range=float(np.median([v['prekey_after16_range'] for v in vv if v['prekey_after16_range'] is not None]))))
        write_csv(out/'threshold_variation_summary.csv',variation_summary)
        summary='MLP/GRU全部5H×5seeds重跑，原risk冻结。150个内部loss早停fit、150个outer refit、50个最终refit。内部max300/patience50监控独立rollout balanced BCE；outer/最终refit使用内部loss选定的轮数，不再次用training loss选checkpoint。'
        image_names=['inner_train_val_loss.png','final_refit_train_loss.png']
    write_csv(out/'loss_stopping_audit.csv',histories);write_csv(out/'paired_previous_comparison.csv',comparison)
    text='\n## Max300 / loss-based patience50 重跑与收敛审计\n\n'+summary+'\n\nTrain与validation loss均已保留，最佳loss点标在图中。训练结束不等同于训练loss收敛；可能是validation loss已回升而停止。与旧版逐seed比较见paired_previous_comparison.csv；所有best-epoch/patience条件已回放核对。旧文件与标签未改，test不用于选epoch。\n\n'
    for name in image_names:text+=f'![Loss curves]({name})\n\n'
    with (out/'README.md').open('a') as f:f.write(text)
    shutil.copyfile(Path(__file__),out/'loss300_report.py')
    verification=json.loads((out/'verification.json').read_text());verification.update(loss_based_best_checkpoint_checked=True,patience50_checked=True,max300_checked=True);dump(out/'verification.json',verification)
    checks={}
    for path in out.iterdir():
        if path.is_file():shutil.copyfile(path,target/path.name);assert sha(path)==sha(target/path.name);checks[path.name]=sha(path)
    dump(target/'copy_sha256_loss300.json',checks)
    for manifest_name in ('copy_sha256.json',):
        if (target/manifest_name).exists():dump(target/manifest_name,checks)
    if (target/'copy_manifest.json').exists():dump(target/'copy_manifest.json',[dict(source=str(relative_path(out/name)),destination=str(relative_path(target/name)),sha256=value) for name,value in checks.items()])
    weekly=(out/'README.md').read_text()
    for path in out.glob('*.png'):weekly=weekly.replace(']('+path.name+')',']('+str(target.relative_to(project_path('WeeklySummary/10.5')))+'/'+path.name+')')
    (project_path('WeeklySummary/10.5', weekly_name)).write_text(weekly)
    env=project_path('environment_reports/SHARPA_LOSS300_20261007_134800.md')
    with env.open('a') as f:f.write('\n'+args.experiment+' COMPLETE '+datetime.datetime.now().astimezone().isoformat()+': '+summary+' Metrics/best-loss/patience/copySHA PASS. Report '+str(relative_path(project_path('WeeklySummary/10.5', weekly_name)))+'.\n')
    for name in ('SETUP_STATUS.md','SYSTEM_INFO.txt'):
        with (project_path(name)).open('a') as f:f.write('\nLoss300 '+args.experiment+' COMPLETE '+datetime.datetime.now().astimezone().isoformat()+': '+str(relative_path(out))+'; loss-based checkpoint/patience checks PASS,weeklycopySHA PASS.\n')
    print('LOSS300_REPORT_COMPLETE',args.experiment,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('experiment',choices=('critical','threshold'));p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path,default=project_path('outputs/sharpa_early_warning/20261006_001500'));p.add_argument('--features',type=Path,default=project_path('outputs/sharpa_gaussian_online_data/20261005_182000'));p.add_argument('--device',default='cuda:2');args=p.parse_args()
    for key in ('output','source','features'):setattr(args,key,getattr(args,key).resolve())
    report(args)


if __name__=='__main__':main()
