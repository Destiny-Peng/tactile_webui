"""Validation-selected two-threshold causal event decisions, no model training."""
from .common import project_path, relative_path
import argparse
import csv
import datetime
import json
import shutil
from pathlib import Path
import numpy as np
from .common import ROOT,dump,sha
from .merged_online import load


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def decision_quantities(p):
    decided=p[:,1]+p[:,2]
    conditional=np.divide(p[:,2],decided,out=np.full_like(decided,.5),where=decided>0)
    return decided,conditional


def decisions(samples,tau_decide,tau_fail,mode='first'):
    result=[]
    for s in samples:
        d,f=decision_quantities(s['probabilities']);instant=np.where(d<tau_decide,0,np.where(f>tau_fail,2,1))
        eligible=np.flatnonzero((d>=tau_decide)&(d>0))
        index=int(eligible[0]) if len(eligible) else None
        prediction=int(instant[index]) if index is not None else 0
        if mode=='last':prediction=int(instant[-1]) if d[-1]>0 else 0
        r=dict(interaction_id=s['interaction_id'],rollout_id=s['rollout_id'],outcome=s['outcome'],prediction=prediction,decided=prediction!=0,first_trigger_index=index,key_frame=s['key'],start_frame=s['start'],observation_end_frame=s['end'],last_valid_frame=int(s['frames'][-1]),last_p_decided=float(d[-1]),last_p_failure_given_decided=float(f[-1]))
        r.update(trigger_frame=int(s['frames'][index]) if index is not None else None,trigger_relative_seconds=float((s['frames'][index]-s['key'])/30) if index is not None else None,trigger_p_decided=float(d[index]) if index is not None else None,trigger_p_failure_given_decided=float(f[index]) if index is not None else None,correct=prediction==s['outcome'])
        # First-trigger decisions are latched; changes below describe unlatched instantaneous evidence.
        active=instant[index:] if index is not None else np.array([],dtype=int)
        r['instantaneous_class_changes_after_trigger']=int((np.diff(active)!=0).sum())
        r['opposite_outcome_seen_after_trigger']=bool(np.any(active==3-int(instant[index]))) if index is not None else False
        r['returned_to_progress_after_trigger']=bool(np.any(active==0)) if index is not None else False
        result.append(r)
    return result


def metrics(rows):
    y=np.array([r['outcome'] for r in rows]);pred=np.array([r['prediction'] for r in rows]);cm=np.array([[int(((y==c)&(pred==p)).sum()) for p in (0,1,2)] for c in (1,2)])
    per_class={}
    for c,name in ((1,'success'),(2,'failure')):
        tp=int(((y==c)&(pred==c)).sum());fp=int(((y!=c)&(pred==c)).sum());fn=int(((y==c)&(pred!=c)).sum());support=int((y==c).sum());precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/support
        per_class[name]=dict(precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,support=support,undecided_rate=float((pred[y==c]==0).mean()))
    def timing(rr):
        all_trigger=[r['trigger_relative_seconds'] for r in rr if r['trigger_relative_seconds'] is not None];correct_trigger=[r['trigger_relative_seconds'] for r in rr if r['correct'] and r['trigger_relative_seconds'] is not None]
        def summarize(values):
            return dict(n=len(values),mean_seconds=float(np.mean(values)),median_seconds=float(np.median(values)),p25_seconds=float(np.percentile(values,25)),p75_seconds=float(np.percentile(values,75)),before_key_rate=float((np.array(values)<0).mean())) if values else dict(n=0,mean_seconds=None,median_seconds=None,p25_seconds=None,p75_seconds=None,before_key_rate=None)
        return dict(all_decisions=summarize(all_trigger),correct_decisions=summarize(correct_trigger))
    decided=pred!=0
    return dict(n=len(rows),event_accuracy=float((y==pred).mean()),success_event_accuracy=per_class['success']['recall'],failure_event_accuracy=per_class['failure']['recall'],balanced_event_accuracy=float(np.mean([v['recall'] for v in per_class.values()])),macro_f1=float(np.mean([v['f1'] for v in per_class.values()])),failure_precision=per_class['failure']['precision'],failure_recall=per_class['failure']['recall'],failure_false_positive_rate=float((pred[y==1]==2).mean()),undecided_rate=float((pred==0).mean()),coverage=float(decided.mean()),decided_only_accuracy=float((y[decided]==pred[decided]).mean()) if decided.any() else None,per_class=per_class,confusion_matrix=cm.tolist(),confusion_gt_order=['success','failure'],confusion_prediction_order=['undecided','success','failure'],detection_time=dict(overall=timing(rows),success=timing([r for r in rows if r['outcome']==1]),failure=timing([r for r in rows if r['outcome']==2])),returned_to_progress_after_trigger_rate=float(np.mean([r['returned_to_progress_after_trigger'] for r in rows if r['first_trigger_index'] is not None])) if decided.any() else None,opposite_outcome_seen_after_trigger_rate=float(np.mean([r['opposite_outcome_seen_after_trigger'] for r in rows if r['first_trigger_index'] is not None])) if decided.any() else None)


def plot_probability_summary(out,summary_rows,td,tf):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(13,10));layout=fig.add_gridspec(4,2,height_ratios=[4,1,4,1],hspace=.48,wspace=.23)
    lo=min(r['relative_time_seconds'] for r in summary_rows);hi=max(r['relative_time_seconds'] for r in summary_rows)
    for si,split in enumerate(('val','test')):
        for oi,name in enumerate(('success','failure')):
            ax=fig.add_subplot(layout[2*si,oi]);support=fig.add_subplot(layout[2*si+1,oi],sharex=ax)
            for quantity,label,color in (('p_decided','P(decided)','tab:purple'),('p_failure_given_decided','P(failure | decided)','tab:orange')):
                rr=[r for r in summary_rows if r['split']==split and r['gt_outcome']==name and r['quantity']==quantity]
                x=np.array([r['relative_time_seconds'] for r in rr]);mean=np.array([r['mean'] for r in rr]);sd=np.array([r['std'] for r in rr])
                ax.plot(x,mean,color=color,label=label);ax.fill_between(x,np.clip(mean-sd,0,1),np.clip(mean+sd,0,1),color=color,alpha=.13)
            n=np.array([r['interaction_count'] for r in rr]);support.plot(x,n,color='black',linewidth=1);support.set_ylim(bottom=0);support.set_ylabel('N');support.set_xlabel('Time relative to Key (s)')
            ax.axhline(td,color='tab:purple',linestyle='--',label=f'tau_decide={td:.2f}');ax.axhline(tf,color='tab:orange',linestyle='--',label=f'tau_fail={tf:.2f}')
            for a in (ax,support):a.axvline(0,color='gray',linestyle=':');a.set_xlim(lo,hi);a.grid(alpha=.15)
            ax.set_ylim(0,1);ax.tick_params(labelbottom=False);ax.set_ylabel('Mean probability ± SD');ax.set_title(f'{split.upper()} | {name.title()} GT | total N={int(n.max())}')
            if si==oi==0:ax.legend(fontsize=8)
    fig.suptitle('Online decision evidence | original Align / Deform GRU / sigma8 / seed42');fig.subplots_adjust(top=.93,bottom=.06)
    fig.savefig(out/'key_relative_decision_probabilities.png',dpi=170);fig.savefig(out/'key_relative_decision_probabilities.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=project_path('outputs/sharpa_gaussian_online/20261005_182000');dataset=project_path('outputs/sharpa_gaussian_online_data/20261005_182000');dm=json.loads((dataset/'dataset_manifest.json').read_text());directory=source/'runs/original_align/deform_gru/sigma_8/seed_42'
    prediction_source=source/'key_aligned_20261005_184000/predictions/original_align/deform_gru';samples={};hashes={str(relative_path(p)):sha(p) for p in (dataset/'dataset_manifest.json',dataset/'split_manifest.json',directory/'best.pt',Path(__file__))}
    grid=np.linspace(0,1,21)
    dump(out/'protocol.json',dict(group='original_align',input='deform',head='gru',sigma=8,seed=42,checkpoint=str(relative_path(directory/'best.pt')),threshold_grid=grid.tolist(),selection='validation first-trigger balanced event accuracy; undecided count wrong. Tie: macroF1, lower undecided, lower failureFPR, higher tau_decide, tau_fail closer .5, smaller tau_fail',primary='first P(decided)>=tau_decide with positive decided mass; classify failure iff P(failure|decided)>tau_fail; latch until new interaction',secondary='last valid frame instantaneous classification; same thresholds; not selection',no_training=True))
    for split in ('val','test'):
        ss=[r for r in dm['records'] if r['group']=='original_align' and r['split']==split];p=load(prediction_source/f'{split}_seed_42.npz');path=prediction_source/f'{split}_seed_42.npz';hashes[str(relative_path(path))]=sha(path);samples[split]=[]
        for i,r in enumerate(ss):
            path=dataset/r['feature_path'];data=load(path);hashes[str(relative_path(path))]=sha(path);a,b=p['offsets'][i:i+2];assert len(data['ticks'])==b-a
            samples[split].append(dict(r,probabilities=p['probabilities'][a:b],frames=data['video_frames']))
    # Only validation is accessed for threshold search; no test metric or timing is in ranking.
    sweep=[]
    for td in grid:
        for tf in grid:
            m=metrics(decisions(samples['val'],float(td),float(tf)));sweep.append(dict(tau_decide=float(td),tau_fail=float(tf),**{k:m[k] for k in ('event_accuracy','balanced_event_accuracy','macro_f1','failure_precision','failure_recall','failure_false_positive_rate','undecided_rate')}))
    def rank(r):return (r['balanced_event_accuracy'],r['macro_f1'],-r['undecided_rate'],-r['failure_false_positive_rate'],r['tau_decide'],-abs(r['tau_fail']-.5),-r['tau_fail'])
    selected=max(sweep,key=rank);dump(out/'selected_thresholds.json',selected);write_csv(out/'validation_threshold_sweep.csv',sweep);td=selected['tau_decide'];tf=selected['tau_fail'];all_metrics={};curves=[]
    for split,ss in samples.items():
        rows=decisions(ss,td,tf);m=metrics(rows);last_metrics=metrics(decisions(ss,td,tf,'last'));last_metrics.pop('detection_time');all_metrics[split]=dict(first_trigger=m,last_frame_instantaneous=last_metrics)
        write_csv(out/f'{split}_events.csv',rows)
        for s in ss:
            d,f=decision_quantities(s['probabilities'])
            for j in range(len(d)):
                curves.append(dict(split=split,interaction_id=s['interaction_id'],rollout_id=s['rollout_id'],outcome=s['outcome'],current_frame=int(s['frames'][j]),relative_time_seconds=float((s['frames'][j]-s['key'])/30),p_progress=float(s['probabilities'][j,0]),p_success=float(s['probabilities'][j,1]),p_failure=float(s['probabilities'][j,2]),p_decided=float(d[j]),p_failure_given_decided=float(f[j]),instantaneous_prediction=0 if d[j]<td or d[j]==0 else 2 if f[j]>tf else 1))
        # Validate saved decisions and timing by directly reapplying the causal rule at each prefix.
        for s,r in zip(ss,rows):
            d,f=decision_quantities(s['probabilities']);prefix_found=None
            for j in range(len(d)):
                if d[j]>=td and d[j]>0:prefix_found=j;break
            assert r['first_trigger_index']==prefix_found
            if prefix_found is not None:assert r['prediction']==(2 if f[prefix_found]>tf else 1)
            else:assert r['prediction']==0
        cm=np.array(m['confusion_matrix']);assert cm.sum()==len(ss) and np.isclose(m['undecided_rate'],cm[:,0].sum()/len(ss))
    dump(out/'metrics.json',all_metrics);write_csv(out/'probability_curves.csv',curves)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary_rows=[]
    for si,split in enumerate(('val','test')):
        ss=samples[split];lo=min(int(s['frames'].min()-s['key']) for s in ss);hi=max(int(s['frames'].max()-s['key']) for s in ss);frames=np.arange(lo,hi+1)
        for oi,outcome in enumerate((1,2)):
            members=[s for s in ss if s['outcome']==outcome];table=np.full((len(members),len(frames),2),np.nan)
            for i,s in enumerate(members):
                # Transform every tick first, then average duplicates: mean conditional != ratio of mean probabilities.
                d,f=decision_quantities(s['probabilities']);relative=s['frames']-s['key']
                for frame in np.unique(relative):table[i,int(frame-lo)]=[d[relative==frame].mean(),f[relative==frame].mean()]
            mean=np.full((len(frames),2),np.nan);sd=mean.copy();counts=np.isfinite(table[:,:,0]).sum(0)
            for j,n in enumerate(counts):
                values=table[:,j];values=values[np.isfinite(values[:,0])]
                if n:mean[j]=values.mean(0)
                if n>1:sd[j]=values.std(0,ddof=1)
                for k,name in enumerate(('p_decided','p_failure_given_decided')):summary_rows.append(dict(split=split,gt_outcome='success' if outcome==1 else 'failure',relative_time_seconds=float(frames[j]/30),quantity=name,mean=float(mean[j,k]),std=float(sd[j,k]),variance=float(sd[j,k]**2),interaction_count=int(n)))
    write_csv(out/'key_relative_mean_variance.csv',summary_rows);plot_probability_summary(out,summary_rows,td,tf)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,split in zip(axes,('val','test')):
        rows=decisions(samples[split],td,tf)
        for outcome,marker,color in ((1,'o','tab:blue'),(2,'x','tab:orange')):
            selected_rows=[r for r in rows if r['outcome']==outcome and r['first_trigger_index'] is not None];x=[r['trigger_relative_seconds'] for r in selected_rows];y=np.arange(len(selected_rows));ax.scatter(x,y,marker=marker,color=color,label=('Success' if outcome==1 else 'Failure')+' GT')
            for xx,yy,r in zip(x,y,selected_rows):ax.annotate('✓' if r['correct'] else 'wrong',(xx,yy),xytext=(5,2),textcoords='offset points',fontsize=7)
        ax.axvline(0,color='gray',linestyle=':');ax.set_title(split.upper()+' first-trigger times');ax.set_xlabel('Time relative to Key (s)');ax.set_ylabel('Within-outcome interaction index');ax.legend()
    fig.tight_layout();fig.savefig(out/'event_detection_times.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));heat=np.array([r['balanced_event_accuracy'] for r in sweep]).reshape(21,21);im=ax.imshow(heat,origin='lower',extent=(-.025,1.025,-.025,1.025),aspect='auto',vmin=0,vmax=1);ax.scatter(tf,td,marker='x',color='red');ax.set_xlabel('tau_fail');ax.set_ylabel('tau_decide');ax.set_title('Validation balanced EVENT accuracy');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'validation_threshold_search.png',dpi=170);plt.close(fig)
    assert selected==max(sweep,key=rank) and all(sha(project_path(p))==v for p,v in hashes.items())
    dump(out/'verification.json',dict(status='PASS',no_training=True,thresholds_validation_only=True,fixed_sigma8_seed42=True,all_val22_test24_interactions=True,undecided_not_removed_from_accuracy=True,first_trigger_causal_recomputed=True,source_hashes_unchanged=True))
    dump(out/'manifest.json',dict(created_at=datetime.datetime.now().astimezone().isoformat(),source=str(relative_path(source)),prediction_source=str(relative_path(prediction_source)),validation='previous CPU checkpoint inference; no selection rerun of sigma/epoch',test='original saved GPU probabilities',input_hashes=hashes))
    lines=['# Online decision evaluation：双阈值事件检测','', '固定original_align / Deform GRU / σ=8 / seed42 / 原rollout split与已有checkpoint(epoch15)。没有重新训练，没有σ sweep，没有step augmentation。复用已有完整validation/test预测：val22Align(14success/8failure)、test24Align(15success/9failure)，都来自各自17rollout。相同rollout内Align不是独立rollout，数据量较小。','',
    '## 概率与在线决策','', 'P(decided)=P(success)+P(failure)=1−P(progress)；P(failure|decided)=P(failure)/P(decided)。该条件概率只是原三类输出的重新归一化，不代表经过概率校准。P(decided)=0时条件概率定义为.5供数值保存，但不允许触发。低decided质量时conditional可能波动，不能只看conditional曲线判断已有证据。','',
    '逐有效tick执行：P(decided)<τ_decide→progress；否则conditional>τ_fail→failure，等于阈值按success。**主要event评估采用首次达到τ_decide时触发并锁定outcome**，换interaction重置。先后顺序按原有效tick，不使用未来数据、Key位置或GT控制触发。没有额外平滑、持续K帧、hysteresis或改判策略。首次决定错误但尾段后来正确仍计错误；直到已标注观察尾段结束未触发为undecided。按用户先前要求保留interaction已标注尾段，原始Align最多到下一个Align开始前，因此这里“结束”指该观察范围末尾，不一定等于Align Key。','',
    '原if规则也可以逐时刻改判，因此另报告last_frame_instantaneous指标：最后一个有效tick按同阈值重新分类。它与首次事件决定是不同口径，不能混为一个最终accuracy；阈值只为首次触发任务选择。metrics.json还报告触发后曾回到progress、出现相反outcome的比例，用于衡量未经锁定的输出稳定性。','',
    '## Validation选择','',f'τ_decide和τ_fail各搜索0到1，步长.05，共441对；选中τ_decide={td:.2f}、τ_fail={tf:.2f}。主目标validation balanced event accuracy=(success正确率+failure正确率)/2，undecided计错误。平分依次取macroF1较高、undecided较低、failureFPR较低、τ_decide较高、τ_fail离.5较近、τ_fail较小。不使用test、不按GT Key前后时间筛阈值、不按decided-only accuracy筛选。','',
    '## Event指标口径','', '每条interaction贡献一次event。Success/Failure event accuracy各为该GT outcome中正确锁定的比例，undecided为错误；event ACC=两类总正确/全部interaction；BA是两类event accuracy均值。MacroF1仅平均success/failure两类F1，undecided对对应GT形成FN。failure precision=正确failure/全部触发failure；recall=正确failure/全部GT failure（包括undecided）；FPR=GT success却触发failure/全部GT success。Undecided rate=未触发/全部interaction，coverage=1−undecided。decided-only accuracy只是补充，不取代主accuracy。','',
    'Detection time=(首次触发last相机帧−该interaction Key帧)/30秒，负值为Key前、正值为Key后。metrics.json同时给全部触发与正确触发的mean/median/IQR、按GT分组和Key前比例；错误触发不被排除出all_decisions统计。末帧口径不复用首次触发的检测时间统计。未触发的时间为None，不填0，不当作延迟无限大混入均值。','',
    '| split | Event ACC % | Success accuracy % | Failure accuracy % | Event BA % | Macro F1 % | Fail precision % | Fail recall % | Fail FPR % | Undecided % |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('val','test'):
        m=all_metrics[split]['first_trigger'];keys=('event_accuracy','success_event_accuracy','failure_event_accuracy','balanced_event_accuracy','macro_f1','failure_precision','failure_recall','failure_false_positive_rate','undecided_rate');lines.append('| '+split+' | '+' | '.join(f'{100*m[k]:.2f}' for k in keys)+' |')
    lines+=['','## Key相对概率曲线','', '所有val/test interaction按最终outcome分组，先将每个tick转成P(decided)/conditional，再对重复相机帧平均，最后在同一Key-relative帧上对有数据的interaction等权平均；不将mean(Pfailure)/mean(Pdecided)当作mean(conditional)。缺失位置不填0或外推，阴影为interaction SD，N与variance在CSV；不是5seed均值。只有seed42。','', '![Decision probability curves](key_relative_decision_probabilities.png)','', '![Detection times](event_detection_times.png)','', '![Validation threshold surface](validation_threshold_search.png)','', '[完整metrics含检测时间与末帧口径](metrics.json) · [test逐interaction决定](test_events.csv) · [val逐interaction决定](val_events.csv) · [441对阈值](validation_threshold_sweep.csv) · [每tick概率](probability_curves.csv) · [均值/方差/N](key_relative_mean_variance.csv) · [验证](verification.json)','']
    (out/'README.md').write_text('\n'.join(lines).replace('## Key相对概率曲线','## 本轮结论与触发稳定性\n\nTest首次锁定：15/24正确，Success正确9/15=60.00%、Failure正确6/9=66.67%；12次failure触发中6次正确，precision50.00%，另6次来自15个success，FPR40.00%；2个failure interaction未触发，undecided2/24=8.33%。BA63.33%、macroF164.57%。这不是原frame-wise指标，也不能把分数变化当作模型训练改进。\n\n全部22次触发的时间中位数为Key前1.15秒（IQR前1.71秒至前.70秒）；15次正确触发中位数为前.87秒。正确success触发中位数前.87秒，正确failure前1.52秒。提前触发仍有误报，不能把负delay直接视为有可靠提前证据。\n\n触发后19/22=86.36%曾回到progress，9/22=40.91%曾出现相反outcome，说明当前瞬时证据不稳定。锁定规则只是固定第一次输出，不会使模型概率本身稳定。若允许末帧改判，同阈值test eventACC仅33.33%，undecided33.33%；因此首次锁定与末帧结果必须分别解读。平均曲线边缘可能只由1条interaction贡献，下方N明确标出，四幅图统一时间范围；不能凭边缘均值推断全体稳定。\n\n阈值由22个validation interaction选择，test仅24个Align且来自17rollout，结论只对应seed42；本次按要求未补seed、未训练。模型仍使用标注interaction起点重置状态，尚未验证无边界提示的全rollout自动检测。\n\n'+'## Key相对概率曲线'));snapshot=out/'code_snapshot';snapshot.mkdir();shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    target=project_path('WeeklySummary/10.5/online_decision', out.name);target.mkdir(parents=True,exist_ok=False);copies=[]
    for p in out.iterdir():
        if p.is_file():q=target/p.name;shutil.copy2(p,q);copies.append(dict(source=str(relative_path(p)),destination=str(relative_path(q)),sha256=sha(p)))
    dump(target/'copy_manifest.json',copies);assert all(sha(project_path(r['source']))==sha(project_path(r['destination']))==r['sha256'] for r in copies)
    with (project_path('WeeklySummary/10.5/10.5.md')).open('a') as f:f.write('\n\n## 12. Online decision evaluation：Gaussian输出双阈值\n\n'+ '\n'.join(lines[2:lines.index('## Key相对概率曲线')])+f'\n\n[完整报告、概率曲线、检测时间及441对阈值](online_decision/{out.name}/README.md)。\n\n![Key-relative decision probabilities](online_decision/{out.name}/key_relative_decision_probabilities.png)\n')
    print('COMPLETE',json.dumps(dict(thresholds=selected,metrics=all_metrics)),flush=True)

if __name__=='__main__':main()
