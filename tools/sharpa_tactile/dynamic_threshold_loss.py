"""Nested rollout-held-out BCE early stopping for causal threshold heads."""
from .common import project_path, relative_path
import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from .common import ROOT,dump,sha
from .dynamic_threshold import coordinates,history_features,operating_point
from .early_warning import HORIZONS,SEEDS,target,write_csv
from .calibrate_early_warning import evaluate,event_scores,select_threshold


def train(args):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    out=args.output;out.mkdir(parents=True,exist_ok=False);samples=coordinates(args.features)
    allowed=None if args.run_spec is None else {(r['model'],r['horizon_frames'],r['seed']) for r in json.loads(args.run_spec.read_text())}
    arrays={};provenance=[]
    for h in HORIZONS:
        for seed in SEEDS:
            arrays[h,seed]={}
            for split in ('val','test'):
                path=args.source/'runs'/f'H{h}'/f'seed_{seed}'/(split+'_predictions.npz')
                with np.load(path) as z:p=z['probabilities'].copy();offsets=z['offsets'].copy()
                assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
                assert np.array_equal(offsets,np.r_[0,np.cumsum([len(s['frames']) for s in samples[split]])])
                arrays[h,seed][split]=(p,offsets);provenance.append(dict(path=str(relative_path(path)),sha256=sha(path)))
    for split,ss in samples.items():
        np.savez_compressed(out/(split+'_coordinates.npz'),frames=np.concatenate([s['frames'] for s in ss]),offsets=np.r_[0,np.cumsum([len(s['frames']) for s in ss])])
    dump(out/'samples.json',{split:[{k:v for k,v in s.items() if k!='frames'} for s in ss] for split,ss in samples.items()})
    dump(out/'score_provenance.json',provenance)
    fold=np.empty(17,np.int64)
    for failure in (False,True):
        ids=[i for i,s in enumerate(samples['val']) if (s['outcome']==2)==failure]
        for j,i in enumerate(ids):fold[i]=j%3
    folds=[]
    for k in range(3):
        outer_train=np.flatnonzero(fold!=k).tolist();outer_holdout=np.flatnonzero(fold==k).tolist()
        failure=[i for i in outer_train if samples['val'][i]['outcome']==2]
        success=[i for i in outer_train if samples['val'][i]['outcome']!=2]
        # Fixed split, independent of H, seed, scores and losses.
        inner_val=[failure[0]]+success[:3]
        inner_train=[i for i in outer_train if i not in inner_val]
        assert not(set(inner_val)&set(inner_train)) and not(set(outer_holdout)&set(outer_train))
        folds.append(dict(fold=k,outer_train=outer_train,outer_holdout=outer_holdout,inner_train=inner_train,inner_val=inner_val))
    dump(out/'folds.json',dict(rollouts=[dict(rollout_id=s['rollout_id'],fold=int(fold[i]),failure=s['outcome']==2) for i,s in enumerate(samples['val'])],nested_splits=folds))
    dump(out/'protocol.json',dict(training_regime='nested_loss_earlystop',source=str(relative_path(args.source)),features=str(relative_path(args.features)),
        modality='frozen original Deform early-warning risk; no base retraining',horizons=list(HORIZONS),seeds=list(SEEDS),
        epochs=args.epochs,patience=args.patience,min_delta=1e-8,stopping_metric='inner rollout-held-out balanced BCE using inner-train class weights',
        mlp='7->Linear16->ReLU->Linear1',gru='7->causal GRU16->Linear1',
        inputs='same seven past-risk/elapsed-time features; current risk excluded from threshold head; no Key/future/outcome/length input',
        loss='same balanced BCEWithLogits(logit(clipped risk)-a(t),y); class weights learned on each fit training subset',
        optimizer='same AdamW lr.003 wd.001 clip1; one full rollout batch per update',
        epoch_selection='for each outer fold, inner train/val disjoint by rollout, max300/patience50 by inner val BCE; refit all outer-train for selected best epoch; OOF outer holdout used only for calibration',
        final_refit='all original val17 for median of three inner-loss-selected best epochs, at most300; no test-based epoch choice, no pretending training loss is independent val loss',
        calibration='same OOF max-margin event FPR<=15%, maximize failure recall, tie higher offset; freeze to original test17',
        baseline='original fixed threshold, exactly same earlier15% calibration',
        caution='only3val failures; inner stopping fold1failure/3success and inner training1failure; base checkpoint previously epoch-selected on originalval; nested protocol changes alongside training duration'))

    class Head(nn.Module):
        def __init__(self,kind):
            super().__init__();self.kind=kind
            self.body=nn.GRU(7,16,batch_first=True) if kind=='gru' else nn.Sequential(nn.Linear(7,16),nn.ReLU())
            self.output=nn.Linear(16,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        def forward(self,x):
            return self.output(self.body(x)[0] if self.kind=='gru' else self.body(x)).squeeze(-1)

    def batch(ss,p,offsets,h):
        width=max(len(s['frames']) for s in ss);n=len(ss)
        x=np.zeros((n,width,7),np.float32);logit=np.zeros((n,width),np.float32);y=np.zeros((n,width),np.float32);mask=np.zeros((n,width),bool);sizes=[]
        for i,s in enumerate(ss):
            v=p[offsets[i]:offsets[i+1]].astype(np.float64);length=len(v);sizes.append(length)
            x[i,:length]=history_features(v,s['frames']);v=np.clip(v,1e-6,1-1e-6);logit[i,:length]=np.log(v/(1-v))
            yy,valid=target(s['frames'],s['key'],s['outcome']==2,h);y[i,:length]=yy;mask[i,:length]=valid
        return [torch.as_tensor(v,device=args.device) for v in (x,logit,y,mask)],sizes

    def fit(kind,seed,b,ids,validation_ids=None,epochs=None,progress=''):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=Head(kind).to(args.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
        x,logit,y,mask=[v[ids] for v in b];pos=y[mask].sum();neg=mask.sum()-pos;assert pos>0 and neg>0
        w1=mask.sum()/(2*pos);w0=mask.sum()/(2*neg)
        def loss_for(xx,ll,yy,mm):
            element=nn.functional.binary_cross_entropy_with_logits(ll-model(xx),yy,reduction='none')
            return (element*torch.where(yy==1,w1,w0))[mm].mean()
        best=float('inf');best_epoch=0;state=None;history=[]
        for epoch in range(1,(epochs or args.epochs)+1):
            model.train();optimizer.zero_grad();loss=loss_for(x,logit,y,mask);assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            model.eval()
            with torch.no_grad():
                train_loss=float(loss_for(x,logit,y,mask))
                val_loss=float(loss_for(*[v[validation_ids] for v in b])) if validation_ids is not None else None
            history.append(dict(epoch=epoch,train_balanced_BCE=train_loss,val_balanced_BCE=val_loss))
            if validation_ids is not None:
                assert np.isfinite(val_loss)
                if val_loss<best-1e-8:
                    best=val_loss;best_epoch=epoch;state={k:v.detach().clone() for k,v in model.state_dict().items()}
                if epoch%50==0:print('INNER_EPOCH',progress,epoch,'train',round(train_loss,6),'val',round(val_loss,6),'best',best_epoch,flush=True)
                if epoch-best_epoch>=args.patience:break
        if validation_ids is not None:model.load_state_dict(state)
        else:best_epoch=len(history)
        metadata=dict(best_epoch=best_epoch,epochs=len(history),best_val_loss=best if validation_ids is not None else None,
                      selection='inner held-out BCE' if validation_ids is not None else 'fixed duration selected by inner held-out BCE',
                      weights=[float(w0),float(w1)])
        return model,history,metadata

    def infer(model,b,sizes):
        model.eval()
        with torch.no_grad():
            a=model(b[0]);margin=b[1]-a;a=a.cpu().numpy();margin=margin.cpu().numpy()
        return np.concatenate([a[i,:n] for i,n in enumerate(sizes)]),np.concatenate([margin[i,:n] for i,n in enumerate(sizes)])

    results=[];events=[];count=0
    for h in HORIZONS:
        for seed in SEEDS:
            if allowed is not None and not any((kind,h,seed) in allowed for kind in ('mlp','gru')):continue
            data={s:batch(samples[s],*arrays[h,seed][s],h) for s in ('val','test')};vp,vo=arrays[h,seed]['val']
            ey,es=event_scores(samples['val'],vp,vo,h);chosen=select_threshold(ey,es,.15)
            baseline=dict(horizon_frames=h,seed=seed,model='fixed',offset=chosen['threshold'],validation_allowed_fp=chosen['allowed_success_fp'],validation_selected_fp=chosen['selected_success_fp'],validation_selected_tp=chosen['selected_failure_tp'])
            for split in ('val','test'):
                f,e,rr=evaluate(samples[split],*arrays[h,seed][split],h,seed,split,baseline['offset'],.15);baseline[split]=f;baseline[split+'_event']=e;events.extend([dict(model='fixed',**v) for v in rr])
            results.append(baseline)
            for kind in ('mlp','gru'):
                if allowed is not None and (kind,h,seed) not in allowed:continue
                directory=out/'runs'/kind/f'H{h}'/f'seed_{seed}';directory.mkdir(parents=True)
                b,sizes=data['val'];oof=np.empty(len(vp),np.float64);fold_history=[];inner_history=[];metadata=[]
                for split in folds:
                    k=split['fold'];model,hist,info=fit(kind,seed+k*100,b,split['inner_train'],split['inner_val'],progress=f'{kind}H{h}s{seed}f{k}')
                    inner_history.append(hist);del model
                    selected_epochs=info['best_epoch'];model,hist,outer_info=fit(kind,seed+k*100,b,split['outer_train'],epochs=selected_epochs)
                    _,margin=infer(model,b,sizes)
                    for i in split['outer_holdout']:oof[vo[i]:vo[i+1]]=margin[vo[i]:vo[i+1]]
                    fold_history.append(hist);metadata.append(dict(fold=k,inner=info,outer_refit=outer_info));del model
                selection=operating_point(samples['val'],oof,vo,h)
                selected_epochs=int(np.median([v['inner']['best_epoch'] for v in metadata]))
                model,hist,refit=fit(kind,seed,b,list(range(17)),epochs=selected_epochs)
                torch.save(dict(state_dict=model.state_dict(),kind=kind,offset=selection['offset'],seed=seed,horizon_frames=h,selected_epochs=selected_epochs,selection='nested inner heldout BCE'),directory/'best.pt')
                result=dict(horizon_frames=h,seed=seed,model=kind,**selection,selected_epochs=selected_epochs,fold_selection=metadata,refit_selection=refit)
                result['oof_frame'],result['oof_event'],_=evaluate(samples['val'],oof,vo,h,seed,'val_oof',selection['offset'],.15)
                np.savez_compressed(directory/'oof_predictions.npz',margins=oof,offsets=vo)
                for split in ('val','test'):
                    bb,ss=data[split];a,margin=infer(model,bb,ss);p,offsets=arrays[h,seed][split]
                    f,e,rr=evaluate(samples[split],margin,offsets,h,seed,split,selection['offset'],.15);result[split]=f;result[split+'_event']=e;events.extend([dict(model=kind,**v) for v in rr])
                    logit=a.astype(np.float64)+selection['offset'];tau=1/(1+np.exp(-np.clip(logit,-700,700)))
                    np.savez_compressed(directory/(split+'_predictions.npz'),risk=p,threshold=tau,threshold_logit=logit,margins=margin,offsets=offsets)
                dump(directory/'metrics.json',result);dump(directory/'history.json',dict(inner=inner_history,folds=fold_history,refit=hist,selection=metadata))
                results.append(result);dump(out/'results.json',results);count+=1
                print('LOSS_THRESHOLD',count,50,kind,h,seed,'refitEpochs',selected_epochs,'testFPR',round(result['test_event']['fpr'],4),'correctFirst',round(result['test_event']['correct_first_alarm_rate'],4),flush=True);del model
    write_csv(out/'event_predictions.csv',events);dump(out/'gpu_complete.json',dict(status='complete',threshold_heads=count,inner_loss_fits=count*3,outer_refit_fits=count*3,full_refit_fits=count))
    shutil.copyfile(Path(__file__),out/'dynamic_threshold_loss.py');torch.cuda.synchronize(args.device);sys.stdout.flush();sys.stderr.flush();os._exit(0)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=project_path('outputs/sharpa_early_warning/20261006_001500'));p.add_argument('--features',type=Path,default=project_path('outputs/sharpa_gaussian_online_data/20261005_182000'));p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:2');p.add_argument('--epochs',type=int,default=300);p.add_argument('--patience',type=int,default=50);p.add_argument('--run-spec',type=Path)
    args=p.parse_args()
    for key in ('source','features','output'):setattr(args,key,getattr(args,key).resolve())
    train(args)


if __name__=='__main__':main()
