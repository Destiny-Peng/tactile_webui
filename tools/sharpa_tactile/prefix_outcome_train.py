"""Causal multimodal outcome GRUs: rollout-mean BCE and val-only thresholds."""
import argparse
import csv
import gzip
import json
import random
from pathlib import Path

import numpy as np

from .common import ROOT,dump,sha
from .early_warning import metrics

MODELS={'tactile':('tactile',),'tactile_rgb':('tactile','rgb'),
        'tactile_pose':('tactile','pose'),'tactile_rgb_pose':('tactile','rgb','pose'),
        'rgb_pose':('rgb','pose')}
SEEDS=(42,43,44,45,46)


def load_data(out,keys):
    records=json.loads((out/'dataset_manifest.json').read_text())['records']
    samples={k:[] for k in ('train','val','test')}
    for r in records:
        with np.load(out/r['alignment_path']) as z:
            sample=dict(r,ticks=z['ticks'],time_s=z['time_s'],frames=z['video_frames'])
            if 'pose' in keys:sample['pose']=z['pose']
        for key in keys:
            if key=='pose':continue
            with np.load(out/'features'/key/(r['id']+'.npz')) as z:
                assert np.array_equal(z['ticks'],sample['ticks'])
                sample[key]=z['features']
        for key in keys:assert len(sample[key])==len(sample['ticks']) and np.isfinite(sample[key]).all()
        samples[r['split']].append(sample)
    return samples


def array_metrics(y,p,threshold,weight=None):
    # threshold scaling is not used: retain raw probability ranking for AUC.
    y=np.asarray(y,int);p=np.asarray(p,float);pred=p>=threshold
    if not len(y):
        return dict(**{k:None for k in ('accuracy','balanced_accuracy','macro_f1','failure_precision','failure_recall','failure_fpr','roc_auc','pr_auc')},
                    confusion_matrix=[[0,0],[0,0]],threshold=float(threshold),counts=[0,0],
                    per_class_precision=[None,None],per_class_recall=[None,None],per_class_f1=[None,None])
    w=np.ones(len(y)) if weight is None else np.asarray(weight,float)
    cm=np.array([[w[(y==a)&(pred==b)].sum() for b in (0,1)] for a in (0,1)])
    recall=np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(2),where=cm.sum(1)>0)
    precision=np.divide(cm.diagonal(),cm.sum(0),out=np.zeros(2),where=cm.sum(0)>0)
    f1=np.divide(2*recall*precision,recall+precision,out=np.zeros(2),where=(recall+precision)>0)
    order=np.argsort(-p,kind='stable');sy,sp,sw=y[order],p[order],w[order]
    ends=np.r_[np.flatnonzero(np.diff(sp)),len(p)-1]
    tp=np.cumsum(sw*(sy==1))[ends];fp=np.cumsum(sw*(sy==0))[ends]
    positives=w[y==1].sum();negatives=w[y==0].sum()
    auc=ap=None
    if positives:
        rr=tp/positives
        ap=float(np.sum(np.diff(np.r_[0,rr])*np.divide(tp,tp+fp)))
        if negatives:auc=float(np.trapz(np.r_[0,rr],np.r_[0,fp/negatives]))
    result=dict(accuracy=float(w[pred==y].sum()/w.sum()),balanced_accuracy=float(recall.mean()),
                macro_f1=float(f1.mean()),failure_precision=float(precision[1]),failure_recall=float(recall[1]),
                failure_fpr=float(1-recall[0]),confusion_matrix=cm.tolist(),threshold=float(threshold),
                per_class_precision=precision.tolist(),per_class_recall=recall.tolist(),per_class_f1=f1.tolist(),
                counts=[int((y==0).sum()),int((y==1).sum())],roc_auc=auc,pr_auc=ap)
    if not all((y==a).any() for a in (0,1)):result['balanced_accuracy']=None
    return result


def calibrate(samples,ps):
    yy=np.concatenate([np.full(len(p),s['label']) for s,p in zip(samples,ps)])
    pp=np.concatenate(ps);ww=np.concatenate([np.full(len(p),1/len(p)) for p in ps])
    order=np.argsort(-pp,kind='stable');y,p,w=yy[order],pp[order],ww[order]
    ends=np.r_[np.flatnonzero(np.diff(p)),len(p)-1]
    tp=np.cumsum(w*(y==1))[ends];fp=np.cumsum(w*(y==0))[ends]
    scores=.5*(tp/w[y==1].sum()+1-fp/w[y==0].sum())
    thresholds=np.r_[np.nextafter(p[0],np.inf),p[ends]]
    scores=np.r_[.5,scores]
    best=int(np.flatnonzero(scores>=scores.max()-1e-12)[0])
    return float(thresholds[best]),dict(objective='maximize rollout-equal frame balanced accuracy on validation only; ties higher threshold',
                                      validation_score=float(scores[best]),candidates=len(thresholds))


def evaluate(samples,ps,threshold):
    y=np.concatenate([np.full(len(p),s['label']) for s,p in zip(samples,ps)])
    p=np.concatenate(ps);w=np.concatenate([np.full(len(p),1/len(p)) for p in ps])
    result=dict(pooled_frames=array_metrics(y,p,threshold),rollout_equal_frames=array_metrics(y,p,threshold,w),
                final_timestep=array_metrics([s['label'] for s in samples],[p[-1] for p in ps],threshold),
                stages={},prefix_endpoints={})
    for stage in range(5):
        ys=[];pp=[];ww=[]
        for s,local in zip(samples,ps):
            # Only evaluation uses full length; no duration/relative-time input.
            indices=np.arange(len(local));mask=(indices*5//len(local))==stage
            if not mask.any():continue
            ys.extend([s['label']]*int(mask.sum()));pp.extend(local[mask]);ww.extend([1/int(mask.sum())]*int(mask.sum()))
        result['stages'][f'{stage*20}-{(stage+1)*20}%']=array_metrics(ys,pp,threshold,ww)
    for fraction in (20,40,60,80,100):
        endpoints=[p[max(0,int(np.ceil(len(p)*fraction/100))-1)] for p in ps]
        result['prefix_endpoints'][str(fraction)]=array_metrics([s['label'] for s in samples],endpoints,threshold)
    return result


def train(args):
    import torch
    from torch import nn
    torch.set_num_threads(2)
    keys=MODELS[args.model];samples=load_data(args.output,keys)
    norm={}
    for key in keys:
        mean=np.mean([s[key].mean(0,dtype=np.float64) for s in samples['train']],axis=0)
        second=np.mean([(s[key].astype(np.float64)**2).mean(0) for s in samples['train']],axis=0)
        norm[key]=(mean.astype(np.float32),np.maximum(np.sqrt(np.maximum(second-mean**2,0)),.01).astype(np.float32))
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.encodings=nn.ModuleDict()
            for key in keys:
                mean,std=norm[key]
                self.register_buffer(key+'_mean',torch.from_numpy(mean.copy()))
                self.register_buffer(key+'_std',torch.from_numpy(std.copy()))
                self.encodings[key]=nn.Sequential(nn.Linear(len(mean),128),nn.ReLU(),nn.Linear(128,128)) if key=='pose' else nn.Linear(len(mean),128)
            self.fusion=nn.Linear(128*len(keys),128)
            self.gru=nn.GRU(128,128,batch_first=True,bidirectional=False)
            self.head=nn.Linear(128,1)
        def forward(self,x):
            v=torch.cat([self.encodings[k]((x[k]-getattr(self,k+'_mean'))/getattr(self,k+'_std')) for k in keys],dim=-1)
            h,_=self.gru(self.fusion(v))
            return self.head(h).squeeze(-1)
    def batch(ss):
        lengths=[len(s['ticks']) for s in ss];width=max(lengths)
        xx={k:np.zeros((len(ss),width,len(norm[k][0])),np.float32) for k in keys}
        mask=np.zeros((len(ss),width),bool)
        for i,s in enumerate(ss):
            for k in keys:xx[k][i,:lengths[i]]=s[k]
            mask[i,:lengths[i]]=True
        return {k:torch.from_numpy(v).to(args.device) for k,v in xx.items()},torch.tensor([s['label'] for s in ss],device=args.device,dtype=torch.float32),torch.from_numpy(mask).to(args.device),lengths
    def infer(model,ss):
        model.eval();ps=[];losses=[]
        with torch.inference_mode():
            for a in range(0,len(ss),args.batch_size):
                x,y,mask,lengths=batch(ss[a:a+args.batch_size]);logits=model(x)
                target=y[:,None].expand_as(logits)
                each=(nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*mask).sum(1)/mask.sum(1)
                losses.extend(each.cpu().tolist());p=logits.sigmoid().cpu().numpy()
                ps.extend(p[i,:length] for i,length in enumerate(lengths))
        return ps,float(np.mean(losses))
    for seed in args.seeds:
        directory=args.output/'runs'/args.model/f'seed_{seed}'
        directory.mkdir(parents=True,exist_ok=args.evaluate_only)
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        model=Model().to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        rng=np.random.default_rng(seed);best=float('inf');best_epoch=0;history=[]
        if args.evaluate_only:
            saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
            assert saved['model']==args.model and saved['seed']==seed
            history=json.loads((directory/'history.json').read_text())
            best_epoch=saved['best_epoch'];best=history[best_epoch-1]['val_rollout_mean_BCE']
        for epoch in range(1,1 if args.evaluate_only else args.epochs+1):
            model.train();order=rng.permutation(len(samples['train']));loss_sum=0;total=0
            for a in range(0,len(order),args.batch_size):
                ss=[samples['train'][int(i)] for i in order[a:a+args.batch_size]]
                x,y,mask,_=batch(ss);optimizer.zero_grad(set_to_none=True);logit=model(x)
                element=nn.functional.binary_cross_entropy_with_logits(logit,y[:,None].expand_as(logit),reduction='none')
                per_rollout=(element*mask).sum(1)/mask.sum(1);loss=per_rollout.mean()
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                loss_sum+=float(per_rollout.detach().sum());total+=len(ss)
            _,val_loss=infer(model,samples['val'])
            history.append(dict(epoch=epoch,train_rollout_mean_BCE=loss_sum/total,val_rollout_mean_BCE=val_loss))
            dump(directory/'history.json',history)
            if val_loss<best-1e-8:
                best=val_loss;best_epoch=epoch
                torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                model=args.model,seed=seed,best_epoch=best_epoch,input_dims={k:len(norm[k][0])for k in keys},
                                protocol_sha256=sha(args.output/'training_protocol.json')),directory/'best.pt')
            if epoch%10==0:print('EPOCH',args.model,seed,epoch,'valBCE',round(val_loss,6),'best',best_epoch,flush=True)
            if epoch-best_epoch>=args.patience:break
        model.load_state_dict(torch.load(directory/'best.pt',map_location=args.device,weights_only=True)['state_dict'])
        val_ps,_=infer(model,samples['val']);threshold,calibration=calibrate(samples['val'],val_ps)
        result=dict(model=args.model,seed=seed,best_epoch=best_epoch,epochs=len(history),
                    threshold=threshold,calibration=calibration,best_val_loss=best,metrics={})
        if seed==42:
            s=samples['val'][0];half=max(1,len(s['ticks'])//2)
            cut=dict(s)
            for k in keys:cut[k]=s[k][:half]
            cut['ticks']=s['ticks'][:half]
            full,_=infer(model,[s]);prefix,_=infer(model,[cut])
            delta=float(np.max(abs(full[0][:half]-prefix[0])))
            assert delta<2e-5
            result['causal_prefix_consistency_max_probability_delta']=delta
        for split in ('train','val','test'):
            ps,loss=infer(model,samples[split])
            result['metrics'][split]=evaluate(samples[split],ps,threshold)
            result['metrics'][split]['rollout_mean_BCE']=loss
            offsets=np.r_[0,np.cumsum([len(p) for p in ps])]
            np.savez_compressed(directory/(split+'_predictions.npz'),probabilities=np.concatenate(ps),offsets=offsets,
                ticks=np.concatenate([s['ticks']for s in samples[split]]),time_s=np.concatenate([s['time_s']for s in samples[split]]),
                video_frames=np.concatenate([s['frames']for s in samples[split]]),labels=np.array([s['label']for s in samples[split]],np.int64))
            dump(directory/(split+'_samples.json'),[dict(rollout_id=s['id'],task=s['task_key'],label=s['label'],timesteps=len(p))for s,p in zip(samples[split],ps)])
            with gzip.open(directory/(split+'_predictions.csv.gz'),'wt',newline='')as stream:
                writer=csv.writer(stream);writer.writerow(['rollout_id','tick','time_s','video_frame','gt','p_failure','prediction','threshold'])
                for s,p in zip(samples[split],ps):
                    writer.writerows((s['id'],int(t),float(time),int(f),s['label'],float(q),int(q>=threshold),threshold)for t,time,f,q in zip(s['ticks'],s['time_s'],s['frames'],p))
        dump(directory/'history.json',history);dump(directory/'metrics.json',result)
        print('RUN_COMPLETE',args.model,seed,'testBA',result['metrics']['test']['rollout_equal_frames']['balanced_accuracy'],flush=True)
        del model,optimizer;torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--patience',type=int,default=50)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--evaluate-only',action='store_true',help='Load existing best checkpoints and regenerate full predictions/metrics without training')
    args=p.parse_args();args.output=args.output.resolve();train(args)
