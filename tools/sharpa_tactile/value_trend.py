"""Tactile adaptation: proxy value -> future value -> trend risk + focal trigger."""
from .common import project_path, relative_path
import argparse
import copy
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT, dump, sha
from .early_warning import read_data, distribution, target, evaluate_arrays, write_csv

SOURCE = project_path('outputs/sharpa_gaussian_online_data/20261005_182000')
SEED = 42


def seed():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


class Proxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(2560,128)
        self.gru = nn.GRU(128,128,batch_first=True)
        self.heads = nn.Linear(128,2)
    def forward(self,x):
        h,_=self.gru(F.relu(self.project(x)))
        return self.heads(h).sigmoid()


class Future(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(2560,128)
        self.gru = nn.GRU(128,128,batch_first=True)
        self.future = nn.Sequential(nn.Linear(128,256),nn.ReLU(),nn.Linear(256,2560))
        self.q = nn.Sequential(nn.Linear(2560,128),nn.ReLU(),nn.Linear(128,2),nn.Sigmoid())
        self.current = nn.Linear(128,1)
    def forward(self,x):
        h,_=self.gru(F.relu(self.project(x)))
        z=self.future(h)
        return z,self.q(z),self.current(h).sigmoid().squeeze(-1)


class Risk(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent=nn.Sequential(nn.Linear(2560,128),nn.ReLU())
        self.hist=nn.Sequential(nn.Linear(17,32),nn.ReLU())
        self.shared=nn.Sequential(nn.Linear(160,128),nn.ReLU())
        self.risk=nn.Linear(128,1)
        self.trigger=nn.Linear(128,1)
    def forward(self,z,h):
        hidden=self.shared(torch.cat((self.latent(z),self.hist(h)),dim=-1))
        return self.risk(hidden).squeeze(-1),self.trigger(hidden).squeeze(-1)


def load_data(out):
    samples=read_data(SOURCE,True)
    normfile=out/'normalization.npz'
    if normfile.exists():
        with np.load(normfile) as z: mean,std=z['mean'],z['std']
    else:
        bank=np.concatenate([s['deform'] for s in samples['train']])
        mean=bank.mean(0,dtype=np.float64).astype('float32')
        std=np.maximum(bank.std(0,dtype=np.float64),.01).astype('float32')
        np.savez(normfile,mean=mean,std=std)
    for ss in samples.values():
        for s in ss:
            s['x']=(s.pop('deform')-mean)/std
            with np.load(SOURCE/s['feature_path']) as z: s['ticks_raw']=z['ticks'].copy()
            s['failure']=s['outcome']==2
    return samples


def segments(s):
    """Reset recurrent context on missing encoder ticks (padding never enters losses)."""
    edges=np.r_[0,np.flatnonzero(np.diff(s['ticks_raw'])!=1)+1,len(s['x'])]
    return [(int(a),int(b)) for a,b in zip(edges[:-1],edges[1:])]


def sequence_forward(model,s,device):
    outputs=[]
    for a,b in segments(s):
        outputs.append(model(torch.as_tensor(s['x'][a:b],device=device)[None]))
    if isinstance(outputs[0],tuple):
        return tuple(torch.cat([v[j][0] for v in outputs]) for j in range(len(outputs[0])))
    return torch.cat([v[0] for v in outputs])


def proxy_loss(model,s,device,negative=None,warmup=False):
    pred=sequence_forward(model,s,device)
    v=pred.min(-1).values
    length=len(v)
    adjacent=torch.as_tensor(np.diff(s['ticks_raw'])==1,device=device)
    # No Bellman transitions across missing tactile ticks; true terminal kept.
    tdmask=torch.cat((adjacent,torch.ones(1,device=device,dtype=torch.bool)))
    nextv=torch.cat((v[1:].detach(),v.new_zeros(1)))
    reward=v.new_zeros(length); reward[-1]=float(not s['failure'])
    done=v.new_zeros(length); done[-1]=1
    td=F.smooth_l1_loss(pred[tdmask],(reward+.99*(1-done)*nextv)[tdmask,None].expand(-1,2),reduction='none').sum(-1).mean()
    label=v.new_zeros(())
    mono=v.new_zeros(())
    if not s['failure']:
        progress=torch.as_tensor((s['frames']-s['frames'][0])/max(1,s['frames'][-1]-s['frames'][0]),dtype=v.dtype,device=device)
        label=F.smooth_l1_loss(v,progress)
        if adjacent.any():mono=F.relu(v[:-1]-v[1:])[adjacent].mean()
    cql=v.new_zeros(())
    if negative is not None:
        neg=sequence_forward(model,negative,device).min(-1).values
        # log-mean-exp differs from paper's log-sum-exp only by a constant.
        cql=torch.logsumexp(neg,0)-np.log(len(neg))-v.mean()
    loss=.3*label+mono+v.sum()*0 if warmup else td+.3*label+mono+.05*cql
    return loss,dict(td=float(td.detach()),progress=float(label.detach()),monotonicity=float(mono.detach()),cql=float(cql.detach()))


def balanced_epoch(ss,rng):
    groups=[[s for s in ss if s['failure']==b] for b in (False,True)]
    count=max(map(len,groups))
    indices=[rng.choice(len(g),count,replace=len(g)<count) for g in groups]
    return [groups[k][int(indices[k][i])] for i in rng.permutation(count) for k in rng.permutation(2)]


def fit(model,ss,lossfn,directory,device,balanced=True,warmup=0):
    directory.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(SEED);best=float('inf');best_epoch=0;history=[]
    for epoch in range(1,301):
        model.train(); losses=[];components=[]
        order=balanced_epoch(ss['train'],rng) if balanced else [ss['train'][i] for i in rng.permutation(len(ss['train']))]
        # Accumulate eight equally weighted rollouts per optimizer update.
        optimizer.zero_grad(set_to_none=True)
        for i,s in enumerate(order):
            loss,c=lossfn(model,s,epoch<=warmup,True)
            batchstart=(i//8)*8; batchsize=min(8,len(order)-batchstart)
            (loss/batchsize).backward();losses.append(float(loss.detach()));components.append(c)
            if (i+1)%8==0 or i+1==len(order):
                nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();optimizer.zero_grad(set_to_none=True)
        model.eval();vals=[]
        with torch.inference_mode():
            for s in ss['val']:
                loss,_=lossfn(model,s,False,False);vals.append((s['failure'],float(loss)))
        # Outcome-balanced validation means, fixed and independent of epoch sampling.
        val=float(np.mean([np.mean([v for f,v in vals if f==b]) for b in (False,True)])) if balanced else float(np.mean([v for _,v in vals]))
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=val)
        for k in components[0]:row['train_'+k]=float(np.mean([c[k] for c in components]))
        history.append(row)
        if epoch>warmup and val<best-1e-8:
            best=val;best_epoch=epoch
            torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},best_epoch=epoch,validation_loss=val,seed=SEED),directory/'best.pt')
        write_csv(directory/'history.csv',history)
        if epoch%10==0 or epoch==1:print(directory.name,epoch,'train',row['train_loss'],'val',val,'best',best_epoch,flush=True)
        if best_epoch and epoch-best_epoch>=50:break
    model.load_state_dict(torch.load(directory/'best.pt',map_location=device,weights_only=True)['state_dict'])
    result=dict(best_epoch=best_epoch,epochs=len(history),validation_loss=best,max_epochs=300,patience=50)
    dump(directory/'training.json',result)
    return result


def prepare(args):
    seed();out=args.output;out.mkdir(parents=True,exist_ok=False)
    # Immutable copy of the exact historical interval anchors and split.
    for name in ('dataset_manifest.json',): shutil.copy2(SOURCE/name,out/('source_'+name))
    upstream=project_path('outputs/sharpa_merged_online_datasets/20261005_164000/dataset_manifest.json')
    shutil.copy2(upstream,out/'source_interval_manifest.json')
    samples=load_data(out)
    write_csv(out/'label_distribution.csv',distribution(samples))
    manifest=json.loads((project_path('datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl')).read_text().splitlines()[0])
    dump(out/'protocol.json',dict(source=str(relative_path(SOURCE)),source_sha256=sha(SOURCE/'dataset_manifest.json'),interval_sha256=sha(upstream),
        source_paper='https://arxiv.org/html/2606.12372v1',anchor='historical final Align end; final failure only; earlier failures and success Keys never positive',
        split=dict(train=80,val=17,test=17),seeds=[42],horizons=[0,8,15,30,45],input='frozen current Deform2560; train-only normalization; no time/progress/rollout length input',
        proxy='Linear128-ReLU-causal GRU128-twin sigmoid values; min; TD gamma.99 + .3 success SmoothL1 progress + monotonic hinge + .05 failed-state CQL; 5 progress-only warmup epochs',
        progress='relative camera-frame position within cached interaction, success only; failure terminal reward0; success terminal reward1',
        future='independent Linear128-causal GRU128 -> MLP predicted next Deform2560; normalized latent MSE + twin Q SmoothL1 frozen next proxy + auxiliary current-value SmoothL1',
        risk='K8 recent value increments; epsilon.005 gamma_r.9; (1-Vt)*discounted sum(epsilon-dV); exact signed formula, no ReLU/min-max future normalization',
        trigger='per-H original Key labels; focal alpha.75 gamma2 + SmoothL1 continuous trend risk, weight1 each; frozen proxy and future stages',
        H0='t>=anchor positive; all ticks retained',Hpositive='[anchor-H,anchor) positive; failure t>=anchor excluded from trigger loss/evaluation',
        optimizer='AdamW lr.001 wd.0001 clip1; gradient accumulation8rollouts; max300 patience50 val loss; proxy/future balanced outcome sampling; risk natural sampling',
        adaptations=['No policy action recorded: future is observation-conditioned, not action-value Q','No VLM/V-JEPA2; frozen Deform latent reused','No extra negative raw-return magnitude invented: zero failure terminal return as appendix TD equation','No intervention mining: user approved original H-Key targets for focal trigger','No offline per-episode min-max normalization at inference: use bounded proxy directly, causal history only','Proxy progress refers to available interaction segment, not full robot episode'],
        validation='stage checkpoints by validation loss only; threshold .5 primary, optional val eventFPR<=15% calibration; test never selects settings'))
    negatives=[s for s in samples['train'] if s['failure']]
    valnegative=[s for s in samples['val'] if s['failure']]
    proxy=Proxy().to(args.device)
    rng=np.random.default_rng(SEED)
    def lossfn(model,s,warm,training):
        pool=negatives if training else valnegative
        # Fixed validation negative rotates by rollout identifier, never test states.
        idx=int(rng.integers(len(pool))) if training else sum(s['rollout_id'].encode())%len(pool)
        return proxy_loss(model,s,args.device,pool[idx],warm)
    fit(proxy,samples,lossfn,out/'proxy',args.device,True,5)
    for ss in samples.values():
        for s in ss:
            with torch.inference_mode():s['proxy']=sequence_forward(proxy,s,args.device).min(-1).values.cpu().numpy()
    del proxy
    future=Future().to(args.device)
    def futureloss(model,s,warm,training):
        z,q,current=sequence_forward(model,s,args.device)
        adjacent=torch.as_tensor(np.diff(s['ticks_raw'])==1,device=args.device)
        xnext=torch.as_tensor(s['x'][1:],device=args.device)
        pv=torch.as_tensor(s['proxy'],device=args.device)
        latent=(F.normalize(z[:-1],dim=-1)-F.normalize(xnext,dim=-1)).square().sum(-1)[adjacent].mean()
        qloss=F.smooth_l1_loss(q[:-1][adjacent],pv[1:][adjacent,None].expand(-1,2),reduction='none').sum(-1).mean()
        currentloss=F.smooth_l1_loss(current,pv)
        return latent+qloss+currentloss,dict(latent=float(latent.detach()),q=float(qloss.detach()),current=float(currentloss.detach()))
    fit(future,samples,futureloss,out/'future',args.device,True)
    (out/'features').mkdir()
    for split,ss in samples.items():
        for s in ss:
            with torch.inference_mode():z,q,_=sequence_forward(future,s,args.device)
            v=s['proxy'];hist=np.zeros((len(v),17),np.float32);risk=np.zeros(len(v),np.float32)
            for a,b in segments(s):
                for t in range(a,b):
                    indices=np.maximum(np.arange(t-8,t+1),a)
                    values=v[indices];dv=np.diff(values)
                    hist[t]=np.r_[values,dv]
                    risk[t]=(1-v[t])*np.sum((.9**np.arange(8))*(.005-dv[::-1]))
            np.savez_compressed(out/'features'/f"{s['rollout_id']}.npz",z=z.cpu().numpy(),q=q.min(-1).values.cpu().numpy(),proxy=v,history=hist,risk=risk,frames=s['frames'],ticks=s['ticks_raw'])
    dump(out/'shared_complete.json',dict(proxy=sha(out/'proxy/best.pt'),future=sha(out/'future/best.pt'),features=114))
    print('SHARED_COMPLETE',flush=True)


def calibrated(ss,ps,offsets,h):
    scores=[];truth=[]
    for i,s in enumerate(ss):
        _,mask=target(s['frames'],s['key'],s['failure'],h)
        scores.append(float(ps[offsets[i]:offsets[i+1]][mask].max()));truth.append(s['failure'])
    scores=np.array(scores);truth=np.array(truth)
    candidates=np.r_[np.nextafter(scores.max(),np.inf),np.unique(scores)]
    eligible=[]
    for tau in candidates:
        pred=scores>=tau; fpr=float(pred[~truth].mean());recall=float(pred[truth].mean())
        if fpr<=.15+1e-10:eligible.append((recall,float(tau),fpr))
    recall,tau,fpr=max(eligible)
    return dict(threshold=tau,validation_event_recall=recall,validation_event_fpr=fpr)


def train_risk(args):
    seed();out=args.output;h=args.horizon;directory=out/'runs'/f'H{h}'
    assert (out/'shared_complete.json').exists()
    samples=read_data(SOURCE,False)
    for ss in samples.values():
        for s in ss:
            s['failure']=s['outcome']==2
            with np.load(out/'features'/f"{s['rollout_id']}.npz") as z:
                for k in ('z','history','risk','proxy','q'):s[k]=torch.as_tensor(z[k].copy(),device=args.device)
            y,mask=target(s['frames'],s['key'],s['failure'],h)
            s['y']=torch.as_tensor(y,dtype=torch.float32,device=args.device);s['mask']=torch.as_tensor(mask,device=args.device)
    model=Risk().to(args.device)
    def lossfn(model,s,warm,training):
        risk,logit=model(s['z'],s['history'])
        y=s['y'];mask=s['mask'];p=logit.sigmoid()
        ce=F.binary_cross_entropy_with_logits(logit,y,reduction='none')
        pt=torch.where(y>0,p,1-p);alpha=torch.where(y>0,.75,.25)
        focal=(alpha*(1-pt).square()*ce)[mask].mean()
        reg=F.smooth_l1_loss(risk[mask],s['risk'][mask])
        return focal+reg,dict(focal=float(focal.detach()),risk=float(reg.detach()))
    training=fit(model,samples,lossfn,directory,args.device,False)
    result=dict(horizon_frames=h,seed=SEED,**training)
    predictions={}
    model.eval()
    for split in ('val','test'):
        ss=samples[split];ps=[];risks=[]
        with torch.inference_mode():
            for s in ss:
                r,l=model(s['z'],s['history']);ps.append(l.sigmoid().cpu().numpy());risks.append(r.cpu().numpy())
        offsets=np.r_[0,np.cumsum([len(p) for p in ps])];ps=np.concatenate(ps)
        predictions[split]=(ps,offsets)
        frame,event,events=evaluate_arrays(ss,ps,offsets,h,split,SEED)
        result[split]=dict(frame=frame,event=event)
        write_csv(directory/f'{split}_events.csv',events)
        np.savez_compressed(directory/f'{split}_predictions.npz',probabilities=ps,offsets=offsets,risk=np.concatenate(risks),
            proxy=np.concatenate([s['proxy'].cpu().numpy() for s in ss]),q=np.concatenate([s['q'].cpu().numpy() for s in ss]),
            frames=np.concatenate([s['frames'] for s in ss]),keys=np.array([s['key'] for s in ss]),
            failure=np.array([s['failure'] for s in ss]),rollout_ids=np.array([s['rollout_id'] for s in ss]))
    calibration=calibrated(samples['val'],*predictions['val'],h);tau=calibration['threshold']
    result['calibration']=calibration
    for split in ('val','test'):
        ps,offsets=predictions[split]
        # evaluate_arrays fixes .5: affine score shift preserves original threshold exactly.
        adjusted=ps.astype(np.float64)-tau+.5
        frame,event,events=evaluate_arrays(samples[split],adjusted,offsets,h,split,SEED)
        # Threshold-free metrics refer to the original probability ranking, unchanged by shift.
        result[split]['calibrated_event']=event
        write_csv(directory/f'{split}_calibrated_events.csv',events)
    dump(directory/'result.json',result)
    print('RISK_COMPLETE',h,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','risk']);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:2');p.add_argument('--horizon',type=int,choices=[0,8,15,30,45]);args=p.parse_args()
    args.output=args.output.resolve();torch.set_num_threads(2);torch.backends.cudnn.benchmark=True
    if args.stage=='prepare':prepare(args)
    else:train_risk(args)


if __name__=='__main__':main()
