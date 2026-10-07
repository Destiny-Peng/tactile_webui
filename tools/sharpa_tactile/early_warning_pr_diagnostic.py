"""Validation-only PR-selected Deform GRU diagnostic; retain every epoch state_dict."""
from .common import project_path, relative_path
import argparse
import datetime
import json
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import ROOT,dump,sha
from .early_warning import read_data,distribution,target,metrics,write_csv

HORIZONS=(0,8,15)
SOURCE=project_path('outputs/sharpa_gaussian_online_data/20261005_182000')


class DeformGRU(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        self.register_buffer('mean',mean.clone());self.register_buffer('std',std.clone())
        self.projection=nn.Linear(2560,128)
        self.gru=nn.GRU(128,128,batch_first=True)
        self.prediction=nn.Linear(128,1)
    def forward(self,x):
        x=self.projection((x-self.mean)/self.std);x,_=self.gru(x)
        return self.prediction(x).squeeze(-1)


def load_model(checkpoint,device='cpu'):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=DeformGRU(state['mean'],state['std']);model.load_state_dict(state)
    return model.to(device).eval()


def train(args):
    torch.set_num_threads(4)
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    all_samples=read_data(SOURCE,False)
    # The test split is retained in source manifest only, never inferred/scored.
    samples={s:all_samples[s] for s in ('train','val')};del all_samples
    for ss in samples.values():
        for s in ss:
            with np.load(SOURCE/s['feature_path']) as z:s['deform']=z['deform'].copy()
    h=args.horizon;seed=42
    dist=distribution(samples);write_csv(out/'label_distribution.csv',[d for d in dist if d['horizon_frames']==h])
    d=next(d for d in dist if d['horizon_frames']==h and d['split']=='train')
    bank=np.concatenate([s['deform'] for s in samples['train']])
    mean=torch.from_numpy(bank.mean(0,dtype=np.float64).astype(np.float32))
    std=torch.from_numpy(np.maximum(bank.std(0,dtype=np.float64),.01).astype(np.float32));del bank
    dump(out/'protocol.json',dict(horizon_frames=h,seed=seed,source=str(relative_path(SOURCE)),source_sha256=sha(SOURCE/'dataset_manifest.json'),
        model='frozen current Deform2560 ->Linear128 ->single-layer unidirectional GRU128 ->Linear1 ->sigmoid',
        input='identical historical dense step0 cache and rollout ordering; reset at rollout start only, matching original baseline',
        label='final Align failure end anchor only; H>0 [anchor-H,anchor) positive, failure t>=anchor excluded; H0 t>=anchor positive; success0',
        split=dict(train=80,val=17),test='no test inference, metrics or threshold calibration',
        loss='train balanced BCE w0=N/(2N0),w1=N/(2N1) from train valid ticks; effective-tick batch mean',
        optimizer='AdamW lr.001 wd.0001 batch8rollouts gradclip1',max_epoch=30,patience=8,
        selection='maximum validation frame PR-AUC (average precision); improvement>1e-8; ties earliest; patience counts epochs since best AP',
        ranking='PR-AUC/ROC-AUC computed using raw logits, monotone-equivalent to sigmoid without numerical saturation ties',
        validation_BCE='natural mean stable BCE(logits,GT) over valid ticks',
        validation_train_weighted_BCE='same train-derived w0/w1 multiplied into val BCE, then valid-tick mean; diagnostic only',
        BA='threshold0.5 diagnostic only; not used for selection or stopping',
        train_loss='online average over minibatches during the epoch, not a separate epoch-end train pass',
        checkpoints='raw state_dict includes normalization buffers; epoch000 initial; epoch001..NNN after completed epoch; best_pr_auc.pt copied from selected epoch',
        created_at=datetime.datetime.now().astimezone().isoformat()))
    shutil.copy2(SOURCE/'dataset_manifest.json',out/'source_dataset_manifest.json')
    def batch(ss):
        lengths=[len(s['deform']) for s in ss];width=max(lengths)
        x=np.zeros((len(ss),width,2560),np.float32);y=np.zeros((len(ss),width),np.float32);mask=np.zeros_like(y,dtype=bool)
        for i,s in enumerate(ss):
            x[i,:lengths[i]]=s['deform'];q,k=target(s['frames'],s['key'],s['outcome']==2,h)
            y[i,:lengths[i]],mask[i,:lengths[i]]=q,k
        return tuple(torch.from_numpy(v).to(args.device) for v in (x,y,mask)),lengths
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=DeformGRU(mean,std).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed);best=-1;best_epoch=0;history=[];checkpoint_rows=[]
    (out/'checkpoints').mkdir();(out/'validation_scores').mkdir()
    def save_epoch(epoch):
        path=out/'checkpoints'/f'epoch_{epoch:03d}.pt'
        torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},path)
        checkpoint_rows.append(dict(epoch=epoch,path=str(path.relative_to(out)),sha256=sha(path),bytes=path.stat().st_size))
        write_csv(out/'checkpoint_index.csv',checkpoint_rows)
        return path
    save_epoch(0)
    for epoch in range(1,31):
        model.train();order=rng.permutation(len(samples['train']));total=0;loss_sum=0
        for a in range(0,len(order),8):
            ss=[samples['train'][int(i)] for i in order[a:a+8]];(x,y,mask),_=batch(ss)
            optimizer.zero_grad(set_to_none=True);logit=model(x)
            element=nn.functional.binary_cross_entropy_with_logits(logit,y,reduction='none')
            weights=torch.where(y>0,d['positive_weight'],d['negative_weight']);loss=(element*weights)[mask].mean()
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            loss_sum+=float((element*weights)[mask].detach().sum());total+=int(mask.sum())
        model.eval();logits=[]
        with torch.inference_mode():
            for a in range(0,len(samples['val']),8):
                (x,_,_),lengths=batch(samples['val'][a:a+8]);raw=model(x).cpu().numpy()
                logits.extend(raw[i,:length] for i,length in enumerate(lengths))
        offsets=np.r_[0,np.cumsum([len(z) for z in logits])];logits=np.concatenate(logits)
        ys,masks=zip(*(target(s['frames'],s['key'],s['outcome']==2,h) for s in samples['val']))
        valid=np.concatenate(masks);allgt=np.concatenate(ys);gt=allgt[valid];z=logits[valid].astype(np.float64)
        bce=np.logaddexp(0,z)-gt*z;weights=np.where(gt>0,d['positive_weight'],d['negative_weight'])
        probability=np.exp(-np.logaddexp(0,-z));frame=metrics(gt,probability);ranking=metrics(gt,z)
        row=dict(epoch=epoch,train_balanced_BCE=loss_sum/total,val_natural_BCE=float(bce.mean()),
            val_train_weighted_BCE=float((bce*weights).mean()),val_PR_AUC=ranking['pr_auc'],val_ROC_AUC=ranking['roc_auc'],val_BA_at_0_5=frame['balanced_accuracy'])
        history.append(row);write_csv(out/'epoch_metrics.csv',history);dump(out/'epoch_metrics.json',history)
        np.savez_compressed(out/'validation_scores'/f'epoch_{epoch:03d}.npz',logits=logits,offsets=offsets,labels=allgt,valid=valid,
            frames=np.concatenate([s['frames'] for s in samples['val']]),rollout_ids=np.array([s['rollout_id'] for s in samples['val']]))
        checkpoint=save_epoch(epoch)
        if row['val_PR_AUC']>best+1e-8:
            best=row['val_PR_AUC'];best_epoch=epoch
            shutil.copy2(checkpoint,out/'best_pr_auc.pt')
            dump(out/'best_pr_auc.json',dict(epoch=best_epoch,val_PR_AUC=best,checkpoint=str(checkpoint.relative_to(out)),sha256=sha(checkpoint),metrics=row))
        print('EPOCH',h,epoch,json.dumps(row),'best_AP_epoch',best_epoch,flush=True)
        if epoch-best_epoch>=8:break
    dump(out/'result.json',dict(horizon_frames=h,seed=seed,epochs=len(history),best_epoch=best_epoch,best_PR_AUC=best,checkpoints=len(checkpoint_rows),test_evaluated=False))
    dump(out/'manifest.json',dict(status='complete',code_sha256=sha(Path(__file__)),completed_at=datetime.datetime.now().astimezone().isoformat()))
    print('COMPLETE',h,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--horizon',type=int,required=True,choices=HORIZONS);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:2')
    args=p.parse_args();args.output=args.output.resolve();train(args)


if __name__=='__main__':main()
