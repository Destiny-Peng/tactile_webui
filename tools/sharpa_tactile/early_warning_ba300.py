"""Controlled repeat of original early_warning.train: max300, BA patience50 only."""
from .common import project_path, relative_path
import argparse
import datetime
import json
import random
from pathlib import Path
import numpy as np
from .common import ROOT, dump, sha
from .early_warning import read_data, distribution, write_csv, target, evaluate_arrays

SEEDS=(42,)
HORIZONS=(0,8,15,30,45)

def train(args):
    import torch
    from torch import nn
    torch.set_num_threads(4)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    print('LOADING_DEFORM_CACHE', flush=True)
    samples = read_data(args.source, True)
    print('CACHE_LOADED', flush=True)
    dist = distribution(samples)
    write_csv(out/'label_distribution.csv', dist)
    bank = np.concatenate([s['deform'] for s in samples['train']])
    mean = torch.from_numpy(bank.mean(0, dtype=np.float64).astype(np.float32))
    std = torch.from_numpy(np.maximum(bank.std(0, dtype=np.float64), .01).astype(np.float32))
    del bank
    dump(out/'protocol.json', dict(source=str(relative_path(args.source)),
         definition='only final Align failure end is irrecoverable anchor; success and earlier Keys never positive',
         horizons=list(HORIZONS), seeds=list(SEEDS), fps=30, threshold=.5,
         inputs='dense current Deform frozen feature2560 ->Linear128->single-layer causal GRU128->Linear1',
         loss='train class-balanced BCE: w0=N/(2*N0),w1=N/(2*N1), valid ticks only',
         terminal_mask='H>0 finalfailure t>=anchor excluded from training and main evaluation; H0 allticks',
         optimizer='AdamW lr.001 wd.0001 batch8 max300 patience50 gradclip1',
         selection='each horizon/seed best epoch by natural validation BA, ties earliest; no test selection',
         split={'train':80,'val':17,'test':17}, created_at=datetime.datetime.now().astimezone().isoformat()))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('mean', mean.clone()); self.register_buffer('std', std.clone())
            self.projection = nn.Linear(2560, 128)
            self.gru = nn.GRU(128, 128, batch_first=True)
            self.prediction = nn.Linear(128, 1)

        def forward(self, x):
            x = self.projection((x-self.mean)/self.std)
            x, _ = self.gru(x)
            return self.prediction(x).squeeze(-1)

    def batch(ss, h):
        lengths = [len(s['deform']) for s in ss]
        width = max(lengths)
        x = np.zeros((len(ss),width,2560),np.float32)
        y = np.zeros((len(ss),width),np.float32)
        mask = np.zeros_like(y, dtype=bool)
        for i,s in enumerate(ss):
            x[i,:lengths[i]] = s['deform']
            q,k = target(s['frames'],s['key'],s['outcome']==2,h)
            y[i,:lengths[i]],mask[i,:lengths[i]] = q,k
        return tuple(torch.from_numpy(v).to(args.device) for v in (x,y,mask)),lengths

    def infer(model, ss, h):
        model.eval(); probabilities=[]
        with torch.inference_mode():
            for a in range(0,len(ss),8):
                (x,_,_),lengths=batch(ss[a:a+8],h)
                p=model(x).sigmoid().cpu().numpy()
                probabilities.extend(p[i,:length] for i,length in enumerate(lengths))
        return np.concatenate(probabilities), np.r_[0,np.cumsum([len(p) for p in probabilities])]

    results=[]
    for h in HORIZONS:
        d=next(r for r in dist if r['split']=='train' and r['horizon_frames']==h)
        for seed in SEEDS:
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            model=Model().to(args.device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            rng=np.random.default_rng(seed)
            directory=out/'runs'/f'H{h}'/f'seed_{seed}'
            directory.mkdir(parents=True)
            best=-1;best_epoch=0;history=[]
            for epoch in range(1,301):
                model.train();order=rng.permutation(len(samples['train']));total=0;loss_sum=0
                for a in range(0,len(order),8):
                    ss=[samples['train'][int(i)] for i in order[a:a+8]]
                    (x,y,mask),_=batch(ss,h)
                    optimizer.zero_grad(set_to_none=True)
                    logit=model(x)
                    element=nn.functional.binary_cross_entropy_with_logits(logit,y,reduction='none')
                    weights=torch.where(y>0,d['positive_weight'],d['negative_weight'])
                    loss=(element*weights)[mask].mean()
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                    loss_sum+=float((element*weights)[mask].detach().sum());total+=int(mask.sum())
                p,offsets=infer(model,samples['val'],h)
                val,_,_=evaluate_arrays(samples['val'],p,offsets,h,'val',seed)
                history.append(dict(epoch=epoch,train_balanced_BCE=loss_sum/total,validation_BA=val['balanced_accuracy']))
                if val['balanced_accuracy']>best+1e-8:
                    best=val['balanced_accuracy'];best_epoch=epoch
                    torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                                    horizon=h,seed=seed,best_epoch=epoch),directory/'best.pt')
                if epoch%10==0 or epoch==1:
                    print('EPOCH',epoch,'H',h,'trainBCE',loss_sum/total,'valBA',val['balanced_accuracy'],'best_epoch',best_epoch,flush=True)
                if epoch-best_epoch>=50:
                    break
            model.load_state_dict(torch.load(directory/'best.pt',map_location=args.device,weights_only=True)['state_dict'])
            result=dict(horizon_frames=h,seed=seed,best_epoch=best_epoch,epochs=len(history))
            for split in ('val','test'):
                p,offsets=infer(model,samples[split],h)
                frame,event,events=evaluate_arrays(samples[split],p,offsets,h,split,seed)
                np.savez_compressed(directory/(split+'_predictions.npz'),probabilities=p,offsets=offsets)
                result[split]=frame;result[split+'_event']=event
                write_csv(directory/(split+'_events.csv'),events)
            dump(directory/'metrics.json',result);dump(directory/'history.json',history)
            results.append(result);dump(out/'results_partial.json',results)
            print('RUN',len(results),1,'H',h,'seed',seed,'valBA',round(result['val']['balanced_accuracy'],4),
                  'testBA',round(result['test']['balanced_accuracy'],4),flush=True)
            del model,optimizer
            torch.cuda.empty_cache()
    dump(out/'results.json',results)
    dump(out/'manifest.json',dict(status='complete',runs=1,horizons=list(HORIZONS),seeds=list(SEEDS),
                                 code_sha256=sha(Path(__file__)),completed_at=datetime.datetime.now().astimezone().isoformat()))
    print('TRAIN_COMPLETE',flush=True)


def main():
    global HORIZONS
    p=argparse.ArgumentParser()
    p.add_argument('--horizon',type=int,required=True,choices=HORIZONS)
    p.add_argument('--source',type=Path,default=project_path('outputs/sharpa_gaussian_online_data/20261005_182000'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:2')
    args=p.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    HORIZONS=(args.horizon,)
    train(args)

if __name__=='__main__':main()
