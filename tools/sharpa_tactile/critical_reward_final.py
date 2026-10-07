"""Only final Align failure Key per rollout: binary critical reward sweep."""
from .common import project_path, relative_path
import argparse
import csv
import datetime
import json
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import ROOT,dump,sha
from .merged_online import MergedProbe,load,save
from .gaussian_online import read_samples
from .train_intervals import normalization


def key_labels(frames,key,outcome,n,m,rule='last'):
    if outcome!=2:return np.zeros(len(frames),dtype=np.int64)
    inside=(frames>=key-n)&(frames<=key+m)
    return inside.astype(np.int64) if rule=='last' else inside.any(1).astype(np.int64)


def binary_metrics(y,p):
    y=np.asarray(y,dtype=np.int64);p=np.asarray(p);pred=(p>=.5).astype(int)
    cm=np.array([[int(((y==c)&(pred==k)).sum()) for k in (0,1)] for c in (0,1)])
    recall=np.divide(np.diag(cm),cm.sum(1),out=np.zeros(2),where=cm.sum(1)!=0);precision=np.divide(np.diag(cm),cm.sum(0),out=np.zeros(2),where=cm.sum(0)!=0);f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(2),where=precision+recall!=0)
    return dict(n=len(y),accuracy=float((pred==y).mean()),balanced_accuracy=float(recall.mean()),macro_f1=float(f1.mean()),precision=float(precision[1]),recall=float(recall[1]),false_positive_rate=float(cm[0,1]/cm[0].sum()) if cm[0].sum() else None,confusion_matrix=cm.tolist(),class_order=['noncritical','critical_failure'],positive_count=int(y.sum()),negative_count=int((y==0).sum()))


def write_csv(path,rows):
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def label_sample(s,n,m,rule):
    frames=s['video_frames'] if rule=='last' else s['rear_frames']
    return key_labels(frames,s['key'],s['outcome'],n,m,rule)


def batch(ss,n,m,rule,device):
    lengths=np.array([len(s['f6']) for s in ss]);width=int(lengths.max());f6=np.zeros((len(ss),width,1280),np.float32);q=np.zeros((len(ss),width),np.float32)
    for i,s in enumerate(ss):f6[i,:lengths[i]]=s['f6'];q[i,:lengths[i]]=label_sample(s,n,m,rule)
    mask=torch.arange(width,device=device)[None]<torch.as_tensor(lengths,device=device)[:,None]
    return torch.from_numpy(f6).to(device),torch.from_numpy(q).to(device),mask,lengths


class CriticalProbe(MergedProbe):
    def __init__(self):
        super().__init__('f6','gru');self.prediction=nn.Linear(128,1)
    def forward(self,f6=None,state=None):
        value,state=super().forward(f6=f6,state=state);return value.squeeze(-1),state


@torch.inference_mode()
def evaluate(model,ss,n,m,rule,device):
    model.eval();ps=[]
    for start in range(0,len(ss),8):
        chosen=ss[start:start+8];x,_,_,lengths=batch(chosen,n,m,rule,device);logits,_=model(x);prob=logits.sigmoid().cpu().numpy()
        ps.extend(prob[i,:lengths[i]] for i in range(len(chosen)))
    p=np.concatenate(ps);y=np.concatenate([label_sample(s,n,m,rule) for s in ss]);fm=binary_metrics(y,p)
    event_y=np.array([s['outcome']==2 for s in ss],int);event_p=np.array([v.max() for v in ps]);em=binary_metrics(event_y,event_p)
    offsets=np.r_[0,np.cumsum([len(v) for v in ps])];return fm,em,p,y,offsets


def read_data(source,rule):
    manifest=json.loads((source/'dataset_manifest.json').read_text());samples=read_samples(source,manifest,'merged_align')
    if rule=='rear':
        upstream=json.loads((project_path('outputs/sharpa_tactile_three_class/20261004_160624/data_manifest.json')).read_text());camera=upstream['signature']['camera'];maps={}
        for ss in samples.values():
            for s in ss:
                rid=s['rollout_id']
                if rid not in maps:
                    record=upstream['records'][rid];rows=[json.loads(x) for x in (project_path(record['synchronized_frames_path'])).read_text().splitlines()];maps[rid]=np.array([r['camera_frame_indices'].get(camera,-1) for r in rows],int)
                ticks=np.maximum(s['ticks'][:,None]+np.arange(-7,1)[None],s['ticks'][0]);s['rear_frames']=maps[rid][ticks];assert np.all(s['rear_frames']>=s['start'])
    return manifest,samples


def audit(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);manifest,samples=read_data(args.source,args.rule);rows=[]
    for n in range(31):
        for m in range(16):
            for split,ss in samples.items():
                y=np.concatenate([label_sample(s,n,m,args.rule) for s in ss]);rows.append(dict(n=n,m=m,split=split,positive=int(y.sum()),negative=int((y==0).sum()),total=len(y),positive_fraction=float(y.mean())))
    write_csv(out/'all_integer_label_distribution.csv',rows);dump(out/'audit.json',dict(status='labels_audited_pending_sweep_choice',rule=args.rule,source=str(relative_path(args.source)),label='only final Align failure Key per rollout1; allsuccess andearlierintervals0; closedend-n:end+m; originalannotationrangeunchanged',frame_units=True,all_496_label_configurations=True))
    print('AUDIT_COMPLETE',len(rows),flush=True)


def train(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);manifest,samples=read_data(args.source,args.rule);norm=normalization(samples['train']);ns=list(range(31)) if args.grid=='integer' else [0,5,10,15,20,25,30];ms=list(range(16)) if args.grid=='integer' else [0,3,5,10,15];configs=[(n,m) for n in ns for m in ms]
    dump(out/'protocol.json',dict(source=str(relative_path(args.source)),rule=args.rule,ns=ns,ms=ms,seeds=list(range(42,47)),models='frozen dense F6->Linear128->causal GRU128->Linear1->sigmoid; one sequence per rollout, reset at rolloutstart',loss='unweighted BCEWithLogits; all valid ticks; no Gaussian/classweights/stepaug',label='only final Align Key per rollout; finalfailure end-n:end+m positive; finalsuccess andearlyintervals have no Keys; allother ticks0; clip to rollout annotation observation range',optimizer='AdamWlr.001wd.0001batch8max30epochs/patience8clip1',threshold=.5,selection='best epoch by own-band validation binary BA; n/m by mean5seed valBA, ties macroF1 then smaller n+m,n,m',GT_caution='each n/m changes frame GT; event any score>=.5 evaluated against fixed interaction outcomes'))
    results=[]
    for n,m in configs:
        for seed in range(42,47):
            directory=out/'runs'/f'n{n}_m{m}'/f'seed_{seed}';directory.mkdir(parents=True);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);model=CriticalProbe()
            for k,v in norm.items():getattr(model,k).copy_(v)
            model.to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);best=-1;best_epoch=0;history=[]
            for epoch in range(1,31):
                model.train();order=rng.permutation(len(samples['train']));loss_sum=0;total=0
                for start in range(0,len(order),8):
                    chosen=[samples['train'][int(i)] for i in order[start:start+8]];x,q,mask,_=batch(chosen,n,m,args.rule,args.device);optimizer.zero_grad(set_to_none=True);logits,_=model(x);element=nn.functional.binary_cross_entropy_with_logits(logits,q,reduction='none');loss=element[mask].mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();loss_sum+=float(element[mask].detach().sum());total+=int(mask.sum())
                val,_,_,_,_=evaluate(model,samples['val'],n,m,args.rule,args.device);history.append(dict(epoch=epoch,train_BCE=loss_sum/total,val_BA=val['balanced_accuracy'],val_macro_F1=val['macro_f1']))
                if val['balanced_accuracy']>best+1e-8:
                    best=val['balanced_accuracy'];best_epoch=epoch;torch.save(dict(model_config='CriticalProbe',state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},n=n,m=m,rule=args.rule,seed=seed,best_epoch=epoch),directory/'best.pt')
                if epoch-best_epoch>=8:break
            blob=torch.load(directory/'best.pt',map_location=args.device,weights_only=True);model.load_state_dict(blob['state_dict']);val,ve,vp,vy,vo=evaluate(model,samples['val'],n,m,args.rule,args.device);test,te,p,y,offsets=evaluate(model,samples['test'],n,m,args.rule,args.device)
            save(directory/'test_predictions.npz',dict(probabilities=p,labels=y,offsets=offsets));save(directory/'val_predictions.npz',dict(probabilities=vp,labels=vy,offsets=vo));r=dict(n=n,m=m,seed=seed,best_epoch=best_epoch,epochs=len(history),validation=val,test=test,validation_event=ve,test_event=te);results.append(r);dump(directory/'metrics.json',r);dump(directory/'history.json',history);dump(out/'results_partial.json',results);print('RUN',len(results),len(configs)*5,n,m,seed,'valBA',round(val['balanced_accuracy'],4),'testBA',round(test['balanced_accuracy'],4),flush=True);del model,optimizer;torch.cuda.empty_cache()
    dump(out/'results.json',results);summary=[]
    for n,m in configs:
        rr=[r for r in results if r['n']==n and r['m']==m];row=dict(n=n,m=m,seeds=5)
        for phase in ('validation','test','validation_event','test_event'):
            for key in ('balanced_accuracy','macro_f1','precision','recall','false_positive_rate'):
                vv=[r[phase][key] for r in rr];row[phase+'_'+key+'_mean']=float(np.mean(vv));row[phase+'_'+key+'_std']=float(np.std(vv,ddof=1))
        row['test_positive']=rr[0]['test']['positive_count'];row['test_negative']=rr[0]['test']['negative_count'];summary.append(row)
    selected=max(summary,key=lambda r:(r['validation_balanced_accuracy_mean'],r['validation_macro_f1_mean'],-r['n']-r['m'],-r['n'],-r['m']));dump(out/'selected_config.json',selected);write_csv(out/'summary.csv',summary);dump(out/'manifest.json',dict(status='complete',created_at=datetime.datetime.now().astimezone().isoformat(),configs=len(configs),runs=len(results),source=str(relative_path(args.source)),scope='final Align Key per rollout only',rule=args.rule,code_sha256=sha(Path(__file__))))
    print('TRAIN_COMPLETE',json.dumps(selected),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','train']);p.add_argument('--source',type=Path,default=project_path('outputs/sharpa_gaussian_online_data/20261005_182000'));p.add_argument('--output',type=Path,required=True);p.add_argument('--rule',choices=['last','rear'],default='last');p.add_argument('--grid',choices=['coarse','integer'],default='coarse');p.add_argument('--device',default='cuda:0');args=p.parse_args();args.source=args.source.resolve();args.output=args.output.resolve();torch.set_num_threads(4);(audit if args.action=='audit' else train)(args)

if __name__=='__main__':main()
