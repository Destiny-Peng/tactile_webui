from __future__ import annotations
import argparse
import copy
import csv
import datetime
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from .common import ROOT, INPUTS, HEADS, dump
from .models import BinaryProbe


def metrics(y, p, threshold=0.5):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix, precision_score, recall_score)
    y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);pred=(p>=threshold).astype(int)
    both=len(np.unique(y))==2
    return {'n':len(y),'failure_n':int(y.sum()),'success_n':int((y==0).sum()),
        'threshold':float(threshold),'accuracy':float(accuracy_score(y,pred)),
        'balanced_accuracy':float(balanced_accuracy_score(y,pred)) if both else None,
        'macro_f1':float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
        'failure_precision':float(precision_score(y,pred,zero_division=0)),
        'failure_recall':float(recall_score(y,pred,zero_division=0)),
        'failure_f1':float(f1_score(y,pred,zero_division=0)),
        'auroc':float(roc_auc_score(y,p)) if both else None,
        'failure_auprc':float(average_precision_score(y,p)) if both else None,
        'confusion_matrix_success_failure':confusion_matrix(y,pred,labels=[0,1]).tolist()}


def load_sequences(output, ids):
    result=[]
    for rid in ids:
        with np.load(output/'features'/(rid+'.npz')) as data:
            starts=np.flatnonzero(data['segment_starts'])
            for first,last in zip(starts,np.r_[starts[1:],len(data['labels'])]):
                result.append({'rollout_id':rid,**{key:data[key][first:last].copy()
                    for key in ('f6','deform','labels','video_frames','ticks')}})
    return result


def batch_data(sequences, device):
    lengths=torch.tensor([len(s['labels']) for s in sequences],dtype=torch.long)
    shape=(len(sequences),int(lengths.max()))
    batch={'f6':torch.zeros(*shape,1280),'deform':torch.zeros(*shape,2560),
           'labels':torch.full(shape,-1,dtype=torch.long),'lengths':lengths}
    for i,s in enumerate(sequences):
        n=len(s['labels'])
        for key in ('f6','deform','labels'):
            batch[key][i,:n]=torch.from_numpy(s[key])
    return {key:value.to(device) if key!='lengths' else value for key,value in batch.items()}


def set_training_normalization(model, train):
    for name in ('f6','deform'):
        rows=np.concatenate([s[name][s['labels']>=0] for s in train])
        mean=rows.mean(0,dtype=np.float64).astype(np.float32)
        std=rows.std(0,dtype=np.float64).astype(np.float32)
        # A 0.01 scale floor prevents near-constant channels from magnifying
        # backend roundoff into artificial temporal or classification signals.
        std=np.maximum(std,0.01)
        getattr(model,name+'_mean').copy_(torch.from_numpy(mean))
        getattr(model,name+'_std').copy_(torch.from_numpy(std))


@torch.inference_mode()
def evaluate(model, sequences, device, batch_size):
    model.eval();ys=[];ps=[];details=[]
    for start in range(0,len(sequences),batch_size):
        chosen=sequences[start:start+batch_size];batch=batch_data(chosen,device)
        logits,_=model(batch['f6'],batch['deform'],lengths=batch['lengths'])
        probabilities=logits.softmax(-1)[...,1].cpu().numpy()
        for i,s in enumerate(chosen):
            valid=s['labels']>=0;p=probabilities[i,:len(s['labels'])]
            ys.extend(s['labels'][valid].tolist());ps.extend(p[valid].tolist())
            details.extend({'rollout_id':s['rollout_id'],'tick':int(t),'video_frame':int(f),
                            'label':int(y),'failure_probability':float(v)}
                for t,f,y,v in zip(s['ticks'][valid],s['video_frames'][valid],s['labels'][valid],p[valid]))
    return np.asarray(ys),np.asarray(ps),details


def choose_threshold(y,p):
    values=[]
    for threshold in np.arange(0.05,0.951,0.01):
        score=metrics(y,p,threshold)['balanced_accuracy']
        values.append((score,-abs(threshold-0.5),threshold))
    return float(max(values)[2])


def interval_metrics(details, events, ids, threshold):
    grouped={rid:[] for rid in ids}
    for row in details:grouped[row['rollout_id']].append(row)
    targets=[];probabilities=[];rows=[]
    for e in events:
        if e['rollout_id'] not in grouped:continue
        p=[r['failure_probability'] for r in grouped[e['rollout_id']]
           if e['start_frame']<=r['video_frame']<=e['end_frame']]
        if not p:continue
        y=int(int(e['event_key']) in (6,7));average=float(np.mean(p))
        targets.append(y);probabilities.append(average)
        rows.append({'rollout_id':e['rollout_id'],'event_index':e['event_index'],'event_key':e['event_key'],
                     'label':y,'failure_probability':average,'samples':len(p)})
    return metrics(targets,probabilities,threshold),rows


def plot_run(directory, history, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['train_loss'] for r in history],label='train CE')
    axes[0].set_title('Class-weighted training loss');axes[0].set_xlabel('Epoch')
    axes[1].plot(epochs,[r['val_balanced_accuracy'] for r in history],label='val balanced accuracy')
    axes[1].plot(epochs,[r['val_auroc'] for r in history],label='val AUROC')
    axes[1].set_ylim(0,1);axes[1].legend();axes[1].set_xlabel('Epoch')
    cm=np.asarray(result['test']['confusion_matrix_success_failure'])
    axes[2].imshow(cm,cmap='Blues');axes[2].set_xticks([0,1],['success','failure']);axes[2].set_yticks([0,1],['success','failure'])
    axes[2].set_xlabel('Predicted');axes[2].set_ylabel('True');axes[2].set_title('Held-out test confusion matrix')
    for i in range(2):
        for j in range(2):axes[2].text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.tight_layout();fig.savefig(directory/'curves.png',dpi=150);plt.close(fig)


def train_group(args, input_kind, head_kind, sequences, split, data):
    name=input_kind+'_'+head_kind;directory=args.output/name;directory.mkdir(exist_ok=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    config={'input_kind':input_kind,'head_kind':head_kind,'hidden':args.hidden,'layers':args.layers}
    model=BinaryProbe(**config)
    set_training_normalization(model,sequences['train']);model.to(args.device)
    labels=np.concatenate([s['labels'][s['labels']>=0] for s in sequences['train']])
    counts=np.bincount(labels,minlength=2)
    weights=len(labels)/(2*counts)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=args.device),ignore_index=-1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best_score=-1;best_epoch=0;history=[];started=time.monotonic()
    rng=np.random.default_rng(args.seed)
    for epoch in range(1,args.epochs+1):
        model.train();order=rng.permutation(len(sequences['train']));total_loss=0;total_n=0
        for first in range(0,len(order),args.batch_size):
            chosen=[sequences['train'][j] for j in order[first:first+args.batch_size]]
            if not any((s['labels']>=0).any() for s in chosen):continue
            batch=batch_data(chosen,args.device);optimizer.zero_grad(set_to_none=True)
            logits,_=model(batch['f6'],batch['deform'],lengths=batch['lengths'])
            loss=loss_fn(logits.reshape(-1,2),batch['labels'].flatten())
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            n=int((batch['labels']>=0).sum());total_loss+=float(loss.detach())*n;total_n+=n
        y,p,_=evaluate(model,sequences['val'],args.device,args.batch_size)
        validation=metrics(y,p);score=validation['balanced_accuracy']
        row={'epoch':epoch,'train_loss':total_loss/total_n,'val_balanced_accuracy':score,
             'val_auroc':validation['auroc'],'elapsed_seconds':time.monotonic()-started}
        history.append(row);print(name,json.dumps(row),flush=True)
        if score>best_score+1e-8:
            best_score=score;best_epoch=epoch
            state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            torch.save({'model_config':config,'state_dict':state,'best_epoch':best_epoch,
                'selection':'validation balanced accuracy at threshold 0.5','seed':args.seed,
                'encoder_sha256':{'f6':data['signature']['f6_sha256'],'deform':data['signature']['deform_sha256']},
                'binary_mapping':{'0':'success (8/9)','1':'failure (6/7)'},
                'split_manifest':'split_manifest.json','normalization':'training supervised frames only; std floor 0.01',
                'architecture':'frozen continuous F6 + frozen deform 2x2 pooling; trainable projections and probe'},directory/'best.pt')
        if epoch-best_epoch>=args.patience:break
    saved=torch.load(directory/'best.pt',map_location=args.device,weights_only=True)
    model.load_state_dict(saved['state_dict']);model.eval()
    vy,vp,_=evaluate(model,sequences['val'],args.device,args.batch_size)
    threshold=choose_threshold(vy,vp)
    ty,tp,details=evaluate(model,sequences['test'],args.device,args.batch_size)
    result={'name':name,'model_config':config,'trainable_parameters':sum(p.numel() for p in model.parameters()),
            'seed':args.seed,'best_epoch':best_epoch,'epochs_run':len(history),'training_class_counts':counts.tolist(),
            'training_class_weights':weights.tolist(),'validation_selected_threshold':threshold,
            'validation':metrics(vy,vp,threshold),'test_at_0_5':metrics(ty,tp),
            'test':metrics(ty,tp,threshold),'duration_seconds':time.monotonic()-started}
    result['test_intervals'],interval_rows=interval_metrics(details,data['source_intervals'],split['test'],threshold)
    result['test_by_task']={}
    for task in sorted({r['task_key'] for r in data['records'].values()}):
        rows=[r for r in details if data['records'][r['rollout_id']]['task_key']==task]
        if rows:result['test_by_task'][task]=metrics([r['label'] for r in rows],[r['failure_probability'] for r in rows],threshold)
    # Save deployment threshold selected only from validation, never from test.
    saved['failure_threshold']=threshold;torch.save(saved,directory/'best.pt')
    dump(directory/'history.json',history);dump(directory/'metrics.json',result)
    dump(directory/'test_interval_predictions.json',interval_rows)
    with (directory/'test_predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    plot_run(directory,history,result)
    print('GROUP_COMPLETE',json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description='Train all six frozen Sharpa tactile binary probes')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--patience',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--hidden',type=int,choices=[128,256],default=128)
    parser.add_argument('--layers',type=int,choices=[1,2],default=1)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--weight-decay',type=float,default=1e-4)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    data=json.loads((args.output/'data_manifest.json').read_text())
    if data['status']!='complete':raise ValueError('Feature extraction incomplete')
    sanity=json.loads((args.output/'sanity.json').read_text())
    if sanity['status']!='executed':raise ValueError('F6 reconstruction check not executed')
    split=json.loads((args.output/'split_manifest.json').read_text())
    if args.seed!=split['seed']:raise ValueError('Training seed must match saved split seed')
    sequences={name:load_sequences(args.output,split[name]) for name in ('train','val','test')}
    dump(args.output/'training_config.json',{**vars(args),'output':str(args.output.relative_to(ROOT)),
         'created_at':datetime.datetime.now().astimezone().isoformat(),'groups':6,
         'class_balance':'weighted CE; weights fitted on training labels',
         'selection':'validation balanced accuracy; threshold calibrated on validation',
         'encoder_finetuning':False,'fusion':'simple concat','lstm':'unidirectional; reset at episode/gap'})
    results=[]
    for input_kind in INPUTS:
        for head_kind in HEADS:
            results.append(train_group(args,input_kind,head_kind,sequences,split,data))
            dump(args.output/'results.json',results)
    fields=['group','best_epoch','test_balanced_accuracy_at_0_5','test_macro_f1_at_0_5','test_balanced_accuracy_val_threshold','validation_threshold','test_auroc','test_failure_auprc','interval_balanced_accuracy_val_threshold']
    with (args.output/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in results:writer.writerow({'group':r['name'],'best_epoch':r['best_epoch'],
            'test_balanced_accuracy_at_0_5':r['test_at_0_5']['balanced_accuracy'],
            'test_macro_f1_at_0_5':r['test_at_0_5']['macro_f1'],
            'test_balanced_accuracy_val_threshold':r['test']['balanced_accuracy'],
            'validation_threshold':r['validation_selected_threshold'],
            **{'test_'+key:r['test'][key] for key in ('auroc','failure_auprc')},
            'interval_balanced_accuracy_val_threshold':r['test_intervals']['balanced_accuracy']})
    print('ALL_SIX_COMPLETE',flush=True)


if __name__=='__main__':main()
