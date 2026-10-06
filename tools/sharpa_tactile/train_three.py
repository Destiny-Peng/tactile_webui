"""Six frozen three-class frame-wise probes; hard targets and argmax evaluation."""
from pathlib import Path
import argparse
import csv
import datetime
import json
import random
import time
import numpy as np
import torch
from torch import nn
from .common import ROOT, INPUTS, HEADS, dump
from .models import FrameProbe
from .train import load_sequences, set_training_normalization

CLASSES = ('background','success','failure')
WEIGHT_MODES = ('inverse_frequency','unweighted','sqrt_inverse')


def class_weights(counts, mode='inverse_frequency'):
    counts = np.asarray(counts,dtype=np.float64)
    if counts.shape != (3,) or (counts <= 0).any():
        raise ValueError('Every training class must have positive support')
    if mode == 'inverse_frequency':
        return counts.sum()/(3*counts)
    if mode == 'unweighted':
        return np.ones(3,dtype=np.float64)
    if mode == 'sqrt_inverse':
        weights = 1/np.sqrt(counts)
        return weights/weights.mean()  # arithmetic mean = 1
    raise ValueError(f'Unknown class-weight mode: {mode}')


def metrics(y,p):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    y = np.asarray(y,dtype=np.int64); p = np.asarray(p,dtype=np.float64)
    assert len(y) and p.shape == (len(y),3) and set(np.unique(y)).issubset({0,1,2})
    assert np.isfinite(p).all() and np.allclose(p.sum(-1),1,atol=1e-5)
    pred = p.argmax(-1)
    precision,recall,f1,support = precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
    cm = confusion_matrix(y,pred,labels=[0,1,2])
    fp = int(cm[:2,2].sum()); non_failure = int(cm[:2].sum())
    return {'failure_false_positives':fp,
        'failure_false_positive_rate':fp/non_failure if non_failure else None,
        'background_to_failure_rate':float(cm[0,2]/cm[0].sum()) if cm[0].sum() else None,
        'success_to_failure_rate':float(cm[1,2]/cm[1].sum()) if cm[1].sum() else None,
        'n':len(y),'decision':'argmax softmax(background,success,failure)',
        'accuracy':float(accuracy_score(y,pred)), 'balanced_accuracy':float(recall.mean()),
        'macro_recall':float(recall.mean()), 'macro_f1':float(f1.mean()),
        'confusion_matrix':confusion_matrix(y,pred,labels=[0,1,2]).tolist(),
        'confusion_matrix_order':list(CLASSES),
        'per_class':{name:{'precision':float(precision[i]),'recall':float(recall[i]),
                     'f1':float(f1[i]),'support':int(support[i])} for i,name in enumerate(CLASSES)}}


def batch_data(sequences,device):
    lengths = torch.tensor([len(s['labels']) for s in sequences],dtype=torch.long)
    shape = (len(sequences),int(lengths.max()))
    batch = {'f6':torch.zeros(*shape,1280),'deform':torch.zeros(*shape,2560),
             'labels':torch.zeros(shape,dtype=torch.long),'lengths':lengths}
    for i,s in enumerate(sequences):
        n = len(s['labels'])
        assert set(np.unique(s['labels'])).issubset({0,1,2})
        for key in ('f6','deform','labels'): batch[key][i,:n] = torch.from_numpy(s[key])
    batch['valid_positions'] = torch.arange(shape[1])[None,:] < lengths[:,None]
    return {key:value.to(device) if key != 'lengths' else value for key,value in batch.items()}


@torch.inference_mode()
def evaluate(model,sequences,device,batch_size):
    model.eval(); ys=[]; ps=[]; details=[]
    for start in range(0,len(sequences),batch_size):
        chosen = sequences[start:start+batch_size]; batch = batch_data(chosen,device)
        logits,_ = model(batch['f6'],batch['deform'],lengths=batch['lengths'])
        probabilities = logits.softmax(-1).cpu().numpy()
        for i,s in enumerate(chosen):
            p = probabilities[i,:len(s['labels'])]; ys.extend(s['labels'].tolist()); ps.extend(p.tolist())
            details.extend({'rollout_id':s['rollout_id'],'tick':int(t),'video_frame':int(f),'label':int(y),
                'prediction':int(v.argmax()),**{name+'_probability':float(v[k]) for k,name in enumerate(CLASSES)}}
                for t,f,y,v in zip(s['ticks'],s['video_frames'],s['labels'],p))
    return np.asarray(ys),np.asarray(ps),details


def plot_run(directory,history,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,3,figsize=(14,4))
    epochs = [row['epoch'] for row in history]
    axes[0].plot(epochs,[row['train_loss'] for row in history]); axes[0].set_title('3-class CE: '+result.get('weight_mode','inverse_frequency')); axes[0].set_xlabel('Epoch')
    axes[1].plot(epochs,[row['val_balanced_accuracy'] for row in history],label='val macro recall')
    axes[1].plot(epochs,[row['val_macro_f1'] for row in history],label='val macro F1')
    axes[1].legend(); axes[1].set_ylim(0,1); axes[1].set_xlabel('Epoch')
    cm = np.asarray(result['test']['confusion_matrix']); axes[2].imshow(cm,cmap='Blues')
    axes[2].set_xticks(range(3),CLASSES,rotation=20); axes[2].set_yticks(range(3),CLASSES)
    axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('True'); axes[2].set_title('Test confusion matrix')
    for i in range(3):
        for j in range(3): axes[2].text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.tight_layout(); fig.savefig(directory/'curves.png',dpi=150); plt.close(fig)


def train_group(args,input_kind,head_kind,sequences,data):
    name = input_kind+'_'+head_kind; directory = args.output/name; directory.mkdir(exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    config = {'input_kind':input_kind,'head_kind':head_kind,'hidden':args.hidden,'layers':args.layers,'num_classes':3}
    model = FrameProbe(**config); set_training_normalization(model,sequences['train']); model.to(args.device)
    labels = np.concatenate([s['labels'] for s in sequences['train']]); counts = np.bincount(labels,minlength=3)
    if (counts == 0).any(): raise ValueError('Training split lacks a class')
    weight_mode = getattr(args,'weight_mode','inverse_frequency')
    weights = class_weights(counts,weight_mode)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=args.device))
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    best_score = -1; best_epoch = 0; history = []; started = time.monotonic(); rng = np.random.default_rng(args.seed)
    for epoch in range(1,args.epochs+1):
        model.train(); order = rng.permutation(len(sequences['train'])); total_loss = 0; total_weight = 0
        for first in range(0,len(order),args.batch_size):
            chosen = [sequences['train'][j] for j in order[first:first+args.batch_size]]
            batch = batch_data(chosen,args.device); optimizer.zero_grad(set_to_none=True)
            logits,_ = model(batch['f6'],batch['deform'],lengths=batch['lengths'])
            mask = batch['valid_positions']; target = batch['labels'][mask]
            loss = loss_fn(logits[mask],target)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            weight = float(loss_fn.weight[target].sum()); total_loss += float(loss.detach())*weight; total_weight += weight
        y,p,_ = evaluate(model,sequences['val'],args.device,args.batch_size); validation = metrics(y,p)
        score = validation['balanced_accuracy']
        row = {'epoch':epoch,'train_loss':total_loss/total_weight,'val_balanced_accuracy':score,
               'val_macro_f1':validation['macro_f1'],'elapsed_seconds':time.monotonic()-started}
        history.append(row); print(name,json.dumps(row),flush=True)
        if score > best_score+1e-8:
            best_score = score; best_epoch = epoch
            torch.save({'model_config':config,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'best_epoch':best_epoch,'seed':args.seed,'selection':'validation 3-class macro recall (argmax)',
                'encoder_sha256':{'f6':data['signature']['f6_sha256'],'deform':data['signature']['deform_sha256']},
                'class_mapping':{'0':'background','1':'success','2':'failure'},'training_class_counts':counts.tolist(),
                'training_class_weights':weights.tolist(),'weight_mode':weight_mode,'split_sha256':data['signature']['split_sha256'],
                'normalization':'all valid training frames; std floor 0.01'},directory/'best.pt')
        if epoch-best_epoch >= args.patience: break
    saved = torch.load(directory/'best.pt',map_location=args.device,weights_only=True); model.load_state_dict(saved['state_dict'])
    vy,vp,_ = evaluate(model,sequences['val'],args.device,args.batch_size)
    ty,tp,details = evaluate(model,sequences['test'],args.device,args.batch_size)
    result = {'name':name,'model_config':config,'trainable_parameters':sum(p.numel() for p in model.parameters()),
        'seed':args.seed,'best_epoch':best_epoch,'epochs_run':len(history),'training_class_counts':counts.tolist(),
        'training_class_weights':weights.tolist(),'weight_mode':weight_mode,'validation':metrics(vy,vp),'test':metrics(ty,tp),
        'duration_seconds':time.monotonic()-started,'test_by_task':{}}
    for task in sorted({r['task_key'] for r in data['records'].values()}):
        rows = [r for r in details if data['records'][r['rollout_id']]['task_key'] == task]
        if rows: result['test_by_task'][task] = metrics([r['label'] for r in rows],
            [[r[name+'_probability'] for name in CLASSES] for r in rows])
    dump(directory/'history.json',history); dump(directory/'metrics.json',result)
    with (directory/'test_predictions.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(details[0])); writer.writeheader(); writer.writerows(details)
    plot_run(directory,history,result); print('GROUP_COMPLETE',name,flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--epochs',type=int,default=30); parser.add_argument('--patience',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=8); parser.add_argument('--hidden',type=int,choices=[128,256],default=128)
    parser.add_argument('--layers',type=int,choices=[1,2],default=1); parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--weight-mode',choices=WEIGHT_MODES,default='inverse_frequency')
    parser.add_argument('--lr',type=float,default=0.001); parser.add_argument('--weight-decay',type=float,default=0.0001)
    args = parser.parse_args(); args.output = args.output.resolve(); torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    data = json.loads((args.output/'data_manifest.json').read_text()); assert data['status'] == 'complete'
    assert data['signature']['num_classes'] == 3
    split = json.loads((args.output/'split_manifest.json').read_text())
    sequences = {name:load_sequences(args.output,split[name]) for name in ('train','val','test')}
    dump(args.output/'training_config.json',{**vars(args),'output':str(args.output.relative_to(ROOT)),
        'created_at':datetime.datetime.now().astimezone().isoformat(),'groups':6,'encoder_finetuning':False,
        'class_balance':args.weight_mode+' from train only','split_seed':split['seed'],'loss':'weighted CE, hard labels, no ignore, no smoothing',
        'padding':'length mask before CE; padding is not a data frame','selection':'validation macro recall, argmax',
        'fusion':'simple concat','lstm':'unidirectional; reset at episode/gap'})
    results = []
    for input_kind in INPUTS:
        for head_kind in HEADS:
            results.append(train_group(args,input_kind,head_kind,sequences,data)); dump(args.output/'results.json',results)
    fields = ['group','best_epoch','test_balanced_accuracy','test_macro_f1']+[name+'_'+metric for name in CLASSES for metric in ('precision','recall','f1')]
    with (args.output/'comparison.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in results: writer.writerow({'group':r['name'],'best_epoch':r['best_epoch'],
            'test_balanced_accuracy':r['test']['balanced_accuracy'],'test_macro_f1':r['test']['macro_f1'],
            **{name+'_'+metric:r['test']['per_class'][name][metric] for name in CLASSES for metric in ('precision','recall','f1')}})
    print('ALL_SIX_THREE_CLASS_COMPLETE',flush=True)


if __name__ == '__main__': main()
