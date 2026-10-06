"""Binary interval supervision: one sample, one output and one loss per interval."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from .common import ROOT, INPUTS, HEADS, dump, sha
from .interval_models import IntervalProbe
from .train import metrics as binary_metrics


def metrics(y,p):
    from sklearn.metrics import precision_recall_fscore_support
    result=binary_metrics(y,p,threshold=.5)
    precision,recall,f1,support=precision_recall_fscore_support(y,np.asarray(p)>=.5,labels=[0,1],zero_division=0)
    result['per_class']={name:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])}
                         for i,name in enumerate(('success','failure'))}
    cm=np.array(result['confusion_matrix_success_failure'])
    result['failure_false_positives']=int(cm[0,1]); result['failure_false_positive_rate']=float(cm[0,1]/cm[0].sum()) if cm[0].sum() else None
    result['unit']='interval'; return result


def load_samples(output,manifest):
    samples={name:[] for name in ('train','val','test')}
    for row in manifest['records']:
        with np.load(output/row['feature_path']) as features:
            assert int(features['label'])==row['label'] and len(features['ticks'])==row['ticks']
            samples[row['split']].append({**row,'f6':features['f6'].copy(),'deform':features['deform'].copy()})
    return samples


def batch_data(samples,device):
    lengths=torch.tensor([len(s['f6']) for s in samples],dtype=torch.long); shape=(len(samples),int(lengths.max()))
    f6=torch.zeros(*shape,1280); deform=torch.zeros(*shape,2560)
    for i,s in enumerate(samples):
        f6[i,:lengths[i]]=torch.from_numpy(s['f6']); deform[i,:lengths[i]]=torch.from_numpy(s['deform'])
    return {'f6':f6.to(device),'deform':deform.to(device),'lengths':lengths,
            'labels':torch.tensor([s['label'] for s in samples],dtype=torch.long,device=device)}


def normalization(samples):
    result={}
    for modality in ('f6','deform'):
        frames=np.concatenate([s[modality] for s in samples])
        result[modality+'_mean']=torch.from_numpy(frames.mean(0,dtype=np.float64).astype(np.float32))
        result[modality+'_std']=torch.from_numpy(np.maximum(frames.std(0,dtype=np.float64).astype(np.float32),.01))
    return result


@torch.inference_mode()
def evaluate(model,samples,device,batch_size):
    model.eval(); labels=[]; probabilities=[]; details=[]
    for start in range(0,len(samples),batch_size):
        chosen=samples[start:start+batch_size]; batch=batch_data(chosen,device)
        p=model(batch['f6'],batch['deform'],lengths=batch['lengths']).softmax(-1)[:,1].cpu().numpy()
        for s,value in zip(chosen,p):
            labels.append(s['label']); probabilities.append(float(value))
            details.append({key:s[key] for key in ('interval_id','rollout_id','event_index','event_key','start_frame','end_frame','label','ticks')})
            details[-1].update(failure_probability=float(value),prediction=int(value>=.5))
    return metrics(labels,probabilities),details


def train_group(args,kind,head,seed,samples,norm,manifest):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    name=kind+'_'+head; directory=args.output/f'seed_{seed}'/name; directory.mkdir(parents=True,exist_ok=True)
    config={'input_kind':kind,'head_kind':head,'hidden':128,'layers':1}
    model=IntervalProbe(**config)
    for key,value in norm.items(): getattr(model,key).copy_(value)
    model.to(args.device)
    counts=np.bincount([s['label'] for s in samples['train']],minlength=2)
    weights=len(samples['train'])/(2*counts)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=args.device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed); best_score=-1; best_epoch=0; history=[]
    for epoch in range(1,31):
        model.train(); order=rng.permutation(len(samples['train'])); loss_sum=0; weight_sum=0
        for start in range(0,len(order),8):
            chosen=[samples['train'][i] for i in order[start:start+8]]; batch=batch_data(chosen,args.device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(batch['f6'],batch['deform'],lengths=batch['lengths']); assert logits.shape==(len(chosen),2)
            loss=loss_fn(logits,batch['labels']); assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
            w=float(loss_fn.weight[batch['labels']].sum()); loss_sum+=float(loss.detach())*w; weight_sum+=w
        validation,_=evaluate(model,samples['val'],args.device,8); score=validation['balanced_accuracy']
        history.append({'epoch':epoch,'train_loss':loss_sum/weight_sum,'val_balanced_accuracy':score,'val_macro_f1':validation['macro_f1']})
        print(seed,name,json.dumps(history[-1]),flush=True)
        if score>best_score+1e-8:
            best_score=score; best_epoch=epoch
            torch.save({'model_class':'IntervalProbe','model_config':config,'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                'seed':seed,'best_epoch':epoch,'label_mapping':{'0':'success','1':'failure'},'output_unit':'interval',
                'training_interval_counts':counts.tolist(),'class_weights':weights.tolist(),'failure_threshold':.5,
                'selection':'validation interval balanced accuracy at .5','f6_context':manifest['signature']['f6_context'],
                'split_sha256':manifest['signature']['split_sha256'],
                'encoder_sha256':{k:manifest['signature'][k+'_sha256'] for k in ('f6','deform')},
                'pooling':'mean after timestep projection' if head=='mlp' else 'final hidden state of unidirectional LSTM'},directory/'best.pt')
        if epoch-best_epoch>=8: break
    checkpoint=torch.load(directory/'best.pt',weights_only=True,map_location=args.device); model.load_state_dict(checkpoint['state_dict'])
    validation,_=evaluate(model,samples['val'],args.device,8); test,details=evaluate(model,samples['test'],args.device,8)
    result={'name':name,'seed':seed,'model_config':config,'best_epoch':best_epoch,'epochs_run':len(history),
        'trainable_parameters':sum(p.numel() for p in model.parameters()),'training_interval_counts':counts.tolist(),
        'training_class_weights':weights.tolist(),'validation':validation,'test':test}
    dump(directory/'history.json',history); dump(directory/'metrics.json',result)
    with (directory/'test_predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0])); writer.writeheader(); writer.writerows(details)
    print('INTERVAL_GROUP_COMPLETE',seed,name,flush=True); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:1'); parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    args=parser.parse_args(); args.output=args.output.resolve(); torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False; torch.set_float32_matmul_precision('highest')
    manifest=json.loads((args.output/'interval_manifest.json').read_text()); assert manifest['status']=='complete'
    assert len(args.seeds)==len(set(args.seeds))
    samples=load_samples(args.output,manifest); norm=normalization(samples['train'])
    dump(args.output/'training_config.json',{'seeds':args.seeds,'device':args.device,'unit':'interval',
        'class_weight':'N_intervals/(2*n_class_intervals)','epochs':30,'patience':8,'batch_size':8,'hidden':128,'layers':1,
        'lr':.001,'weight_decay':.0001,'gradient_clip':1,'fusion':'timestep concat','threshold':.5,
        'split_sha256':manifest['signature']['split_sha256'],'encoder_finetuning':False,
        'normalization':'train interval features only, std floor .01','created_at':datetime.datetime.now().astimezone().isoformat()})
    results=[]
    for seed in args.seeds:
        for kind in INPUTS:
            for head in HEADS:
                results.append(train_group(args,kind,head,seed,samples,norm,manifest)); dump(args.output/'results.json',results)
    assert sha(args.output/'split_manifest.json')==manifest['signature']['split_sha256']
    print('ALL_INTERVAL_GROUPS_COMPLETE',len(results),flush=True)


if __name__=='__main__': main()
