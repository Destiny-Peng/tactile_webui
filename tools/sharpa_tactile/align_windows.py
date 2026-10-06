"""Train frozen tactile MLP/GRU probes on Align Key-labelled sliding windows."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from .common import ROOT, INPUTS, dump, sha
from .models import FrameProbe
from .align_windows_data import CLASSES, WINDOW, load_events, labels_for, expanded_counts
from .train_three import class_weights


class AlignWindowProbe(FrameProbe):
    def __init__(self, input_kind, head_kind, hidden=128, layers=1):
        if head_kind not in ('mlp','gru'):
            raise ValueError('Align window heads are MLP or causal GRU')
        super().__init__(input_kind, 'mlp', hidden, layers, num_classes=3)
        self.head_kind=head_kind
        if head_kind=='gru':
            width=256 if input_kind=='f6_deform' else 128
            self.context=nn.GRU(width,hidden,layers,batch_first=True,bidirectional=False,
                                dropout=.1 if layers>1 else 0)

    def forward(self, f6=None, deform=None):
        if self.head_kind=='mlp':
            # Projection/normalization are affine, so temporal mean commutes with them.
            values=self.project(f6.mean(1) if f6 is not None else None,
                                deform.mean(1) if deform is not None else None)
            hidden=self.context(values)
        else:
            values=self.project(f6,deform)
            _, state=self.context(values)  # independent windows; no hidden state carried across windows
            hidden=state[-1]
        return self.prediction(hidden)

    def sequence(self, f6=None, deform=None, state=None):
        """GRU step API: only current/past projected features, useful for causality checks."""
        if self.head_kind!='gru':
            raise ValueError('sequence() requires GRU')
        values=self.project(f6,deform); output,state=self.context(values,state)
        return self.prediction(output),state


def metrics(labels, probabilities):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    y=np.asarray(labels,dtype=np.int64); p=np.asarray(probabilities,dtype=np.float64)
    assert p.shape==(len(y),3) and len(y)>0 and np.allclose(p.sum(1),1,atol=1e-5)
    assert np.isfinite(p).all() and set(np.unique(y)).issubset({0,1,2})
    predicted=p.argmax(1); cm=confusion_matrix(y,predicted,labels=[0,1,2])
    precision,recall,f1,support=precision_recall_fscore_support(y,predicted,labels=[0,1,2],zero_division=0)
    return dict(n=len(y),unit='fixed 16-sample window',accuracy=float((predicted==y).mean()),
        balanced_accuracy=float(recall.mean()),macro_recall=float(recall.mean()),macro_f1=float(f1.mean()),
        confusion_matrix=cm.tolist(),confusion_matrix_order=list(CLASSES),
        per_class={c:dict(precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(support[i])) for i,c in enumerate(CLASSES)},
        failure_false_positives=int(cm[:2,2].sum()),
        failure_false_positive_rate=float(cm[:2,2].sum()/cm[:2].sum()) if cm[:2].sum() else None,
        decision='argmax P(in_progress),P(success),P(failure)')


def read_dataset(output, config):
    result={}; directory=output/'datasets'/config['id']
    for split in ('train','val','test'):
        with np.load(directory/(split+'.npz')) as arr:
            result[split]={key:arr[key].copy() for key in arr.files}
        result[split]['metadata']=json.loads((directory/(split+'_metadata.json')).read_text())
        result[split]['labels']=labels_for(result[split]['frames'],result[split]['keys'],result[split]['outcomes'],config['membership'])
    return result


def normalization(features, indices):
    rows=np.unique(indices)
    selected=features[rows].astype(np.float64)
    return (torch.from_numpy(selected.mean(0).astype(np.float32)),
            torch.from_numpy(np.maximum(selected.std(0),.01).astype(np.float32)))


def batch_features(tables, indices, kind):
    return {key:table[indices] if key in kind else None for key,table in tables.items()}


@torch.inference_mode()
def evaluate(model, data, tables, device, batch_size=256, labels=None):
    model.eval(); probabilities=[]
    for start in range(0,len(data['indices']),batch_size):
        index=torch.as_tensor(data['indices'][start:start+batch_size],device=device)
        features=batch_features(tables,index,model.input_kind)
        probabilities.append(model(**features).softmax(-1).cpu().numpy())
    p=np.concatenate(probabilities); y=data['labels'] if labels is None else labels
    return metrics(y,p),p


def save_predictions(path, dataset, probabilities):
    records=[]
    for row,label,p in zip(dataset['metadata'],dataset['labels'],probabilities):
        record={k:row[k] for k in ('event_id','rollout_id','event_key','event_index','key','step','window_start_frame','window_end_frame','padded_frames')}
        record.update(label=int(label),prediction=int(p.argmax()),sample_frames=json.dumps(row['sample_frames']),sample_ticks=json.dumps(row['sample_ticks']))
        record.update({c+'_probability':float(p[i]) for i,c in enumerate(CLASSES)}); records.append(record)
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)


def train_group(args, config, kind, head, seed, dataset, tables, norms, manifest):
    directory=args.output/'runs'/config['id']/f'seed_{seed}'/(kind+'_'+head); directory.mkdir(parents=True,exist_ok=True)
    result_path=directory/'metrics.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        assert result['config']==config and result['seed']==seed
        assert (directory/'best.pt').exists() and (directory/'test_predictions.csv').exists()
        return result
    started=time.monotonic(); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model_config=dict(input_kind=kind,head_kind=head,hidden=128,layers=1)
    model=AlignWindowProbe(**model_config)
    for modality,(mean,std) in norms.items():
        getattr(model,modality+'_mean').copy_(mean); getattr(model,modality+'_std').copy_(std)
    model.to(args.device)
    counts=expanded_counts(dataset['train']['metadata'],config)
    weights=class_weights(counts,config['weight_mode'])
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=args.device))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed); history=[]; best_score=-1; best_epoch=0
    keys=dataset['train']['keys']; outcomes=dataset['train']['outcomes']; frames=dataset['train']['frames']
    event_ids=[r['event_id'] for r in dataset['train']['metadata']]
    unique_events=sorted(set(event_ids)); event_numbers=np.array([unique_events.index(i) for i in event_ids])
    offsets=np.arange(-config['jitter'],config['jitter']+1)
    permutations=np.stack([rng.permutation(offsets) for _ in unique_events])
    # Every integer Key shift is visited for every event before early stopping can fire.
    min_epochs=len(offsets)
    for epoch in range(1,args.epochs+1):
        model.train(); shift=permutations[:,(epoch-1)%len(offsets)][event_numbers]
        labels=labels_for(frames,keys+shift,outcomes,config['membership'])
        order=rng.permutation(len(labels)); loss_sum=0.; denominator=0.
        for start in range(0,len(order),args.batch_size):
            chosen=order[start:start+args.batch_size]
            index=torch.as_tensor(dataset['train']['indices'][chosen],device=args.device)
            features=batch_features(tables,index,kind); target=torch.as_tensor(labels[chosen],device=args.device)
            optimizer.zero_grad(set_to_none=True); logits=model(**features)
            assert logits.shape==(len(chosen),3)
            loss=loss_fn(logits,target); assert torch.isfinite(loss)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
            weight=float(loss_fn.weight[target].sum()); loss_sum+=float(loss.detach())*weight; denominator+=weight
        validation,_=evaluate(model,dataset['val'],tables,args.device,args.batch_size*2)
        score=validation['balanced_accuracy']; history.append(dict(epoch=epoch,train_loss=loss_sum/denominator,
            val_balanced_accuracy=score,val_macro_f1=validation['macro_f1'],epoch_class_counts=np.bincount(labels,minlength=3).tolist(),
            jitter_cycle_index=(epoch-1)%len(offsets),elapsed_seconds=time.monotonic()-started))
        if epoch>=min_epochs and score>best_score+1e-8:
            best_score=score; best_epoch=epoch
            torch.save(dict(model_class='AlignWindowProbe',model_config=model_config,
                state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                output_unit='align_window',class_mapping=dict(enumerate(CLASSES)),window=WINDOW,
                config=config,seed=seed,best_epoch=epoch,encoder_sha256=manifest['signature']['encoder_sha256'],
                split_sha256=manifest['signature']['split_sha256'],training_class_counts=counts.tolist(),
                training_class_weights=weights.tolist(),selection='validation 3-class balanced accuracy after full Key-offset cycle; unshifted Key'),directory/'best.pt')
        if epoch>=min_epochs and epoch-best_epoch>=args.patience:
            break
    blob=torch.load(directory/'best.pt',weights_only=True,map_location=args.device); model.load_state_dict(blob['state_dict'])
    val,vp=evaluate(model,dataset['val'],tables,args.device,args.batch_size*2)
    test,tp=evaluate(model,dataset['test'],tables,args.device,args.batch_size*2)
    result=dict(config=config,name=kind+'_'+head,seed=seed,model_config=model_config,best_epoch=best_epoch,
        epochs_run=len(history),duration_seconds=time.monotonic()-started,
        trainable_parameters=sum(p.numel() for p in model.parameters()),training_class_counts=counts.tolist(),
        class_weight_unit='all training-window Key variants, not frames or intervals',training_class_weights=weights.tolist(),
        train_base_windows=len(keys),jitter_offsets=offsets.tolist(),validation=val,test=test)
    dump(directory/'history.json',history); save_predictions(directory/'validation_predictions.csv',dataset['val'],vp)
    save_predictions(directory/'test_predictions.csv',dataset['test'],tp)
    # Same held-out inputs, shifted hard GT: report annotation tolerance separately; never tune on this.
    stress={}
    for delta in (-10,-5,-2,0,2,5,10):
        stress[str(delta)]=metrics(labels_for(dataset['test']['frames'],dataset['test']['keys']+delta,
                                   dataset['test']['outcomes'],config['membership']),tp)
    dump(directory/'key_shift_stress.json',stress); dump(result_path,result)
    del model, optimizer; torch.cuda.empty_cache()
    print('ALIGN_RUN_COMPLETE',config['id'],seed,kind,head,'BA',round(test['balanced_accuracy'],4),'seconds',round(result['duration_seconds'],1),flush=True)
    return result


def run(args):
    args.output=args.output.resolve(); torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False; torch.set_float32_matmul_precision('highest')
    manifest,events,f6,deform=load_events(args.output)
    local=[]
    for event in events:
        with np.load(args.output/event['feature_path']) as arr: local.append(arr['f6_local'].copy())
    f6_local=np.concatenate(local)
    grid=json.loads((args.output/'dataset_grid.json').read_text())
    if args.only_config: grid=[g for g in grid if g['config']['id']==args.only_config]
    assert grid
    dump(args.output/'suite_config.json',dict(epochs=args.epochs,patience=args.patience,batch_size=args.batch_size,
        optimizer='AdamW lr=.001 weight_decay=.0001 clip_grad_norm=1',hidden=128,layers=1,
        device=args.device,screen_seed=42,repeat_seeds=[42,43,44,45,46],repeat_top_configs=args.repeat_top,
        selection='mean validation BA over six groups at seed42; top primary configs only; test never used; checkpoint epoch >= 2n+1',
        frozen_encoders=True,decoder_used=False,key_jitter='hard labels; event-wise cyclic permutation of all integer offsets; minimum epochs 2n+1',
        evaluation='original Key; val/test never jittered; no training windows shared across rollout splits',
        normalization='unique training feature rows actually referenced by windows; std floor .01',
        created_at=datetime.datetime.now().astimezone().isoformat()))
    results=[]; gpu_deform=torch.from_numpy(deform).to(args.device)
    for item in grid:
        if not item['eligible']:
            print('INELIGIBLE_GRID_CELL',item['config']['id'],item['reason'],flush=True); continue
        config=item['config']; dataset=read_dataset(args.output,config)
        selected_f6=f6_local if config['context']=='align_local' else f6
        tables=dict(f6=torch.from_numpy(selected_f6).to(args.device),deform=gpu_deform)
        norms={k:normalization(v,dataset['train']['indices']) for k,v in [('f6',selected_f6),('deform',deform)]}
        for kind in INPUTS:
            for head in ('mlp','gru'):
                results.append(train_group(args,config,kind,head,42,dataset,tables,norms,manifest))
                dump(args.output/'results.json',results)
        del tables; torch.cuda.empty_cache()
    ranking=[]
    for item in grid:
        c=item['config']; chosen=[r for r in results if r['config']['id']==c['id']]
        if chosen and c['phase']=='primary':
            ranking.append(dict(config=c,validation_mean_balanced_accuracy=float(np.mean([r['validation']['balanced_accuracy'] for r in chosen]))))
    ranking.sort(key=lambda r:(-r['validation_mean_balanced_accuracy'],r['config']['id']))
    dump(args.output/'validation_ranking.json',ranking)
    for row in ranking[:args.repeat_top]:
        config=row['config']; dataset=read_dataset(args.output,config)
        selected_f6=f6_local if config['context']=='align_local' else f6
        tables=dict(f6=torch.from_numpy(selected_f6).to(args.device),deform=gpu_deform)
        norms={k:normalization(v,dataset['train']['indices']) for k,v in [('f6',selected_f6),('deform',deform)]}
        for seed in (43,44,45,46):
            for kind in INPUTS:
                for head in ('mlp','gru'):
                    results.append(train_group(args,config,kind,head,seed,dataset,tables,norms,manifest))
                    dump(args.output/'results.json',results)
        del tables; torch.cuda.empty_cache()
    dump(args.output/'suite_status.json',dict(status='training_complete',runs=len(results),grid_cells=len(grid),
        eligible_cells=sum(g['eligible'] for g in grid),repeat_configs=[r['config']['id'] for r in ranking[:args.repeat_top]]))
    print('ALIGN_SUITE_TRAINING_COMPLETE',len(results),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1'); p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--patience',type=int,default=8); p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--repeat-top',type=int,default=3); p.add_argument('--only-config')
    a=p.parse_args()
    if a.epochs < 21: raise ValueError('Need >=21 epochs to cover every offset for n=10')
    run(a)


if __name__=='__main__': main()
