"""Diagnostic binary classification of each merged Align interval."""
import argparse
import csv
import datetime
import json
import random
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import ROOT,dump,sha
from .data import episode_arrays
from .models import FrozenEncoders
from .merged_online import MergedProbe,load,save
from .prepare_align_online import candidate_windows
from .train_intervals import metrics,batch_data,normalization


class MergedIntervalProbe(MergedProbe):
    def __init__(self,input_kind,head_kind):
        super().__init__(input_kind,head_kind);self.prediction=nn.Linear(128,2)
    def forward(self,f6=None,deform=None,lengths=None):
        values=[]
        for key,feature in [('f6',f6),('deform',deform)]:
            if feature is not None and key in self.input_kind:
                values.append(getattr(self,key+'_projection')((feature-getattr(self,key+'_mean'))/getattr(self,key+'_std')))
        value=torch.cat(values,-1) if len(values)==2 else values[0]
        if self.head_kind=='mlp':
            mask=torch.arange(value.shape[1],device=value.device)[None,:]<lengths.to(value.device)[:,None]
            pooled=(value*mask[:,:,None]).sum(1)/lengths.to(value.device)[:,None]
            hidden=self.context(pooled)
        else:
            packed=nn.utils.rnn.pack_padded_sequence(value,lengths.cpu(),batch_first=True,enforce_sorted=False)
            _,state=self.context(packed);hidden=state[-1]
        return self.prediction(hidden)


def prepare(args):
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    source=args.source;dm=json.loads((source/'dataset_manifest.json').read_text())
    old=ROOT/dm['source'];oldm=json.loads((old/'dataset_manifest.json').read_text());original=ROOT/oldm['source']
    originalm=json.loads((original/'data_manifest.json').read_text())
    banks=load(source/'features.npz');meta=load(source/'windows.npz')
    encoder=FrozenEncoders().to(args.device);assert not any(p.requires_grad for p in encoder.parameters())
    hashes={str(p.relative_to(ROOT)):sha(p) for p in (source/'dataset_manifest.json',source/'features.npz',source/'windows.npz',encoder.f6_path,encoder.deform_path,Path(__file__))}
    assert all(sha(ROOT/p)==v for p,v in dm['input_hashes'].items())
    rows=[];encoded_count=0;reused_count=0
    for ri,row in enumerate(dm['rollouts']):
        unit=row['merged_align'];start=unit['start_frame'];end=unit['end_frame']
        cached_path=old/oldm['rollouts'][ri]['feature_path'];hashes[str(cached_path.relative_to(ROOT))]=sha(cached_path)
        cached=load(cached_path);selected=(cached['video_frames']>=start)&(cached['video_frames']<=end)
        cached={k:v[selected] for k,v in cached.items()};assert len(cached['ticks'])
        boundary=dict(row,annotation_start=start,annotation_end=end)
        windows=candidate_windows(cached,boundary,step=3)
        ticks=cached['ticks'][windows['indices']];frames=windows['frames']
        assert frames.min()>=start and frames.max()<=end and (np.diff(ticks,axis=1)>=0).all()
        lookup={tuple(meta['ticks'][i].tolist()):int(i) for i in np.flatnonzero(meta['rollout_index']==ri)}
        f6=np.empty((len(ticks),1280),dtype=np.float32);missing=[]
        for i,window in enumerate(ticks):
            key=tuple(window.tolist())
            if key in lookup:f6[i]=banks['f6'][lookup[key]];reused_count+=1
            else:missing.append(i)
        if missing:
            raw=episode_arrays(originalm['records'][row['rollout_id']],row['events'],originalm['signature']['camera'],num_classes=3)
            assert np.all(raw['valid'][ticks])
            with torch.inference_mode():
                for pos in range(0,len(missing),256):
                    chosen=np.asarray(missing[pos:pos+256]);value=torch.from_numpy(raw['f6'][ticks[chosen]]).to(args.device)
                    f6[chosen]=encoder.f6_features(value).cpu().numpy()
            encoded_count+=len(missing)
        deform=cached['deform'][windows['indices'][:,-1]]
        assert np.isfinite(f6).all() and np.isfinite(deform).all()
        path=output/'features'/(row['rollout_id']+'.npz');binary_label=int(unit['outcome']==2)
        save(path,dict(f6=f6,deform=deform,frames=frames,ticks=ticks,padded=windows['padded'],label=np.array(binary_label)))
        rows.append(dict(interval_id=row['rollout_id']+'__merged_align',rollout_id=row['rollout_id'],split=row['split'],start_frame=start,end_frame=end,label=binary_label,member_indices=unit['member_indices'],feature_path=str(path.relative_to(output)),windows=len(f6),step=3))
        print('INTERVAL_FEATURES',ri+1,114,row['rollout_id'],'windows',len(f6),'prefix_encoded',len(missing),flush=True)
    del encoder
    shutil.copy2(source/'split_manifest.json',output/'split_manifest.json')
    counts={split:np.bincount([r['label'] for r in rows if r['split']==split],minlength=2).tolist() for split in ('train','val','test')}
    assert counts=={'train':[64,16],'val':[14,3],'test':[14,3]}
    signature=json.dumps([(r['rollout_id'],r['start_frame'],r['end_frame'],r['label']) for r in rows],sort_keys=True)
    import hashlib
    corpus_sha=hashlib.sha256(signature.encode()).hexdigest()
    dump(output/'group_equivalence.json',dict(groups=dm['groups'],interval_corpus_sha256={g:corpus_sha for g in dm['groups']},identical=True,reason='Both groups share merged intervals; only Key bands differ, which binary interval supervision does not use.',unique_training_runs=30))
    dump(output/'interval_manifest.json',dict(status='complete',created_at=datetime.datetime.now().astimezone().isoformat(),source=str(source.relative_to(ROOT)),records=rows,class_mapping={'0':'success','1':'failure'},counts=counts,split_sha256=sha(source/'split_manifest.json'),input_hashes=hashes,source_groups=dm['groups'],groups_identical=True,feature_context='one exact16 raw F6 window encoded once + last-frame deform; interval-only edge padding; step3',reencoded_windows=encoded_count,reused_windows=reused_count,encoder_frozen=True))
    assert all(sha(ROOT/p)==v for p,v in hashes.items())
    snapshot=output/'code_snapshot';snapshot.mkdir();shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    print('INTERVAL_DATA_COMPLETE',counts,'reused',reused_count,'reencoded',encoded_count,flush=True)


def read_samples(source,manifest):
    samples={split:[] for split in ('train','val','test')}
    for row in manifest['records']:
        data=load(source/row['feature_path']);assert int(data['label'])==row['label'] and len(data['f6'])==row['windows']
        samples[row['split']].append(dict(row,**{k:data[k] for k in ('f6','deform')}))
    return samples


@torch.inference_mode()
def evaluate(model,samples,device):
    model.eval();probabilities=[]
    for start in range(0,len(samples),8):
        batch=batch_data(samples[start:start+8],device)
        p=model(batch['f6'],batch['deform'],batch['lengths']).softmax(-1).cpu().numpy();probabilities.append(p)
    p=np.concatenate(probabilities);assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
    return metrics([r['label'] for r in samples],p[:,1]),p


def train(args):
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    dm=json.loads((args.source/'interval_manifest.json').read_text());samples=read_samples(args.source,dm)
    assert all(sha(ROOT/p)==v for p,v in dm['input_hashes'].items())
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [args.source/'interval_manifest.json',args.source/'group_equivalence.json',args.source/'split_manifest.json',Path(__file__)]+[args.source/r['feature_path'] for r in dm['records']]}
    norm=normalization(samples['train']);counts=np.bincount([r['label'] for r in samples['train']],minlength=2);weights=len(samples['train'])/(2*counts)
    results=[];started=time.monotonic()
    for seed in range(42,47):
        for kind in ('f6','deform','f6_deform'):
            for head in ('mlp','gru'):
                torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
                directory=output/'runs'/f'seed_{seed}'/(kind+'_'+head);directory.mkdir(parents=True)
                config=dict(input_kind=kind,head_kind=head);model=MergedIntervalProbe(**config)
                for k,v in norm.items():getattr(model,k).copy_(v)
                model.to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                criterion=nn.CrossEntropyLoss(weight=torch.as_tensor(weights,dtype=torch.float32,device=args.device))
                best=-1;best_epoch=0;history=[]
                for epoch in range(1,31):
                    model.train();order=rng.permutation(len(samples['train']));loss_sum=0;weight_sum=0
                    for pos in range(0,len(order),8):
                        batch=batch_data([samples['train'][int(i)] for i in order[pos:pos+8]],args.device)
                        optimizer.zero_grad(set_to_none=True);logits=model(batch['f6'],batch['deform'],batch['lengths'])
                        assert logits.shape==(len(batch['labels']),2)
                        loss=criterion(logits,batch['labels']);assert torch.isfinite(loss)
                        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                        denom=float(criterion.weight[batch['labels']].sum());loss_sum+=float(loss.detach())*denom;weight_sum+=denom
                    val,_=evaluate(model,samples['val'],args.device)
                    history.append(dict(epoch=epoch,train_loss=loss_sum/weight_sum,val_balanced_accuracy=val['balanced_accuracy'],val_macro_f1=val['macro_f1']))
                    if val['balanced_accuracy']>best+1e-8:
                        best=val['balanced_accuracy'];best_epoch=epoch
                        torch.save(dict(model_class='MergedIntervalProbe',model_config=config,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},seed=seed,best_epoch=epoch,class_mapping={'0':'success','1':'failure'},training_interval_counts=counts.tolist(),class_weights=weights.tolist(),split_sha256=dm['split_sha256'],output_unit='merged_align_interval',threshold=.5),directory/'best.pt')
                    if epoch-best_epoch>=8:break
                checkpoint=torch.load(directory/'best.pt',map_location=args.device,weights_only=True);model.load_state_dict(checkpoint['state_dict'])
                val,_=evaluate(model,samples['val'],args.device);test,p=evaluate(model,samples['test'],args.device)
                result=dict(input=kind,head=head,seed=seed,best_epoch=best_epoch,epochs=len(history),train_counts=counts.tolist(),class_weights=weights.tolist(),validation=val,test=test,source_groups=dm['source_groups'])
                dump(directory/'history.json',history);dump(directory/'metrics.json',result);save(directory/'test_predictions.npz',dict(labels=np.array([r['label'] for r in samples['test']]),probabilities=p,predictions=(p[:,1]>=.5).astype(int)))
                records=[]
                for row,prob in zip(samples['test'],p):
                    records.append(dict(rollout_id=row['rollout_id'],interval_id=row['interval_id'],start_frame=row['start_frame'],end_frame=row['end_frame'],windows=row['windows'],label=row['label'],prediction=int(prob[1]>=.5),p_success=float(prob[0]),p_failure=float(prob[1])))
                with (directory/'test_predictions.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
                results.append(result);dump(output/'results.json',results)
                print('INTERVAL_TRAIN',len(results),30,seed,kind,head,'BA',test['balanced_accuracy'],'F1',test['macro_f1'],'epoch',best_epoch,'elapsed',round(time.monotonic()-started),flush=True)
                del model,optimizer
    assert all(sha(ROOT/p)==v for p,v in hashes.items())
    snapshot=output/'code_snapshot';snapshot.mkdir();shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    dump(output/'run_manifest.json',dict(status='complete',dataset=str(args.source.relative_to(ROOT)),source_groups=dm['source_groups'],groups_identical=True,unique_runs=30,seeds=list(range(42,47)),device=args.device,environment='repos/ProcVLM/.venv',torch_version=str(torch.__version__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_hashes=hashes,elapsed_seconds=time.monotonic()-started,loss='weighted CE; weights=N/(2*n_class_intervals)',selection='maximum validation interval BA, first tie, threshold .5',epochs=30,patience=8,batch_size=8,lr=.001,weight_decay=.0001,gradient_clip=1,hidden=128,gru_layers=1,encoder_frozen=True,split_sha256=dm['split_sha256']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','train'));parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    if args.command=='prepare':prepare(args)
    else:train(args)

if __name__=='__main__':main()
