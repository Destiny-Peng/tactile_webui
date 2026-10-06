"""One merged Align per rollout; one F6 encoding and last-frame deform per window."""
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
from .common import ROOT, dump, sha
from .data import episode_arrays
from .models import FrozenEncoders
from .prepare_align_online import candidate_windows, window_labels
from .align_windows import metrics
from .normalize_align_online import normalized_confusions

GROUPS=('symmetric_merged','asymmetric_merged')
CLASSES=('in_progress','success','failure')


def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**data)


def load(path):
    with np.load(path) as data:
        return {k:data[k].copy() for k in data.files}


def regions(row,group,n):
    u=row['merged_align']
    upper=u['end_frame']+n if group=='symmetric_merged' or u['insert_start'] is None else u['insert_start']+n
    return dict(start=max(row['annotation_start'],u['end_frame']-n),end=min(row['annotation_end'],upper),label=u['outcome'])


def label(row,group,n,frames):
    r=regions(row,group,n)
    timeline=np.zeros(row['total_frames'],dtype=np.int64)
    assert r['start']<=r['end']
    timeline[r['start']:r['end']+1]=r['label']
    return window_labels(frames,timeline)


def prepare(args):
    output=args.output; output.mkdir(parents=True,exist_ok=False)
    old=json.loads((args.source/'dataset_manifest.json').read_text())
    original=ROOT/old['source']; dm=json.loads((original/'data_manifest.json').read_text())
    source_manifest={json.loads(line)['id']:json.loads(line) for line in (ROOT/dm['manifest']).read_text().splitlines() if line.strip()}
    encoder=FrozenEncoders().to(args.device)
    assert encoder.vq.cfg.granularity=='finger' and not any(p.requires_grad for p in encoder.parameters())
    hashes={str((args.source/'dataset_manifest.json').relative_to(ROOT)):sha(args.source/'dataset_manifest.json')}
    for path in (encoder.f6_path,encoder.deform_path,ROOT/dm['manifest'],Path(__file__)):
        hashes[str(path.relative_to(ROOT))]=sha(path)
    rows=[]; paths=[]; f6_banks=[]; deform_banks=[]; meta_parts=[]; offset=0
    for ri,original_row in enumerate(old['rollouts']):
        row={k:original_row[k] for k in ('rollout_id','split','annotation_start','annotation_end','total_frames','events')}
        aligns=sorted([e for e in row['events'] if e['event_key'] in (6,8)],key=lambda e:(e['end_frame'],e['event_index']))
        outcome=1 if source_manifest[row['rollout_id']]['ground_truth_outcome']=='success' else 2
        assert (aligns[-1]['event_key']==8)==(outcome==1)
        last=aligns[-1]; following=[e for e in row['events'] if e['event_key'] in (7,9) and e['event_index']>last['event_index']]
        insert=min(following,key=lambda e:e['event_index']) if following else None
        row['merged_align']=dict(start_frame=min(e['start_frame'] for e in aligns),end_frame=max(e['end_frame'] for e in aligns),outcome=outcome,member_indices=[e['event_index'] for e in aligns],insert_start=insert['start_frame'] if insert else None)
        covered=set()
        for e in aligns:covered.update(range(e['start_frame'],e['end_frame']+1))
        row['merged_gap_frames']=row['merged_align']['end_frame']-row['merged_align']['start_frame']+1-len(covered)
        record=dm['records'][row['rollout_id']]
        for k in ('synchronized_frames_path','tactile_events_path'):
            hashes[record[k]]=sha(ROOT/record[k])
        raw=episode_arrays(record,row['events'],dm['signature']['camera'],num_classes=3)
        source_path=args.source/original_row['feature_path'];hashes[str(source_path.relative_to(ROOT))]=sha(source_path)
        cached=load(source_path)
        segment=np.cumsum(np.r_[False,np.diff(cached['ticks'])!=1])
        unique={}; ticks=[]; frames=[]; last_index=[]; sid=[]; step_values=[]; padded=[]
        for step in (3,1,5,8,12):
            bank=candidate_windows(cached,row,step); ids=[]
            for i,indices in enumerate(bank['indices']):
                identity=tuple(indices.tolist())
                if identity not in unique:
                    unique[identity]=len(ticks); ticks.append(cached['ticks'][indices]); frames.append(bank['frames'][i]); last_index.append(indices[-1]);sid.append(segment[indices[-1]])
                    step_values.append(step);padded.append(bank['padded'][i])
                ids.append(unique[identity]+offset)
            for seg in np.unique(segment[bank['indices'][:,-1]]):
                chosen=np.flatnonzero(segment[bank['indices'][:,-1]]==seg)
                paths.append(dict(rollout_index=ri,step=step,segment=int(seg),indices=np.array(ids,dtype=np.int64)[chosen].tolist()))
        ticks=np.asarray(ticks);frames=np.asarray(frames);last_index=np.asarray(last_index)
        assert np.all(raw['valid'][ticks]) and frames.min()>=row['annotation_start'] and frames.max()<=row['annotation_end']
        encoded=[]
        with torch.inference_mode():
            for start in range(0,len(ticks),512):
                value=encoder.f6_features(torch.from_numpy(raw['f6'][ticks[start:start+512]]).to(args.device))
                assert value.shape[1:]==(1280,)
                encoded.append(value.cpu().numpy())
        f6_banks.append(np.concatenate(encoded));deform_banks.append(cached['deform'][last_index])
        meta_parts.append(dict(frames=frames,ticks=ticks,rollout_index=np.full(len(ticks),ri,dtype=np.int64),step=np.asarray(step_values),padded=np.asarray(padded)))
        row['feature_offset']=offset;row['feature_rows']=len(ticks);rows.append(row);offset+=len(ticks)
        print('ENCODE',ri+1,114,row['rollout_id'],len(ticks),'merged',len(aligns),'outcome',outcome,flush=True)
    del encoder
    banks={key:np.concatenate(value) for key,value in [('f6',f6_banks),('deform',deform_banks)]}
    assert all(np.isfinite(v).all() for v in banks.values());save(output/'features.npz',banks)
    meta={key:np.concatenate([part[key] for part in meta_parts]) for key in meta_parts[0]};save(output/'windows.npz',meta)
    dump(output/'paths.json',paths)
    distribution=[];rng=np.random.default_rng(42)
    for group in GROUPS:
        positives={}; reference={s:[] for s in ('train','val','test')}
        for n in (5,10,2,0):
            for step in (3,1,5,8,12):
                for path in paths:
                    if path['step']!=step:continue
                    ri=path['rollout_index'];row=rows[ri];indices=np.asarray(path['indices']);labels=label(row,group,n,meta['frames'][indices])
                    if n==5 and step==3:
                        reference[row['split']].extend((int(i),int(y),n,step) for i,y in zip(indices,labels))
                    if row['split']=='train':
                        for i,y in zip(indices,labels):
                            if y and int(i) not in positives:positives[int(i)]=(int(i),int(y),n,step)
        negatives=[r for r in reference['train'] if r[1]==0 and r[0] not in positives]
        pools=[negatives]+[[r for r in positives.values() if r[1]==c] for c in (1,2)]
        target=min(map(len,pools));assert target>0
        selected=[pool[int(i)] for pool in pools for i in rng.choice(len(pool),target,replace=False)]
        rng.shuffle(selected)
        for split in ('train','val','test'):
            records=selected if split=='train' else reference[split]
            data={k:np.asarray([r[i] for r in records],dtype=np.int64) for i,k in enumerate(('indices','labels','n','step'))}
            assert len(np.unique(data['indices']))==len(data['indices'])
            for i,y,n in zip(data['indices'],data['labels'],data['n']):
                ri=meta['rollout_index'][i]; assert rows[int(ri)]['split']==split
                assert label(rows[int(ri)],group,int(n),meta['frames'][i:i+1])[0]==y
            save(output/'ready'/group/(split+'.npz'),data)
            counts=np.bincount(data['labels'],minlength=3).tolist()
            distribution.append(dict(group=group,split=split,counts=counts,total=len(records)))
        assert counts[2]>0
    assert sum(r['merged_align']['outcome']==1 for r in rows)==92 and sum(r['merged_align']['outcome']==2 for r in rows)==22
    assert all(sha(ROOT/path)==value for path,value in hashes.items())
    split_path=args.source/'split_manifest.json';shutil.copy2(split_path,output/'split_manifest.json')
    dump(output/'dataset_manifest.json',dict(status='complete',created_at=datetime.datetime.now().astimezone().isoformat(),source=str(args.source.relative_to(ROOT)),rollouts=rows,groups=list(GROUPS),class_mapping=list(CLASSES),window=16,posterior_indices=list(range(8,16)),reference_n=5,reference_step=3,ns=[0,2,5,10],steps=[1,3,5,8,12],split_sha256=sha(split_path),input_hashes=hashes,distribution=distribution,encoder_mode='finger',f6_dim=1280,deform_dim=2560,input='one F6 encoding of exact 16 sampled raw points; deform at last point only',training_started=False))
    dump(output/'verification.json',dict(status='PASS',one_merged_align_per_rollout=True,rollout_outcomes={'success':92,'failure':22},fixed_rollout_split=True,labels_checked=True,source_hashes_unchanged=True,encoder_frozen=True))
    snapshot=output/'code_snapshot';snapshot.mkdir();shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    print('DATA_COMPLETE',json.dumps(distribution),flush=True)


class MergedProbe(nn.Module):
    def __init__(self,input_kind,head_kind):
        super().__init__();self.input_kind=input_kind;self.head_kind=head_kind
        for key,dim in [('f6',1280),('deform',2560)]:
            self.register_buffer(key+'_mean',torch.zeros(dim));self.register_buffer(key+'_std',torch.ones(dim))
        self.f6_projection=nn.Linear(1280,128) if 'f6' in input_kind else None
        self.deform_projection=nn.Linear(2560,128) if 'deform' in input_kind else None
        width=256 if input_kind=='f6_deform' else 128
        self.context=nn.GRU(width,128,batch_first=True) if head_kind=='gru' else nn.Sequential(nn.Linear(width,128),nn.ReLU(),nn.Dropout(.1))
        self.prediction=nn.Linear(128,3)
    def forward(self,f6=None,deform=None,state=None):
        values=[]
        for key,feature in [('f6',f6),('deform',deform)]:
            if feature is not None:values.append(getattr(self,key+'_projection')((feature-getattr(self,key+'_mean'))/getattr(self,key+'_std')))
        value=torch.cat(values,-1) if len(values)==2 else values[0]
        if self.head_kind=='gru':value,state=self.context(value,state)
        else:value=self.context(value)
        return self.prediction(value),state


def features(tables,indices,kind):
    return {key:value[indices] if key in kind else None for key,value in tables.items()}


def sequences(paths,ready,rows,split):
    selected={int(i):(int(y),int(step)) for i,y,step in zip(ready['indices'],ready['labels'],ready['step'])}
    result=[]
    for path in paths:
        if rows[path['rollout_index']]['split']!=split:continue
        if split!='train' and path['step']!=3:continue
        ids=np.asarray(path['indices']);target=np.zeros(len(ids),dtype=np.int64);mask=np.zeros(len(ids),dtype=bool)
        for j,i in enumerate(ids):
            if int(i) in selected and selected[int(i)][1]==path['step']:
                target[j]=selected[int(i)][0];mask[j]=True
        if mask.any():result.append(dict(indices=ids,labels=target,mask=mask,rollout_index=path['rollout_index'],step=path['step'],segment=path['segment']))
    assert sum(int(s['mask'].sum()) for s in result)==len(ready['labels'])
    return result


def sequence_batch(paths,tables,kind,device):
    width=max(len(p['indices']) for p in paths)
    ids=np.zeros((len(paths),width),dtype=np.int64);labels=np.zeros_like(ids);mask=np.zeros_like(ids,dtype=bool)
    for j,path in enumerate(paths):
        length=len(path['indices']);ids[j,:length]=path['indices'];labels[j,:length]=path['labels'];mask[j,:length]=path['mask']
    return features(tables,torch.as_tensor(ids,device=device),kind),torch.as_tensor(labels,device=device),torch.as_tensor(mask,device=device)


@torch.inference_mode()
def evaluate(model,ready,paths,tables,device):
    model.eval();probabilities=np.empty((len(ready['indices']),3),dtype=np.float32)
    positions={int(i):j for j,i in enumerate(ready['indices'])}
    if model.head_kind=='mlp':
        for start in range(0,len(ready['indices']),512):
            idx=torch.as_tensor(ready['indices'][start:start+512],device=device)
            logits,_=model(**features(tables,idx,model.input_kind));probabilities[start:start+len(idx)]=logits.softmax(-1).cpu().numpy()
    else:
        for path in paths:
            idx=torch.as_tensor(path['indices'][None],device=device)
            logits,_=model(**features(tables,idx,model.input_kind));p=logits.softmax(-1)[0].cpu().numpy()
            for j,i in enumerate(path['indices']):
                if path['mask'][j]:probabilities[positions[int(i)]]=p[j]
    result=metrics(ready['labels'],probabilities);result['unit']='one encoder window prediction'
    return result,probabilities


def train(args):
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.source/'dataset_manifest.json').read_text());rows=manifest['rollouts']
    paths=json.loads((args.source/'paths.json').read_text());banks=load(args.source/'features.npz')
    hashes={str(p.relative_to(ROOT)):sha(p) for p in (args.source/'dataset_manifest.json',args.source/'features.npz',args.source/'windows.npz',args.source/'paths.json',Path(__file__))}
    assert all(sha(ROOT/p)==v for p,v in manifest['input_hashes'].items())
    tables={k:torch.from_numpy(v).to(args.device) for k,v in banks.items()};results=[];start_time=time.monotonic()
    for group in GROUPS:
        data={split:load(args.source/'ready'/group/(split+'.npz')) for split in ('train','val','test')}
        for split in data:
            p=args.source/'ready'/group/(split+'.npz');hashes[str(p.relative_to(ROOT))]=sha(p)
        seq={split:sequences(paths,data[split],rows,split) for split in data}
        counts=np.bincount(data['train']['labels'],minlength=3);weights=1/counts;weights/=weights.mean()
        assert counts[0]==counts[1]==counts[2]
        norms={}
        for k,value in banks.items():
            x=value[np.unique(data['train']['indices'])].astype(np.float64)
            norms[k]=(torch.from_numpy(x.mean(0).astype(np.float32)),torch.from_numpy(np.maximum(x.std(0),.01).astype(np.float32)))
        for seed in range(42,47):
            for kind in ('f6','deform','f6_deform'):
                for head in ('mlp','gru'):
                    directory=output/'runs'/group/f'seed_{seed}'/(kind+'_'+head);directory.mkdir(parents=True)
                    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
                    config=dict(input_kind=kind,head_kind=head);model=MergedProbe(**config)
                    for k,(mean,std) in norms.items():getattr(model,k+'_mean').copy_(mean);getattr(model,k+'_std').copy_(std)
                    model.to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                    loss_fn=nn.CrossEntropyLoss(weight=torch.as_tensor(weights,dtype=torch.float32,device=args.device))
                    best=-1;best_epoch=0;history=[]
                    for epoch in range(1,31):
                        model.train();loss_sum=0;sample_sum=0
                        order=rng.permutation(len(data['train']['labels']) if head=='mlp' else len(seq['train']))
                        batch_size=128 if head=='mlp' else 8
                        for start in range(0,len(order),batch_size):
                            chosen=order[start:start+batch_size]
                            if head=='mlp':
                                index=torch.as_tensor(data['train']['indices'][chosen],device=args.device)
                                value=features(tables,index,kind);target=torch.as_tensor(data['train']['labels'][chosen],device=args.device);mask=None
                            else:value,target,mask=sequence_batch([seq['train'][int(i)] for i in chosen],tables,kind,args.device)
                            optimizer.zero_grad(set_to_none=True);logits,_=model(**value)
                            if mask is not None:logits=logits[mask];target=target[mask]
                            loss=loss_fn(logits,target);assert torch.isfinite(loss)
                            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                            loss_sum+=float(loss.detach())*len(target);sample_sum+=len(target)
                        assert sample_sum==len(data['train']['labels'])
                        val,_=evaluate(model,data['val'],seq['val'],tables,args.device)
                        history.append(dict(epoch=epoch,train_loss=loss_sum/sample_sum,train_supervised_windows=sample_sum,val_balanced_accuracy=val['balanced_accuracy'],val_macro_f1=val['macro_f1']))
                        if val['balanced_accuracy']>best+1e-8:
                            best=val['balanced_accuracy'];best_epoch=epoch
                            torch.save(dict(model_class='MergedProbe',model_config=config,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},group=group,seed=seed,best_epoch=epoch,class_mapping=dict(enumerate(CLASSES)),output_unit='one_f6_window_last_deform',training_counts=counts.tolist(),class_weights=weights.tolist(),split_sha256=manifest['split_sha256']),directory/'best.pt')
                        if epoch-best_epoch>=8:break
                    blob=torch.load(directory/'best.pt',weights_only=True,map_location=args.device);model.load_state_dict(blob['state_dict'])
                    val,_=evaluate(model,data['val'],seq['val'],tables,args.device)
                    test,p=evaluate(model,data['test'],seq['test'],tables,args.device)
                    result=dict(group=group,input=kind,head=head,seed=seed,best_epoch=best_epoch,epochs=len(history),train_counts=counts.tolist(),class_weights=weights.tolist(),validation=val,test=test)
                    dump(directory/'history.json',history);dump(directory/'metrics.json',result)
                    save(directory/'test_predictions.npz',dict(**data['test'],probabilities=p,predictions=p.argmax(1)))
                    results.append(result);dump(output/'results.json',results)
                    print('TRAIN',len(results),60,group,seed,kind,head,'BA',test['balanced_accuracy'],'F1',test['macro_f1'],'epoch',best_epoch,'elapsed',round(time.monotonic()-start_time),flush=True)
                    del model,optimizer
    assert all(sha(ROOT/p)==v for p,v in hashes.items())
    snapshot=output/'code_snapshot';snapshot.mkdir();shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    dump(output/'run_manifest.json',dict(status='complete',dataset=str(args.source.relative_to(ROOT)),device=args.device,seeds=list(range(42,47)),groups=list(GROUPS),input_hashes=hashes,split_sha256=manifest['split_sha256'],elapsed_seconds=time.monotonic()-start_time,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),environment='repos/ProcVLM/.venv',torch_version=str(torch.__version__),optimizer='AdamW lr=.001 wd=.0001 clip=1',epochs=30,patience=8,mlp_batch_size=128,gru_sequence_batch_size=8,gru_training='full chronological segment; unselected context has no loss; state reset only rollout/tactile gap; no TBPTT',selection='maximum validation BA first tie',encoder_frozen=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','train'));p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    if args.command=='prepare':prepare(args)
    elif args.command=='train':train(args)

if __name__=='__main__':main()
