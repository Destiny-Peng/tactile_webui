"""Align-only one-sided Gaussian supervision with fixed last-timestep hard GT."""
import argparse
import datetime
import json
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import ROOT,dump,sha
from .data import episode_arrays
from .models import FrozenEncoders
from .merged_online import MergedProbe,load,save
from .train_intervals import normalization
from .align_windows import metrics
from .gaussian_targets import SIGMAS,CLASSES,one_sided_targets,fixed_reference

GROUPS=('merged_align','original_align')


def prepare(args):
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    source=args.source;origin=json.loads((source/'dataset_manifest.json').read_text())
    dense=ROOT/origin['source'];old=json.loads((dense/'dataset_manifest.json').read_text())
    upstream=ROOT/old['source'];dm=json.loads((upstream/'data_manifest.json').read_text())
    encoder=FrozenEncoders().to(args.device)
    assert encoder.vq.cfg.granularity=='finger' and not any(p.requires_grad for p in encoder.parameters())
    hashes={str(p.relative_to(ROOT)):sha(p) for p in (source/'dataset_manifest.json',dense/'dataset_manifest.json',upstream/'data_manifest.json',encoder.f6_path,encoder.deform_path,Path(__file__),Path(__file__).with_name('gaussian_targets.py'))}
    rows=[];first_prefix=None
    for ri,row in enumerate(origin['rollouts']):
        record=dm['records'][row['rollout_id']]
        for key,hashkey in [('synchronized_frames_path','frames_sha256'),('tactile_events_path','events_sha256')]:
            p=ROOT/record[key];value=sha(p);assert value==dm['signature']['episode_sources'][row['rollout_id']][hashkey];hashes[record[key]]=value
        raw=episode_arrays(record,row['events'],dm['signature']['camera'],num_classes=3)
        path=dense/'features'/(row['rollout_id']+'.npz');hashes[str(path.relative_to(ROOT))]=sha(path);cached=load(path)
        aligns=sorted([e for e in row['events'] if e['event_key'] in (6,8)],key=lambda e:(e['start_frame'],e['event_index']))
        units=[dict(group='merged_align',start=row['merged_align']['start_frame'],key=row['merged_align']['end_frame'],end=row['annotation_end'],outcome=row['merged_align']['outcome'],member_indices=row['merged_align']['member_indices'])]
        for j,e in enumerate(aligns):
            end=aligns[j+1]['start_frame']-1 if j+1<len(aligns) else row['annotation_end']
            assert end>=e['end_frame']
            units.append(dict(group='original_align',start=e['start_frame'],key=e['end_frame'],end=end,outcome=1 if e['event_key']==8 else 2,member_indices=[e['event_index']]))
        for u in units:
            mask=(cached['video_frames']>=u['start'])&(cached['video_frames']<=u['end']);indices=np.flatnonzero(mask)
            assert len(indices)>0
            data={k:v[indices].copy() for k,v in cached.items()};ticks=data['ticks'];frames=data['video_frames'];first=int(ticks[0])
            prefix=np.flatnonzero(ticks<first+15)
            if len(prefix):
                windows=np.maximum(ticks[prefix,None]+np.arange(-15,1)[None],first)
                assert raw['valid'][windows].all() and np.all(windows<=ticks[prefix,None])
                with torch.inference_mode():
                    enc=encoder.f6_features(torch.from_numpy(raw['f6'][windows]).to(args.device)).cpu().numpy()
                assert enc.shape==(len(prefix),1280);data['f6'][prefix]=enc
                if first_prefix is None:first_prefix=dict(raw=raw['f6'][windows[:1]],features=enc[:1])
            key_index=int(np.searchsorted(frames,u['key'],side='left'));assert 0<key_index<len(frames)
            positions=np.arange(len(frames));labels=fixed_reference(positions,key_index,u['outcome'])
            assert np.array_equal(labels,np.where(frames<u['key'],0,u['outcome']))
            for sigma in SIGMAS:
                target=one_sided_targets(positions,key_index,u['outcome'],sigma)
                np.testing.assert_allclose(target.sum(1),1,atol=1e-6)
                assert np.all(np.diff(target[:,u['outcome']])>=0) and np.all(target[key_index:,u['outcome']]==1) and np.all(target[:,3-u['outcome']]==0)
            identity=f"{u['group']}_{len(rows):04d}";relative='features/'+identity+'.npz'
            save(output/relative,dict(**data,labels=labels,positions=positions))
            rows.append(dict(u,interaction_id=identity,rollout_id=row['rollout_id'],split=row['split'],feature_path=relative,ticks=len(frames),key_index=key_index,post_key_ticks=len(frames)-key_index,prefix_reencoded=len(prefix),gaps=int((np.diff(ticks)!=1).sum())))
        print('PREPARE',ri+1,len(origin['rollouts']),flush=True)
    del encoder
    counts=[]
    for group in GROUPS:
        for split in ('train','val','test'):
            selected=[r for r in rows if r['group']==group and r['split']==split]
            y=np.concatenate([load(output/r['feature_path'])['labels'] for r in selected])
            counts.append(dict(group=group,split=split,interactions=len(selected),success=sum(r['outcome']==1 for r in selected),failure=sum(r['outcome']==2 for r in selected),class_ticks=np.bincount(y,minlength=3).tolist(),total_ticks=len(y)))
    assert sum(r['group']=='merged_align' for r in rows)==114 and sum(r['group']=='original_align' for r in rows)==163
    shutil.copy2(source/'split_manifest.json',output/'split_manifest.json')
    assert all(sha(ROOT/p)==v for p,v in hashes.items())
    save(output/'prefix_encoding_check.npz',first_prefix)
    dump(output/'dataset_manifest.json',dict(status='complete',created_at=datetime.datetime.now().astimezone().isoformat(),source=str(source.relative_to(ROOT)),groups=list(GROUPS),records=rows,distribution=counts,input_hashes=hashes,split_sha256=sha(output/'split_manifest.json'),sigmas=list(SIGMAS),sigma_unit='valid synchronized prediction ticks; no downsampling',encoder_window=16,f6_dim=1280,deform_dim=2560,encoder_mode='finger',class_order=list(CLASSES),history='single past16 raw F6 encoding plus current deform; GRU state retained within interaction; MLP current encoding only',key='Align end; no fuzzy band; original 6/8 only; merged final Align outcome',tail='until next Align start minus1; last Align and merged until last annotated end',hard_gt='t<Key:0; t>=Key:final outcome; invariant across sigma',soft_target='g=exp(-max(key_index-position,0)^2/(2*sigma^2)); success=[1-g,g,0]; failure=[1-g,0,g]',loss='unweighted soft-target CE over valid ticks; no augmentation',training_started=False))
    dump(output/'verification.json',dict(status='PASS',targets_monotonic_and_normalized=True,hard_gt_sigma_invariant=True,no_insert_targets=True,rollout_split_preserved=True,source_sensor_hashes_verified=True,source_features_unchanged=True,causal_f6_prefix=True,pretrained_frozen=True))
    snapshot=output/'code_snapshot';snapshot.mkdir()
    for name in ('gaussian_online.py','gaussian_targets.py'):shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    print('DATA_COMPLETE',json.dumps(counts),flush=True)


def read_samples(source,manifest,group):
    samples={s:[] for s in ('train','val','test')}
    for row in manifest['records']:
        if row['group']==group:samples[row['split']].append(dict(row,**load(source/row['feature_path'])))
    return samples


def batch(samples,kind,sigma,device):
    lengths=np.asarray([len(s['f6']) for s in samples]);width=int(lengths.max());features={}
    for k,dim in [('f6',1280),('deform',2560)]:
        if k not in kind:continue
        a=np.zeros((len(samples),width,dim),dtype=np.float32)
        for i,s in enumerate(samples):a[i,:lengths[i]]=s[k]
        features[k]=torch.from_numpy(a).to(device)
    target=np.zeros((len(samples),width,3),dtype=np.float32)
    for i,s in enumerate(samples):target[i,:lengths[i]]=one_sided_targets(s['positions'],s['key_index'],s['outcome'],sigma)
    mask=torch.arange(width,device=device)[None]<torch.as_tensor(lengths,device=device)[:,None]
    return features,torch.from_numpy(target).to(device),mask,lengths


@torch.inference_mode()
def evaluate(model,samples,sigma,device):
    model.eval();ps=[];ys=[];soft_loss=0;total=0
    for start in range(0,len(samples),8):
        chosen=samples[start:start+8];x,q,mask,lengths=batch(chosen,model.input_kind,sigma,device)
        logits,_=model(**x);p=logits.softmax(-1)
        soft_loss+=float(-(q*logits.log_softmax(-1)).sum(-1)[mask].sum());total+=int(mask.sum())
        for i,s in enumerate(chosen):ps.append(p[i,:lengths[i]].cpu().numpy());ys.append(s['labels'])
    p=np.concatenate(ps);y=np.concatenate(ys);m=metrics(y,p);m['unit']='valid last-timestep within Align interaction and retained annotated tail';m['soft_ce']=soft_loss/total
    return m,p


def train_run(output,group,kind,head,seed,sigma,samples,norm,manifest,device):
    directory=output/'runs'/group/(kind+'_'+head)/f'sigma_{sigma}'/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    config=dict(input_kind=kind,head_kind=head);model=MergedProbe(**config)
    for k,v in norm.items():getattr(model,k).copy_(v)
    model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    history=[];best=-1;best_epoch=0
    for epoch in range(1,31):
        model.train();order=rng.permutation(len(samples['train']));loss_sum=0;total=0
        for start in range(0,len(order),8):
            chosen=[samples['train'][int(i)] for i in order[start:start+8]];x,q,mask,_=batch(chosen,kind,sigma,device)
            optimizer.zero_grad(set_to_none=True);logits,_=model(**x)
            element=-(q*logits.log_softmax(-1)).sum(-1);loss=element[mask].mean();assert torch.isfinite(loss)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            loss_sum+=float(element[mask].detach().sum());total+=int(mask.sum())
        val,_=evaluate(model,samples['val'],sigma,device);score=val['balanced_accuracy']
        history.append(dict(epoch=epoch,train_soft_ce=loss_sum/total,val_balanced_accuracy=score,val_macro_f1=val['macro_f1']))
        if score>best+1e-8:
            best=score;best_epoch=epoch
            torch.save(dict(model_config=config,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},sigma=sigma,seed=seed,group=group,best_epoch=epoch,split_sha256=manifest['split_sha256'],class_order=list(CLASSES),model_class='MergedProbe',loss='unweighted soft-target CE',state_reset='new interaction only'),directory/'best.pt')
        if epoch-best_epoch>=8:break
    blob=torch.load(directory/'best.pt',map_location=device,weights_only=True);model.load_state_dict(blob['state_dict'])
    val,_=evaluate(model,samples['val'],sigma,device);test,p=evaluate(model,samples['test'],sigma,device)
    y=np.concatenate([s['labels'] for s in samples['test']]);offsets=np.r_[0,np.cumsum([len(s['labels']) for s in samples['test']])]
    save(directory/'test_predictions.npz',dict(labels=y,probabilities=p,offsets=offsets))
    result=dict(group=group,input=kind,head=head,seed=seed,sigma=sigma,best_epoch=best_epoch,epochs=len(history),validation=val,test=test,directory=str(directory.relative_to(ROOT)))
    dump(directory/'metrics.json',result);dump(directory/'history.json',history)
    del model,optimizer;torch.cuda.empty_cache()
    print('RUN_COMPLETE',group,kind,head,sigma,seed,'valBA',round(best,4),'testBA',round(test['balanced_accuracy'],4),flush=True)
    return result


def train(args):
    output=args.output;output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    manifest=json.loads((args.source/'dataset_manifest.json').read_text())
    assert all(sha(ROOT/p)==v for p,v in manifest['input_hashes'].items())
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [args.source/'dataset_manifest.json',args.source/'split_manifest.json',Path(__file__),Path(__file__).with_name('gaussian_targets.py'),Path(__file__).with_name('merged_online.py')]+[args.source/r['feature_path'] for r in manifest['records']]}
    dump(output/'protocol.json',dict(source=str(args.source.relative_to(ROOT)),groups=list(GROUPS),sigmas=list(SIGMAS),screen_seed=42,repeat_seeds=[43,44,45,46],selection='per group/input/head: maximize seed42 validation fixed-hard-GT BA; tie smaller sigma; no test-based selection',epochs=30,patience=8,batch_interactions=8,optimizer='AdamW lr=.001 wd=.0001 clip1',loss='unweighted soft-target CE, valid tick mean per batch; no augmentation',input_hashes=hashes))
    results=[];selections=[]
    for group in GROUPS:
        samples=read_samples(args.source,manifest,group);norm=normalization(samples['train'])
        for kind in ('f6','deform','f6_deform'):
            for head in ('mlp','gru'):
                screen=[]
                for sigma in SIGMAS:
                    result=train_run(output,group,kind,head,42,sigma,samples,norm,manifest,args.device);results.append(result);screen.append(result);dump(output/'results_partial.json',results)
                winner=max(screen,key=lambda r:(r['validation']['balanced_accuracy'],-r['sigma']))
                selections.append(dict(group=group,input=kind,head=head,sigma=winner['sigma'],seed42_val_balanced_accuracy=winner['validation']['balanced_accuracy']))
                dump(output/'sigma_selection.json',selections)
                for seed in range(43,47):
                    results.append(train_run(output,group,kind,head,seed,winner['sigma'],samples,norm,manifest,args.device));dump(output/'results_partial.json',results)
    assert len(results)==120 and all(sha(ROOT/p)==v for p,v in hashes.items())
    dump(output/'results.json',results);dump(output/'training_manifest.json',dict(status='complete',runs=len(results),elapsed_seconds=time.monotonic()-started,source=str(args.source.relative_to(ROOT)),input_hashes=hashes))
    snapshot=output/'code_snapshot';snapshot.mkdir()
    for name in ('gaussian_online.py','gaussian_targets.py','merged_online.py'):shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    print('TRAIN_COMPLETE',len(results),time.monotonic()-started,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','train']);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();args.source=args.source.resolve();args.output=args.output.resolve();torch.set_num_threads(4)
    (prepare if args.action=='prepare' else train)(args)

if __name__=='__main__':main()
