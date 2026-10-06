"""Original event intervals, sixteen equal-weight hindsight outcome endpoints."""
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
from .merged_online import load,save
from .train_intervals import metrics,normalization,batch_data

POSITIONS=16


def endpoint_indices(length):
    assert length>0
    return (np.ceil(np.arange(1,POSITIONS+1)*length/POSITIONS)-1).astype(np.int64)


class PrefixProbe(nn.Module):
    def __init__(self,input_kind,head_kind):
        super().__init__();self.input_kind=input_kind;self.head_kind=head_kind
        for k,d in [('f6',1280),('deform',2560)]:
            self.register_buffer(k+'_mean',torch.zeros(d));self.register_buffer(k+'_std',torch.ones(d))
        self.f6_projection=nn.Linear(1280,128) if 'f6' in input_kind else None
        self.deform_projection=nn.Linear(2560,128) if 'deform' in input_kind else None
        width=256 if input_kind=='f6_deform' else 128
        self.context=nn.GRU(width,128,batch_first=True) if head_kind=='gru' else nn.Sequential(nn.Linear(width,128),nn.ReLU(),nn.Dropout(.1))
        self.prediction=nn.Linear(128,1)
    def project(self,f6=None,deform=None):
        values=[]
        for k,x in [('f6',f6),('deform',deform)]:
            if x is not None and k in self.input_kind:values.append(getattr(self,k+'_projection')((x-getattr(self,k+'_mean'))/getattr(self,k+'_std')))
        return torch.cat(values,-1) if len(values)==2 else values[0]
    def forward(self,f6=None,deform=None,state=None):
        value=self.project(f6,deform)
        if self.head_kind=='gru':hidden,state=self.context(value,state)
        else:
            previous_sum,count=(torch.zeros_like(value[:,0]),0) if state is None else state
            cumulative=value.cumsum(1)+previous_sum[:,None]
            divisors=torch.arange(count+1,count+value.shape[1]+1,device=value.device,dtype=value.dtype)
            hidden=self.context(cumulative/divisors[None,:,None]);state=(cumulative[:,-1],count+value.shape[1])
        return self.prediction(hidden).squeeze(-1),state


def prefix_loss(selected_logits,labels,pos_weight):
    assert selected_logits.ndim==2 and selected_logits.shape[1]==POSITIONS
    target=labels.float()[:,None].expand_as(selected_logits)
    element=nn.functional.binary_cross_entropy_with_logits(selected_logits,target,pos_weight=pos_weight,reduction='none')
    return element.mean(1).mean()


def prepare(args):
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    source=args.source;original=json.loads((source/'interval_manifest.json').read_text())
    assert original['status']=='complete' and original['signature']['f6_context']=='interval_only'
    verified=json.loads((source/'verification.json').read_text());assert all(v=='PASS' for v in verified.values())
    upstream=ROOT/original['signature']['source'];origin=json.loads((upstream/'data_manifest.json').read_text())
    assert sha(upstream/'data_manifest.json')==original['signature']['source_manifest_sha256']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in (source/'interval_manifest.json',source/'verification.json',source/'split_manifest.json',upstream/'data_manifest.json',Path(__file__))}
    for key,filename in [('f6','f6_tactile_vqvae.pt'),('deform','sharpa_wave_deform_encoder.pth')]:
        path=ROOT/'checkpoints/T-Rex/encoders'/filename;assert sha(path)==original['signature'][key+'_sha256'];hashes[str(path.relative_to(ROOT))]=sha(path)
    for record in origin['records'].values():
        signature=origin['signature']['episode_sources'][record['id']]
        for key,hashkey in [('synchronized_frames_path','frames_sha256'),('tactile_events_path','events_sha256')]:
            path=ROOT/record[key];assert sha(path)==signature[hashkey];hashes[record[key]]=signature[hashkey]
    events={(e['rollout_id'],e['event_index']):e for e in origin['source_intervals']};rows=[]
    for row in original['records']:
        event=events[(row['rollout_id'],row['event_index'])]
        assert event['event_key']==row['event_key'] and event['start_frame']==row['start_frame'] and event['end_frame']==row['end_frame']
        assert row['label']==int(row['event_key'] in (6,7))
        path=source/row['feature_path'];hashes[str(path.relative_to(ROOT))]=sha(path);data=load(path)
        assert data['f6'].shape==(row['ticks'],1280) and data['deform'].shape==(row['ticks'],2560)
        assert np.isfinite(data['f6']).all() and np.isfinite(data['deform']).all()
        assert data['video_frames'].min()>=row['start_frame'] and data['video_frames'].max()<=row['end_frame']
        endpoints=endpoint_indices(row['ticks']);assert endpoints[-1]==row['ticks']-1 and endpoints.min()>=0
        target=output/row['feature_path'];save(target,dict(**data,endpoints=endpoints,endpoint_ticks=data['ticks'][endpoints],endpoint_frames=data['video_frames'][endpoints],endpoint_labels=np.full(POSITIONS,row['label'],dtype=np.int64)))
        rows.append(dict(row,supervised_positions=POSITIONS,endpoint_indices=endpoints.tolist(),endpoint_ticks=data['ticks'][endpoints].tolist(),endpoint_frames=data['video_frames'][endpoints].tolist()))
    shutil.copy2(source/'split_manifest.json',output/'split_manifest.json')
    assert len(rows)==259
    counts=original['split_counts'];assert counts=={'train':{'intervals':184,'success':129,'failure':55},'val':{'intervals':36,'success':28,'failure':8},'test':{'intervals':39,'success':29,'failure':10}}
    assert all(sha(ROOT/path)==value for path,value in hashes.items())
    dump(output/'prefix_manifest.json',dict(status='complete',created_at=datetime.datetime.now().astimezone().isoformat(),source=str(source.relative_to(ROOT)),records=rows,split_counts=counts,split_sha256=sha(output/'split_manifest.json'),input_hashes=hashes,class_mapping={'0':'success (8/9)','1':'failure (6/7)'},unit='original annotated event interval; no merge; no background',encoder_window=16,supervised_positions=16,fractions=(np.arange(1,17)/16).tolist(),endpoint_rule='ceil(k*T/16)-1 for k=1..16 over valid feature sequence; last=end; duplicates retain weight if T<16',features='full per-valid-tick sequence; continuous past16 F6 encoding and same-tick deform; interval-only F6 left padding; no sparse downstream input',hindsight_labels=True,training_started=False))
    dump(output/'verification.json',dict(status='PASS',original_annotations_unchanged=True,original_259_intervals=True,equal_16_supervision_slots=True,closed_interval_feature_bounds=True,rollout_split_unchanged=True,source_sensor_hashes_verified=True,feature_source_verified=True,encoders_unchanged=True))
    snapshot=output/'code_snapshot';snapshot.mkdir();shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    print('PREFIX_DATA_COMPLETE',json.dumps(counts),flush=True)


def read_samples(source,manifest):
    samples={split:[] for split in ('train','val','test')}
    for row in manifest['records']:
        data=load(source/row['feature_path']);samples[row['split']].append(dict(row,**{k:data[k] for k in ('f6','deform','endpoints')}))
    return samples


def endpoint_batch(samples,device):
    batch=batch_data(samples,device);batch['endpoints']=torch.as_tensor(np.stack([s['endpoints'] for s in samples]),device=device)
    return batch


@torch.inference_mode()
def evaluate(model,samples,device):
    model.eval();probabilities=[]
    for start in range(0,len(samples),8):
        batch=endpoint_batch(samples[start:start+8],device)
        logits,_=model(batch['f6'],batch['deform']);chosen=logits.gather(1,batch['endpoints'])
        probabilities.append(chosen.sigmoid().cpu().numpy())
    p=np.concatenate(probabilities);y=np.array([r['label'] for r in samples]);assert p.shape==(len(samples),16) and np.isfinite(p).all()
    per_endpoint=[metrics(y,p[:,i]) for i in range(16)]
    pooled=metrics(np.repeat(y,16),p.reshape(-1));pooled['unit']='equal16 causal positions per interval (correlated within interval)'
    result=dict(intervals=len(samples),supervised_positions_per_interval=16,mean_prefix_balanced_accuracy=float(np.mean([m['balanced_accuracy'] for m in per_endpoint])),mean_prefix_macro_f1=float(np.mean([m['macro_f1'] for m in per_endpoint])),pooled_positions=pooled,final_endpoint=per_endpoint[-1],per_endpoint=per_endpoint)
    assert np.isclose(result['mean_prefix_balanced_accuracy'],pooled['balanced_accuracy'])
    return result,p


def train(args):
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.source/'prefix_manifest.json').read_text());samples=read_samples(args.source,manifest)
    assert all(sha(ROOT/path)==value for path,value in manifest['input_hashes'].items())
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [args.source/'prefix_manifest.json',args.source/'split_manifest.json',Path(__file__)]+[args.source/r['feature_path'] for r in manifest['records']]}
    norm=normalization(samples['train']);counts=np.bincount([r['label'] for r in samples['train']],minlength=2);positive_weight=float(counts[0]/counts[1]);results=[];started=time.monotonic()
    for seed in range(42,47):
        for kind in ('f6','deform','f6_deform'):
            for head in ('mlp','gru'):
                torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
                directory=output/'runs'/f'seed_{seed}'/(kind+'_'+head);directory.mkdir(parents=True)
                config=dict(input_kind=kind,head_kind=head);model=PrefixProbe(**config)
                for k,v in norm.items():getattr(model,k).copy_(v)
                model.to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                pos_weight=torch.tensor(positive_weight,dtype=torch.float32,device=args.device)
                best=-1;best_epoch=0;history=[]
                for epoch in range(1,31):
                    model.train();order=rng.permutation(len(samples['train']));loss_sum=0;sample_sum=0
                    for start in range(0,len(order),8):
                        chosen=[samples['train'][int(i)] for i in order[start:start+8]];batch=endpoint_batch(chosen,args.device)
                        optimizer.zero_grad(set_to_none=True);logits,_=model(batch['f6'],batch['deform'])
                        selected=logits.gather(1,batch['endpoints']);loss=prefix_loss(selected,batch['labels'],pos_weight);assert torch.isfinite(loss)
                        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                        loss_sum+=float(loss.detach())*len(chosen);sample_sum+=len(chosen)
                    assert sample_sum==184
                    val,_=evaluate(model,samples['val'],args.device);score=val['mean_prefix_balanced_accuracy']
                    history.append(dict(epoch=epoch,train_loss=loss_sum/sample_sum,train_intervals=sample_sum,train_supervised_slots=sample_sum*16,val_mean_prefix_balanced_accuracy=score,val_mean_prefix_macro_f1=val['mean_prefix_macro_f1'],val_final_balanced_accuracy=val['final_endpoint']['balanced_accuracy']))
                    if score>best+1e-8:
                        best=score;best_epoch=epoch
                        torch.save(dict(model_class='PrefixProbe',model_config=config,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},seed=seed,best_epoch=epoch,class_mapping={'0':'success','1':'failure'},output_unit='P(final failure|interval start:current)',training_interval_counts=counts.tolist(),pos_weight=positive_weight,supervised_positions=16,encoder_window=16,split_sha256=manifest['split_sha256'],threshold=.5),directory/'best.pt')
                    if epoch-best_epoch>=8:break
                blob=torch.load(directory/'best.pt',map_location=args.device,weights_only=True);model.load_state_dict(blob['state_dict'])
                val,_=evaluate(model,samples['val'],args.device);test,p=evaluate(model,samples['test'],args.device)
                result=dict(input=kind,head=head,seed=seed,best_epoch=best_epoch,epochs=len(history),train_counts=counts.tolist(),pos_weight=positive_weight,validation=val,test=test)
                dump(directory/'history.json',history);dump(directory/'metrics.json',result)
                save(directory/'test_predictions.npz',dict(labels=np.array([r['label'] for r in samples['test']]),probabilities=p,endpoints=np.stack([r['endpoints'] for r in samples['test']]),predictions=(p>=.5).astype(int)))
                records=[]
                for row,prob in zip(samples['test'],p):
                    for k in range(16):records.append(dict(interval_id=row['interval_id'],rollout_id=row['rollout_id'],event_key=row['event_key'],start_frame=row['start_frame'],end_frame=row['end_frame'],position=k+1,requested_fraction=(k+1)/16,feature_index=int(row['endpoints'][k]),current_tick=row['endpoint_ticks'][k],current_frame=row['endpoint_frames'][k],label=row['label'],p_failure=float(prob[k]),prediction=int(prob[k]>=.5)))
                with (directory/'test_predictions.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
                results.append(result);dump(output/'results.json',results)
                print('PREFIX_TRAIN',len(results),30,seed,kind,head,'mean BA',test['mean_prefix_balanced_accuracy'],'final BA',test['final_endpoint']['balanced_accuracy'],'epoch',best_epoch,'elapsed',round(time.monotonic()-started),flush=True)
                del model,optimizer
    assert all(sha(ROOT/path)==value for path,value in hashes.items())
    snapshot=output/'code_snapshot';snapshot.mkdir();shutil.copy2(Path(__file__),snapshot/Path(__file__).name)
    dump(output/'run_manifest.json',dict(status='complete',dataset=str(args.source.relative_to(ROOT)),seeds=list(range(42,47)),unique_runs=30,device=args.device,environment='repos/ProcVLM/.venv',torch_version=str(torch.__version__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),input_hashes=hashes,split_sha256=manifest['split_sha256'],elapsed_seconds=time.monotonic()-started,loss='BCEWithLogitsLoss pos_weight=129/55; mean16 endpoints per interval then mean intervals',selection='maximum validation mean BA across16 positions; first tie',batch_size=8,epochs=30,patience=8,lr=.001,weight_decay=.0001,clip=1,hidden=128,gru_layers=1,encoder_frozen=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','train'));p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args();args.source=args.source.resolve();args.output=args.output.resolve()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    if args.command=='prepare':prepare(args)
    else:train(args)

if __name__=='__main__':main()
