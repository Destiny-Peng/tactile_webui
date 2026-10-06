from __future__ import annotations
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, dump, binary_timeline
from .data import rollout_split
from .models import BinaryProbe, FrozenEncoders


def main():
    parser=argparse.ArgumentParser(description='Verify causal heads, online state and frozen representations')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(7)
    results={}
    events=[{'event_key':6,'start_frame':1,'end_frame':3},
            {'event_key':8,'start_frame':3,'end_frame':5}]
    labels,conflict=binary_timeline(events,7)
    assert labels.tolist()==[-1,1,1,-1,0,0,-1] and conflict.sum()==1
    results['binary_mapping_conflict_and_background']='PASS'
    records={str(i):{'task_key':'usb'} for i in range(12)}
    ys={rid:np.array([0,1,-1]) for rid in records}
    split=rollout_split(records,ys)
    assert not set(split['train'])&set(split['val'])
    assert not set(split['test'])&(set(split['train'])|set(split['val']))
    results['rollout_split_disjoint']='PASS'
    for kind in ('f6','deform','f6_deform'):
        for head in ('mlp','lstm'):
            model=BinaryProbe(kind,head).eval()
            f6=torch.randn(1,7,1280);deform=torch.randn(1,7,2560)
            with torch.inference_mode():
                full,_=model(f6,deform)
                prefix,_=model(f6[:,:4],deform[:,:4])
                torch.testing.assert_close(full[:,:4],prefix,atol=2e-6,rtol=2e-5)
                changed=f6.clone();changed[:,4:]+=100
                changed_deform=deform.clone();changed_deform[:,4:]-=100
                perturbed,_=model(changed,changed_deform)
                torch.testing.assert_close(full[:,:4],perturbed[:,:4],atol=2e-6,rtol=2e-5)
                state=None;stream=[]
                for t in range(7):
                    value,state=model(f6[:,t:t+1],deform[:,t:t+1],state=state);stream.append(value)
                torch.testing.assert_close(full,torch.cat(stream,dim=1),atol=2e-6,rtol=2e-5)
            results[kind+'_'+head+'_prefix_and_online_equivalence']='PASS'
    encoders=FrozenEncoders()
    before={key:value.clone() for key,value in encoders.state_dict().items()}
    encoders.train(True)
    assert not encoders.training and not encoders.vq.training and not encoders.deform.training
    assert all(not p.requires_grad for p in encoders.parameters())
    features=encoders.f6_features(torch.randn(2,16,5,6))
    assert features.shape==(2,1280) and not features.requires_grad
    assert all(torch.equal(value,encoders.state_dict()[key]) for key,value in before.items())
    results['encoders_frozen_parameters_and_state']='PASS'
    results['deform_512D_contract']='128 channels x pooled 2 x 2 per finger; five fingers = 2560D'
    dump(args.output/'verification.json',results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
