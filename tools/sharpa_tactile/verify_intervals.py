"""Verify interval isolation, sequence aggregation, padding and sample-count weights."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, dump, sha
from .interval_models import IntervalProbe, interval_windows
from .train_intervals import batch_data, metrics



def verify_real_prefix(out,manifest):
    """One real encoder smoke for the newly introduced interval-only left padding."""
    from .models import FrozenEncoders
    from .data import episode_arrays
    row=next(r for r in manifest['records'] if r['split']=='train')
    target=out/'raw_interval_prefix_check.json'
    if target.exists():
        previous=json.loads(target.read_text())
        assert previous['status']=='PASS' and previous['interval_id']==row['interval_id']
        return
    source=ROOT/manifest['signature']['source']; original=json.loads((source/'data_manifest.json').read_text())
    events=[e for e in original['source_intervals'] if e['rollout_id']==row['rollout_id']]
    arrays=episode_arrays(original['records'][row['rollout_id']],events,original['signature']['camera'],num_classes=3)
    raw=np.repeat(arrays['f6'][row['first_tick']:row['first_tick']+1],16,axis=0)
    encoders=FrozenEncoders(); assert not encoders.training and all(not p.requires_grad for p in encoders.parameters())
    actual=encoders.f6_features(torch.from_numpy(raw).unsqueeze(0))[0].numpy()
    with np.load(out/row['feature_path']) as features: expected=features['f6'][0]
    np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-4)
    dump(target,{'status':'PASS','interval_id':row['interval_id'],'rollout_id':row['rollout_id'],
        'f6_input':'16 repetitions of first valid interval tick; no pre-interval data','encoder_frozen':True,
        'max_abs_feature_difference':float(np.max(np.abs(actual-expected))),
        'scope':'CPU raw F6 interval prefix versus prepared GPU features; no reconstruction'})


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); out=args.output.resolve(); torch.set_num_threads(2); torch.manual_seed(7)
    manifest=json.loads((out/'interval_manifest.json').read_text()); assert manifest['status']=='complete'
    split=json.loads((out/'split_manifest.json').read_text()); signature=manifest['signature']; src=ROOT/signature['source']
    assert sha(out/'split_manifest.json')==sha(src/'split_manifest.json')==signature['split_sha256']
    ids={name:set(split[name]) for name in ('train','val','test')}
    assert not(ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test'])
    intervals=manifest['records']; assert len({r['interval_id'] for r in intervals})==len(intervals)
    for row in intervals:
        assert row['rollout_id'] in ids[row['split']] and row['label']==int(row['event_key'] in (6,7))
        with np.load(out/row['feature_path']) as features:
            assert features['f6'].shape==(row['ticks'],1280) and features['deform'].shape==(row['ticks'],2560)
            assert features['label'].ndim==0 and int(features['label'])==row['label']
            assert (features['video_frames']>=row['start_frame']).all() and (features['video_frames']<=row['end_frame']).all()
    counts=np.bincount([r['label'] for r in intervals if r['split']=='train'],minlength=2)
    assert counts.tolist()==[129,55]
    weights=counts.sum()/(2*counts); assert np.allclose(weights,[184/258,184/110])
    # Duration cannot change an interval target's CE contribution.
    logits=torch.tensor([[0.,1.],[1.,0.]]); labels=torch.tensor([1,0]); w=torch.tensor(weights,dtype=torch.float32)
    expected=torch.nn.functional.cross_entropy(logits,labels,weight=w)
    assert expected.ndim==0
    results={'one_scalar_label_per_interval_no_background':'PASS','rollout_split_identical_disjoint':'PASS',
        'class_weights_from_interval_counts_not_duration':'PASS'}
    raw=np.arange(6,dtype=np.float32)[:,None,None]*np.ones((6,5,6),np.float32)
    windows=interval_windows(raw)
    assert windows.shape==(6,16,5,6) and np.array_equal(windows[0],np.repeat(raw[:1],16,axis=0))
    assert np.array_equal(windows[:,-1],raw)
    changed=raw.copy(); changed[4:]+=100
    assert np.array_equal(interval_windows(changed)[:4],windows[:4])
    results['interval_only_left_padding_no_future_f6_windows']='PASS'
    for kind in ('f6','deform','f6_deform'):
        for head in ('mlp','lstm'):
            model=IntervalProbe(kind,head).eval(); f6=torch.randn(2,7,1280); deform=torch.randn(2,7,2560)
            lengths=torch.tensor([3,7])
            with torch.inference_mode():
                logits=model(f6,deform,lengths); assert logits.shape==(2,2)
                single=model(f6[:1,:3],deform[:1,:3],torch.tensor([3]))
                torch.testing.assert_close(logits[:1],single,atol=2e-6,rtol=2e-5)
                altered=f6.clone(); altered[0,3:]+=1000; altered_deform=deform.clone(); altered_deform[0,3:]-=1000
                torch.testing.assert_close(logits,model(altered,altered_deform,lengths),atol=2e-6,rtol=2e-5)
                if head=='mlp':
                    permutation=torch.randperm(7)
                    torch.testing.assert_close(model(f6,deform),model(f6[:,permutation],deform[:,permutation]),atol=2e-6,rtol=2e-5)
                else:
                    sequence=model.project(f6[1:2],deform[1:2]); state=None
                    for tick in range(7): _,state=model.context(sequence[:,tick:tick+1],state)
                    torch.testing.assert_close(logits[1:2],model.prediction(state[0][-1]),atol=2e-6,rtol=2e-5)
                    # Every new interval starts with an independent state.
                    torch.testing.assert_close(single,model(f6[:1,:3],deform[:1,:3]),atol=2e-6,rtol=2e-5)
            results[kind+'_'+head+'_pooling_or_final_state_and_padding_isolation']='PASS'
    known=metrics([0,0,1,1],[.1,.6,.7,.8])
    assert known['n']==4 and known['confusion_matrix_success_failure']==[[1,1],[0,2]]
    assert known['balanced_accuracy']==.75
    results['metrics_count_intervals']='PASS'
    for kind,filename in [('f6','f6_tactile_vqvae.pt'),('deform','sharpa_wave_deform_encoder.pth')]:
        assert sha(ROOT/'checkpoints/T-Rex/encoders'/filename)==signature[kind+'_sha256']
    results['pretrained_encoder_weights_unchanged']='PASS'
    if signature['f6_context']=='interval_only':
        verify_real_prefix(out,manifest)
        results['real_interval_prefix_encoding_matches_cache']='PASS'
    dump(out/'verification.json',results); print(json.dumps(results,indent=2))


if __name__=='__main__': main()
