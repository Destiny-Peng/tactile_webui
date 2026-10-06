"""Audit hard Key labels, temporal sampling, split integrity, GRU causality and saved metrics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, dump, sha
from .align_windows_data import CLASSES, STEPS, WINDOW, load_events, make_windows, labels_for, expanded_counts
from .align_windows import AlignWindowProbe, metrics, read_dataset, normalization, evaluate
from .train_three import class_weights


def verify(output, completed=False):
    torch.set_num_threads(2); manifest,events,f6,deform=load_events(output); signature=manifest['signature']
    split=json.loads((output/'split_manifest.json').read_text()); sets=[set(split[s]) for s in ('train','val','test')]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert sha(output/'split_manifest.json')==signature['split_sha256']
    source=ROOT/signature['source']; original=json.loads((source/'data_manifest.json').read_text())
    assert sha(source/'data_manifest.json')==signature['source_manifest_sha256']
    assert sha(ROOT/original['intervals'])==signature['annotation_sha256']
    for key,digest in signature['encoder_sha256'].items():
        path=ROOT/'checkpoints/T-Rex/encoders'/('f6_tactile_vqvae.pt' if key=='f6' else 'sharpa_wave_deform_encoder.pth')
        assert sha(path)==digest
    for path,digest in signature['source_hashes'].items(): assert sha(ROOT/path)==digest
    assert {e['event_key'] for e in events}=={6,8}
    assert len(events)==sum(int(e['event_key']) in (6,8) for e in original['source_intervals'])
    for event in events:
        assert event['rollout_id'] in split[event['split']]
        assert event['outcome']==(1 if event['event_key']==8 else 2)
        frames=event['video_frames']
        for insert in original['source_intervals']:
            if insert['rollout_id']==event['rollout_id'] and int(insert['event_key']) in (7,9):
                assert not ((frames>=insert['start_frame'])&(frames<=insert['end_frame'])).any()
    assert np.isfinite(f6).all() and np.isfinite(deform).all()
    # Explicit closed-boundary and skipped-Key semantics.
    sampled=np.tile(np.arange(0,46,3),(5,1)); keys=np.array([-1,0,2,45,46]); outcomes=np.array([2,1,2,2,1])
    assert labels_for(sampled,keys,outcomes,'span').tolist()==[0,1,2,2,0]
    assert labels_for(sampled,keys,outcomes,'sampled').tolist()==[0,1,0,2,0]
    grid=json.loads((output/'dataset_grid.json').read_text()); checked=0; primary_banks={}
    for cell in grid:
        config=cell['config']; data=read_dataset(output,config)
        for name in ('train','val','test'):
            indices,meta=make_windows(events,config,name)
            assert np.array_equal(indices,data[name]['indices']) and meta==data[name]['metadata']
            assert indices.shape==(len(meta),WINDOW)
            for row in meta:
                frames=np.asarray(row['sample_frames']); ticks=np.asarray(row['sample_ticks'])
                assert (np.diff(frames)>=0).all() and frames[-1]==row['window_end_frame']
                assert ticks.max()==ticks[-1] and len(frames)==16
                if row['padded_frames']==0: assert (np.diff(frames)==row['step']).all()
                assert row['event_key'] in (6,8) and row['rollout_id'] in split[name]
                if config['padding']=='drop': assert row['padded_frames']==0
            assert set(np.unique(data[name]['labels'])).issubset({0,1,2})
            counts=expanded_counts(meta,config if name=='train' else {**config,'jitter':0})
            assert counts.tolist()==cell['splits'][name]['class_counts']
            if cell['eligible']: assert (counts>0).all()
            if name!='train': assert {r['step'] for r in meta}.issubset(STEPS)
            if config['phase']=='primary' and name!='train':
                if name not in primary_banks: primary_banks[name]=(indices.copy(),data[name]['labels'].copy())
                assert np.array_equal(primary_banks[name][0],indices) and np.array_equal(primary_banks[name][1],data[name]['labels'])
            checked+=len(meta)
        counts=expanded_counts(data['train']['metadata'],config)
        if cell['eligible']:
            for mode in ('inverse_frequency','unweighted','sqrt_inverse'):
                assert np.isfinite(class_weights(counts,mode)).all()
    # Deterministic synthetic metrics with independent known confusion counts.
    truth=np.array([0,0,1,1,2,2]); predictions=np.array([0,2,1,0,2,2]); probs=np.eye(3)[predictions]
    result=metrics(truth,probs)
    assert result['confusion_matrix']==[[1,0,1],[1,1,0],[0,0,2]]
    assert abs(result['accuracy']-4/6)<1e-12 and abs(result['balanced_accuracy']-2/3)<1e-12
    # GRU future perturbation leaves all earlier outputs invariant, streamed state equals batch state.
    torch.manual_seed(123); synthetic_f6=torch.randn(2,16,1280); synthetic_deform=torch.randn(2,16,2560)
    model_checks={}
    for kind in ('f6','deform','f6_deform'):
        model=AlignWindowProbe(kind,'gru').eval()
        with torch.no_grad():
            logits,_=model.sequence(synthetic_f6,synthetic_deform)
            altered_f6=synthetic_f6.clone(); altered_deform=synthetic_deform.clone()
            altered_f6[:,9:]+=100; altered_deform[:,9:]-=100
            changed,_=model.sequence(altered_f6,altered_deform)
            assert torch.allclose(logits[:,:9],changed[:,:9],atol=1e-6,rtol=1e-5)
            state=None; streamed=[]
            for t in range(16):
                p,state=model.sequence(synthetic_f6[:,t:t+1],synthetic_deform[:,t:t+1],state=state); streamed.append(p)
            assert torch.allclose(torch.cat(streamed,1),logits,atol=1e-6,rtol=1e-5)
            assert torch.allclose(model(synthetic_f6,synthetic_deform),logits[:,-1],atol=1e-6,rtol=1e-5)
            batch=model(synthetic_f6,synthetic_deform)
            assert torch.allclose(batch,torch.cat([model(synthetic_f6[i:i+1],synthetic_deform[i:i+1]) for i in range(2)]),atol=1e-6,rtol=1e-5)
        mlp=AlignWindowProbe(kind,'mlp').eval(); permutation=torch.randperm(16)
        with torch.no_grad():
            assert torch.allclose(mlp(synthetic_f6,synthetic_deform),mlp(synthetic_f6[:,permutation],synthetic_deform[:,permutation]),atol=1e-6,rtol=1e-5)
        model_checks[kind]='PASS: GRU causality, stream/final equivalence, reset, MLP mean invariance'
    run_count=0; checkpoint_count=0
    if completed:
        status=json.loads((output/'suite_status.json').read_text()); results=json.loads((output/'results.json').read_text())
        assert status['status']=='training_complete' and len(results)==status['runs']
        expected={(g['config']['id'],42,k+'_'+h) for g in grid if g['eligible'] for k in ('f6','deform','f6_deform') for h in ('mlp','gru')}
        expected|={(c,s,k+'_'+h) for c in status['repeat_configs'] for s in (43,44,45,46) for k in ('f6','deform','f6_deform') for h in ('mlp','gru')}
        actual={(r['config']['id'],r['seed'],r['name']) for r in results}
        assert expected==actual and len(actual)==len(results)
        ranking=json.loads((output/'validation_ranking.json').read_text())
        for ranked in ranking:
            runs=[r for r in results if r['seed']==42 and r['config']['id']==ranked['config']['id']]
            assert abs(np.mean([r['validation']['balanced_accuracy'] for r in runs])-ranked['validation_mean_balanced_accuracy'])<1e-12
        for row in results:
            directory=output/'runs'/row['config']['id']/f"seed_{row['seed']}"/row['name']
            counts=expanded_counts(read_dataset(output,row['config'])['train']['metadata'],row['config'])
            assert row['training_class_counts']==counts.tolist()
            assert np.allclose(row['training_class_weights'],class_weights(counts,row['config']['weight_mode']))
            for splitname,metricname in [('test','test'),('validation','validation')]:
                records=list(csv.DictReader((directory/(splitname+'_predictions.csv')).open()))
                calculated=metrics([int(r['label']) for r in records],[[float(r[c+'_probability']) for c in CLASSES] for r in records])
                assert calculated==row[metricname]
                assert all(int(r['prediction'])==np.argmax([float(r[c+'_probability']) for c in CLASSES]) for r in records)
            history=json.loads((directory/'history.json').read_text())
            assert len(history)>=2*row['config']['jitter']+1
            assert row['best_epoch']==max([h for h in history if h['epoch']>=2*row['config']['jitter']+1],key=lambda h:h['val_balanced_accuracy'])['epoch']
            blob=torch.load(directory/'best.pt',weights_only=True,map_location='cpu')
            assert blob['output_unit']=='align_window' and blob['encoder_sha256']==signature['encoder_sha256']
            assert blob['split_sha256']==signature['split_sha256'] and blob['config']==row['config']
            assert all(not k.startswith(('vq.','deform.')) for k in blob['state_dict'])
            run_count+=1; checkpoint_count+=1
        # Replay real held-out predictions from one checkpoint per group at the validation-selected configuration.
        best=status['repeat_configs'][0]; chosen_config=next(g['config'] for g in grid if g['config']['id']==best)
        dataset=read_dataset(output,chosen_config)
        local=[]
        for event in events:
            with np.load(output/event['feature_path']) as cache: local.append(cache['f6_local'].copy())
        tables=dict(f6=torch.from_numpy(np.concatenate(local)),deform=torch.from_numpy(deform))
        replay={}
        for kind in ('f6','deform','f6_deform'):
            for head in ('mlp','gru'):
                directory=output/'runs'/best/'seed_42'/(kind+'_'+head)
                blob=torch.load(directory/'best.pt',weights_only=True,map_location='cpu'); model=AlignWindowProbe(**blob['model_config']).eval()
                model.load_state_dict(blob['state_dict'],strict=True)
                small={key:value[:16] if isinstance(value,np.ndarray) else value[:16] for key,value in dataset['test'].items()}
                _,p=evaluate(model,small,tables,'cpu',8)
                records=list(csv.DictReader((directory/'test_predictions.csv').open()))[:16]
                expected_probs=np.array([[float(r[c+'_probability']) for c in CLASSES] for r in records])
                difference=float(np.max(np.abs(p-expected_probs)))
                assert difference<2e-4 and np.array_equal(p.argmax(1),expected_probs.argmax(1))
                replay[kind+'_'+head]=difference
        dump(output/'checkpoint_replay.json',dict(status='PASS',cpu_gpu_max_probability_errors=replay))
    report=dict(status='PASS',stage='completed' if completed else 'prepared',events=len(events),
        grid_cells=len(grid),eligible_cells=sum(g['eligible'] for g in grid),checked_window_rows=checked,
        original_split_unchanged=True,original_features_and_encoder_hashes_unchanged=True,hard_key_labels=True,
        no_insert_samples_or_frames=True,common_primary_evaluation_banks=True,
        model_checks=model_checks,metrics_runs_checked=run_count,checkpoints_checked=checkpoint_count)
    dump(output/('verification.json' if completed else 'preflight_verification.json'),report)
    print('ALIGN_VERIFY_PASS',json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--completed',action='store_true')
    a=p.parse_args();verify(a.output.resolve(),a.completed)


if __name__=='__main__':main()
