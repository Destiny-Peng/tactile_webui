"""Meaningful contracts for hard-label three-class data, padding and causal probes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .common import dump, three_class_timeline, sha
from .models import FrameProbe, FrozenEncoders
from .train_three import metrics, batch_data


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); torch.set_num_threads(2); torch.manual_seed(7); results = {}
    events = [{'event_key':6,'start_frame':1,'end_frame':2},{'event_key':7,'start_frame':2,'end_frame':3},
              {'event_key':8,'start_frame':4,'end_frame':4},{'event_key':9,'start_frame':5,'end_frame':5}]
    assert three_class_timeline(events,7).tolist() == [0,2,2,2,1,1,0]
    assert three_class_timeline([{'event_key':1,'start_frame':0,'end_frame':6}],7).tolist() == [0]*7
    try: three_class_timeline(events+[{'event_key':8,'start_frame':3,'end_frame':4}],7)
    except ValueError: pass
    else: raise AssertionError('Conflicting labels silently accepted')
    results['hard_labels_closed_intervals_background_no_ignore'] = 'PASS'
    data = json.loads((args.output/'data_manifest.json').read_text())
    from .common import ROOT
    previous = ROOT/data['signature']['previous_output']
    assert sha(args.output/'split_manifest.json') == sha(previous/'split_manifest.json')
    split = json.loads((args.output/'split_manifest.json').read_text())
    groups = [set(split[n]) for n in ('train','val','test')]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    for name in ('train','val','test'):
        counts = np.zeros(3,dtype=np.int64)
        for rid in split[name]:
            with np.load(args.output/'features'/(rid+'.npz')) as cache:
                assert set(np.unique(cache['labels'])).issubset({0,1,2})
                timeline = np.load(args.output/'frame_labels'/(rid+'.npy'))
                assert np.array_equal(cache['labels'],timeline[cache['video_frames']])
                counts += np.bincount(cache['labels'],minlength=3)
        assert counts.tolist() == [data['split_counts'][name][key] for key in ('background_rows','success_rows','failure_rows')]
        assert (counts > 0).all()
    results['exact_previous_split_and_all_cached_frame_labels'] = 'PASS'
    toy = metrics([0,0,1,1,2,2],np.eye(3)[[0,1,1,1,2,0]])
    assert toy['confusion_matrix'] == [[1,1,0],[0,2,0],[1,0,1]]
    assert abs(toy['balanced_accuracy']-2/3) < 1e-12
    assert toy['per_class']['background']['support'] == 2
    results['three_class_metrics_known_confusion_matrix'] = 'PASS'
    sequences = [{'labels':np.array([0,1,2]),'f6':np.zeros((3,1280),np.float32),'deform':np.zeros((3,2560),np.float32)},
                 {'labels':np.array([2]),'f6':np.zeros((1,1280),np.float32),'deform':np.zeros((1,2560),np.float32)}]
    batch = batch_data(sequences,'cpu'); mask = batch['valid_positions']
    assert mask.sum() == 4 and (batch['labels'] >= 0).all()
    logits = torch.randn(2,3,3,requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits[mask],batch['labels'][mask],weight=torch.tensor([0.5,2.,3.]))
    changed = logits.detach().clone(); changed[~mask] += 1000
    torch.testing.assert_close(loss,torch.nn.functional.cross_entropy(changed[mask],batch['labels'][mask],weight=torch.tensor([0.5,2.,3.])))
    loss.backward(); assert (logits.grad[~mask] == 0).all()
    results['padding_excluded_by_lengths_not_ignore_label'] = 'PASS'
    for kind in ('f6','deform','f6_deform'):
        for head in ('mlp','lstm'):
            model = FrameProbe(kind,head,num_classes=3).eval(); f6 = torch.randn(1,7,1280); deform = torch.randn(1,7,2560)
            with torch.inference_mode():
                full,_ = model(f6,deform); assert full.shape == (1,7,3)
                prefix,_ = model(f6[:,:4],deform[:,:4])
                torch.testing.assert_close(full[:,:4],prefix,atol=2e-6,rtol=2e-5)
                changed_f6 = f6.clone(); changed_deform = deform.clone()
                changed_f6[:,4:] += 100; changed_deform[:,4:] -= 100
                changed,_ = model(changed_f6,changed_deform)
                torch.testing.assert_close(full[:,:4],changed[:,:4],atol=2e-6,rtol=2e-5)
                state = None; steps = []
                for t in range(7):
                    value,state = model(f6[:,t:t+1],deform[:,t:t+1],state=state); steps.append(value)
                torch.testing.assert_close(full,torch.cat(steps,dim=1),atol=2e-6,rtol=2e-5)
            results[kind+'_'+head+'_three_outputs_causal_stream_equivalence'] = 'PASS'
    encoders = FrozenEncoders(); before = {key:v.clone() for key,v in encoders.state_dict().items()}
    encoders.train(True); assert not encoders.training and all(not p.requires_grad for p in encoders.parameters())
    encoders.f6_features(torch.randn(2,16,5,6))
    assert all(torch.equal(v,encoders.state_dict()[key]) for key,v in before.items())
    results['encoders_frozen_parameters_and_state'] = 'PASS'
    old = torch.load(previous/'f6_deform_lstm/best.pt',weights_only=True,map_location='cpu')
    FrameProbe(**old['model_config']).load_state_dict(old['state_dict'],strict=True)
    results['previous_binary_checkpoint_still_loads'] = 'PASS'
    dump(args.output/'verification.json',results); print(json.dumps(results,indent=2))


if __name__ == '__main__': main()
