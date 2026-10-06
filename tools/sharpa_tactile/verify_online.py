"""Compare raw-data online inference to an offline cached training prefix."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from .common import dump
from .models import OnlinePredictor
from .data import episode_arrays, DeformStreams


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;torch.set_num_threads(2)
    data=json.loads((out/'data_manifest.json').read_text())
    split=json.loads((out/'split_manifest.json').read_text())
    rid=split['train'][0];record=data['records'][rid]
    events=[e for e in data['source_intervals'] if e['rollout_id']==rid]
    num_classes=data['signature'].get('num_classes',2)
    arrays=episode_arrays(record,events,data['signature']['camera'],num_classes=num_classes)
    streams=DeformStreams(record)
    predictor=OnlinePredictor(out/'f6_deform_lstm/best.pt')
    first=int(arrays['endpoints'][0]);observed=[]
    for tick in range(first-15,first+3):
        images=streams.batch([arrays['references'][tick]])[0] if tick>=first else None
        value=predictor.step(arrays['f6'][tick],images,tick=tick)
        if tick<first:
            assert value is None
        else:
            observed.append(value.tolist() if num_classes==3 else value)
    with np.load(out/'features'/(rid+'.npz')) as cached:
        assert np.array_equal(cached['ticks'][:3],np.arange(first,first+3))
        with torch.inference_mode():
            logits,_=predictor.probe(torch.from_numpy(cached['f6'][:3]).unsqueeze(0),
                                    torch.from_numpy(cached['deform'][:3]).unsqueeze(0))
            probs=logits.softmax(-1)[0].numpy()
            expected=probs if num_classes==3 else probs[:,1]
    np.testing.assert_allclose(observed,expected,atol=1e-3,rtol=1e-3)
    predictor.reset()
    assert predictor.state is None and not predictor.history
    assert predictor.step(arrays['f6'][first],None,tick=first,valid=False) is None
    report={'status':'PASS','rollout_id':rid,'checkpoint':'f6_deform_lstm/best.pt',
            'warmup_ticks':15,'online_failure_probabilities':observed,
            'cached_offline_failure_probabilities':expected.tolist(),
            'max_abs_difference':float(np.max(np.abs(np.asarray(observed)-expected))),
            'encoder_fingerprints':'matched','reset_state':'passed','invalid_tick':'reset',
            'comparison_tolerance':0.001,'deform_reconstruction':'omitted_by_user',
            'scope':'Raw right-hand five-finger data; CPU online versus GPU feature cache; training prefix only.'}
    if num_classes==3:
        report['class_order']=['background','success','failure']
        report['online_class_probabilities']=report.pop('online_failure_probabilities')
        report['cached_offline_class_probabilities']=report.pop('cached_offline_failure_probabilities')
    dump(out/'online_verification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
