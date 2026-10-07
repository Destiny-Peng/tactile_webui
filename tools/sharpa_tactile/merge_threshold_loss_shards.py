"""Merge disjoint learned-threshold heads; replay events from archived margins."""
from .common import project_path, relative_path
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
from .common import ROOT,dump,sha
from .early_warning import write_csv
from .calibrate_early_warning import evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--shards',type=Path,nargs='+',required=True);a=p.parse_args();out=a.output.resolve()
    existing=json.loads((out/'completed_before_parallel.json').read_text());lookup={(r['model'],r['horizon_frames'],r['seed']):r for r in existing};provenance=[]
    for shard in a.shards:
        shard=shard.resolve();assert json.loads((shard/'gpu_complete.json').read_text())['status']=='complete'
        for r in json.loads((shard/'results.json').read_text()):
            key=r['model'],r['horizon_frames'],r['seed']
            if r['model']=='fixed':
                if key in lookup:assert lookup[key]==r
                else:lookup[key]=r
                continue
            assert key not in lookup;lookup[key]=r
            rel=Path('runs')/r['model']/f"H{r['horizon_frames']}"/f"seed_{r['seed']}";src=shard/rel;dst=out/rel
            if dst.exists():
                backup=out/'interrupted_before_parallel'/rel;backup.parent.mkdir(parents=True,exist_ok=True);shutil.move(dst,backup)
            shutil.copytree(src,dst)
            for name in ('best.pt','history.json','metrics.json','val_predictions.npz','test_predictions.npz','oof_predictions.npz'):assert sha(src/name)==sha(dst/name)
            provenance.append(dict(model=r['model'],horizon_frames=r['horizon_frames'],seed=r['seed'],source=str(relative_path(shard))))
    expected={(k,h,s) for k in ('fixed','mlp','gru') for h in (0,8,15,30,45) for s in range(42,47)};assert set(lookup)==expected
    results=sorted(lookup.values(),key=lambda r:(r['horizon_frames'],r['seed'],['fixed','mlp','gru'].index(r['model'])));dump(out/'results.json',results)
    samples=json.loads((out/'samples.json').read_text())
    for split,ss in samples.items():
        with np.load(out/(split+'_coordinates.npz')) as z:frames=z['frames'].copy();offsets=z['offsets'].copy()
        for i,s in enumerate(ss):s['frames']=frames[offsets[i]:offsets[i+1]]
    source=project_path(json.loads((out/'protocol.json').read_text())['source']);events=[]
    for r in results:
        for split in ('val','test'):
            if r['model']=='fixed':
                path=source/'runs'/f"H{r['horizon_frames']}"/f"seed_{r['seed']}"/(split+'_predictions.npz');field='probabilities'
            else:
                path=out/'runs'/r['model']/f"H{r['horizon_frames']}"/f"seed_{r['seed']}"/(split+'_predictions.npz');field='margins'
            with np.load(path) as z:score=z[field].copy();offsets=z['offsets'].copy()
            f,e,rows=evaluate(samples[split],score,offsets,r['horizon_frames'],r['seed'],split,r['offset'],.15)
            assert f['confusion_matrix']==r[split]['confusion_matrix'] and e['confusion_matrix']==r[split+'_event']['confusion_matrix']
            events.extend([dict(model=r['model'],**v) for v in rows])
    write_csv(out/'event_predictions.csv',events)
    dump(out/'parallel_provenance.json',dict(completed_before_parallel=sum(r['model']!='fixed' for r in existing),merged=provenance,user_authorized_parallel=True,source_SHA_checked=True))
    dump(out/'gpu_complete.json',dict(status='complete',threshold_heads=50,inner_loss_fits=150,outer_refit_fits=150,full_refit_fits=50,parallel=True))
    shutil.copyfile(Path(__file__).with_name('dynamic_threshold_loss.py'),out/'dynamic_threshold_loss.py')
    print('THRESHOLD_MERGE_COMPLETE50',flush=True)


if __name__=='__main__':main()
