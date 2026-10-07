"""Merge completed critical runs and independent GPU shards without retraining."""
from .common import relative_path
import argparse
import datetime
import json
import shutil
from pathlib import Path
import numpy as np
from .common import ROOT,dump,sha
from .early_warning import write_csv


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--shards',type=Path,nargs='+',required=True);a=p.parse_args();out=a.output.resolve()
    results=json.loads((out/'completed_before_parallel.json').read_text());seen={(r['n'],r['m'],r['seed']) for r in results};provenance=[]
    for shard in a.shards:
        shard=shard.resolve();assert json.loads((shard/'gpu_complete.json').read_text())['status']=='complete'
        rr=json.loads((shard/'results.json').read_text())
        for r in rr:
            key=r['n'],r['m'],r['seed'];assert key not in seen;seen.add(key)
            rel=Path('runs')/f"n{r['n']}_m{r['m']}"/f"seed_{r['seed']}";src=shard/rel;dst=out/rel
            if dst.exists():
                backup=out/'interrupted_before_parallel'/rel;backup.parent.mkdir(parents=True,exist_ok=True);shutil.move(dst,backup)
            shutil.copytree(src,dst)
            for name in ('best.pt','metrics.json','history.json','val_predictions.npz','test_predictions.npz'):
                assert sha(src/name)==sha(dst/name)
            provenance.append(dict(n=r['n'],m=r['m'],seed=r['seed'],source=str(relative_path(shard))))
        results.extend(rr)
    expected={(n,m,s) for n in (0,5,10,15,20,25,30) for m in (0,3,5,10,15) for s in range(42,47)}
    assert seen==expected and len(results)==175;results.sort(key=lambda r:(r['n'],r['m'],r['seed']));dump(out/'results.json',results);summary=[]
    for n in (0,5,10,15,20,25,30):
        for m in (0,3,5,10,15):
            rr=[r for r in results if (r['n'],r['m'])==(n,m)];row=dict(n=n,m=m,seeds=5)
            for phase in ('validation','test','validation_event','test_event'):
                for key in ('balanced_accuracy','macro_f1','precision','recall','false_positive_rate'):
                    values=[r[phase][key] for r in rr];row[phase+'_'+key+'_mean']=float(np.mean(values));row[phase+'_'+key+'_std']=float(np.std(values,ddof=1))
            for phase in ('validation','test'):
                values=[r[phase]['bce_loss'] for r in rr];row[phase+'_bce_loss_mean']=float(np.mean(values));row[phase+'_bce_loss_std']=float(np.std(values,ddof=1))
            row['test_positive']=rr[0]['test']['positive_count'];row['test_negative']=rr[0]['test']['negative_count'];summary.append(row)
    selected=max(summary,key=lambda r:(r['validation_balanced_accuracy_mean'],r['validation_macro_f1_mean'],-r['n']-r['m'],-r['n'],-r['m']))
    dump(out/'selected_config.json',selected);write_csv(out/'summary.csv',summary)
    dump(out/'parallel_provenance.json',dict(completed_before_parallel=97,merged=provenance,source_sha_checked=True,user_authorized_parallel=True))
    dump(out/'manifest.json',dict(status='complete',runs=175,configs=35,created_at=datetime.datetime.now().astimezone().isoformat(),parallel=True,selection='validation BCE per-run, valBA across n/m'))
    dump(out/'gpu_complete.json',dict(status='complete',runs=175,epochs=300,patience=50,stopping='validation BCE',parallel=True))
    print('MERGE_COMPLETE175',selected['n'],selected['m'],flush=True)


if __name__=='__main__':main()
