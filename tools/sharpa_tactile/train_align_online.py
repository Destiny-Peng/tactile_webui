"""Train on immutable prepared online-window labels, without relabelling."""
import argparse
import csv
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
from .align_windows import AlignWindowProbe, normalization, batch_features, evaluate, metrics

GROUPS = ('symmetric_separate', 'asymmetric_merged_success')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=8)
    args = parser.parse_args()
    args.dataset = args.dataset.resolve(); args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((args.dataset/'dataset_manifest.json').read_text())
    assert json.loads((args.dataset/'verification.json').read_text())
    rollouts = manifest['rollouts']
    arrays = {'f6': [], 'deform': []}; hashes = {}
    def record(path):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    record(args.dataset/'dataset_manifest.json')
    offset = 0
    for row in rollouts:
        path = args.dataset/row['feature_path']; record(path)
        with np.load(path) as data:
            assert row['feature_offset'] == offset
            for key in arrays:
                array = data[key].copy(); assert np.isfinite(array).all()
                arrays[key].append(array)
            offset += len(data['f6'])
    banks = {key: np.concatenate(value) for key, value in arrays.items()}
    tables = {key: torch.from_numpy(value).to(args.device) for key, value in banks.items()}
    for path, expected in manifest['source_hashes'].items():
        if path.startswith('checkpoints/'):
            assert sha(ROOT/path) == expected
            hashes[path] = expected
    record(Path(__file__))
    record(ROOT/'tools/sharpa_tactile/align_windows.py')
    record(ROOT/'tools/sharpa_tactile/models.py')
    results = []; started = time.monotonic()
    for group in GROUPS:
        dataset = {}
        for split in ('train', 'val', 'test'):
            path = args.dataset/'ready'/group/(split+'.npz'); record(path)
            with np.load(path) as data:
                dataset[split] = {key: data[key].copy() for key in data.files}
            data = dataset[split]
            assert data['indices'].shape == (len(data['labels']),16)
            for ri in np.unique(data['rollout_index']):
                row = rollouts[int(ri)]; selected = data['indices'][data['rollout_index']==ri]
                assert row['split']==split and selected.min()>=row['feature_offset']
                assert selected.max()<row['feature_offset']+row['feature_rows']
        counts = np.bincount(dataset['train']['labels'], minlength=3)
        assert counts.min()>0
        weights = 1/counts; weights = weights/weights.mean()
        norms = {key: normalization(value,dataset['train']['indices']) for key,value in banks.items()}
        for seed in range(42,47):
            for kind in ('f6','deform','f6_deform'):
                for head in ('mlp','gru'):
                    directory = args.output/'runs'/group/f'seed_{seed}'/(kind+'_'+head)
                    directory.mkdir(parents=True)
                    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                    config = dict(input_kind=kind,head_kind=head,hidden=128,layers=1)
                    model = AlignWindowProbe(**config)
                    for key,(mean,std) in norms.items():
                        getattr(model,key+'_mean').copy_(mean); getattr(model,key+'_std').copy_(std)
                    model.to(args.device)
                    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=args.device))
                    rng = np.random.default_rng(seed); history = []; best = -1; best_epoch = 0
                    for epoch in range(1,args.epochs+1):
                        model.train(); order = rng.permutation(len(dataset['train']['labels'])); total = 0
                        for start in range(0,len(order),128):
                            chosen = order[start:start+128]
                            indices = torch.as_tensor(dataset['train']['indices'][chosen],device=args.device)
                            target = torch.as_tensor(dataset['train']['labels'][chosen],device=args.device)
                            optimizer.zero_grad(set_to_none=True)
                            loss = criterion(model(**batch_features(tables,indices,kind)),target)
                            assert torch.isfinite(loss)
                            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
                            total += float(loss.detach())*len(chosen)
                        val,_ = evaluate(model,dataset['val'],tables,args.device)
                        history.append(dict(epoch=epoch,train_loss=total/len(order),val_balanced_accuracy=val['balanced_accuracy'],val_macro_f1=val['macro_f1']))
                        if val['balanced_accuracy']>best+1e-8:
                            best = val['balanced_accuracy']; best_epoch = epoch
                            torch.save(dict(model_class='AlignWindowProbe',model_config=config,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},class_mapping={0:'in_progress',1:'success',2:'failure'},output_unit='align_window',group=group,seed=seed,best_epoch=epoch,training_class_counts=counts.tolist(),training_class_weights=weights.tolist(),dataset_manifest_sha256=hashes[str((args.dataset/'dataset_manifest.json').relative_to(ROOT))],split_sha256=manifest['split_sha256']),directory/'best.pt')
                        if epoch-best_epoch>=args.patience:
                            break
                    checkpoint = torch.load(directory/'best.pt',weights_only=True,map_location=args.device)
                    model.load_state_dict(checkpoint['state_dict'])
                    val,_ = evaluate(model,dataset['val'],tables,args.device)
                    test,p = evaluate(model,dataset['test'],tables,args.device)
                    result = dict(group=group,input=kind,head=head,seed=seed,best_epoch=best_epoch,epochs=len(history),train_counts=counts.tolist(),class_weights=weights.tolist(),validation=val,test=test)
                    dump(directory/'history.json',history); dump(directory/'metrics.json',result)
                    np.savez_compressed(directory/'test_predictions.npz',**dataset['test'],probabilities=p,predictions=p.argmax(1))
                    records=[]
                    for i,label in enumerate(dataset['test']['labels']):
                        data=dataset['test']; records.append(dict(rollout_id=rollouts[int(data['rollout_index'][i])]['rollout_id'],n=int(data['n'][i]),step=int(data['step'][i]),sample_frames=json.dumps(data['frames'][i].tolist()),sample_ticks=json.dumps(data['ticks'][i].tolist()),label=int(label),prediction=int(p[i].argmax()),p_in_progress=float(p[i,0]),p_success=float(p[i,1]),p_failure=float(p[i,2])))
                    with (directory/'test_predictions.csv').open('w',newline='') as file:
                        writer=csv.DictWriter(file,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
                    assert metrics(dataset['test']['labels'],p)==test
                    assert best_epoch == next(row['epoch'] for row in history if abs(row['val_balanced_accuracy']-best)<1e-8)
                    results.append(result)
                    dump(args.output/'results.json',results)
                    print(f'{len(results)}/60 {group} {seed} {kind} {head}: BA={test["balanced_accuracy"]:.4f} F1={test["macro_f1"]:.4f} epoch={best_epoch} elapsed={time.monotonic()-started:.0f}s',flush=True)
                    del model,optimizer
    assert all(sha(ROOT/path)==value for path,value in hashes.items())
    snapshot=args.output/'code_snapshot'; snapshot.mkdir()
    for name in ('train_align_online.py','align_windows.py','models.py'):
        shutil.copy2(ROOT/'tools/sharpa_tactile'/name,snapshot/name)
    dump(args.output/'run_manifest.json',dict(status='complete',dataset=str(args.dataset.relative_to(ROOT)),device=args.device,seeds=list(range(42,47)),groups=list(GROUPS),epochs=args.epochs,patience=args.patience,batch_size=128,learning_rate=.001,weight_decay=.0001,hidden=128,layers=1,selection='maximum validation balanced accuracy; first tie',loss='inverse-frequency weighted cross entropy normalized to mean 1',frozen_encoders=True,labels='saved ready NPZ labels unchanged',input_hashes=hashes,split_sha256=manifest['split_sha256'],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),torch_version=str(torch.__version__),elapsed_seconds=time.monotonic()-started))
    dump(args.output/'verification.json',dict(status='PASS',runs=len(results),input_hashes_unchanged=True,rollout_split_isolation=True,saved_labels_unchanged=True,metrics_recomputed=True,selection_checked=True))

if __name__ == '__main__':
    main()
