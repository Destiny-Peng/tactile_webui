"""Dataset-only revision: posterior-half Key bands and merged-success control."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
from .common import ROOT, dump, sha
from .data import episode_arrays
from .models import FrozenEncoders

CLASSES=('in_progress','success','failure')
GROUPS=('symmetric_separate','asymmetric_merged_success')
WINDOW=16
HALF=8


def units_for(events, group):
    aligns=sorted([e for e in events if e['event_key'] in (6,8)],key=lambda e:(e['end_frame'],e['event_index']))
    if group==GROUPS[1] and {e['event_key'] for e in aligns}=={6,8}:
        assert aligns[-1]['event_key']==8, 'User assumes a mixed rollout ends in Align success'
        last=aligns[-1]
        return [dict(start_frame=min(e['start_frame'] for e in aligns),end_frame=max(e['end_frame'] for e in aligns),
            outcome=1,event_index=last['event_index'],merged=True,member_indices=[e['event_index'] for e in aligns])]
    return [dict(start_frame=e['start_frame'],end_frame=e['end_frame'],outcome=1 if e['event_key']==8 else 2,
        event_index=e['event_index'],merged=False,member_indices=[e['event_index']]) for e in aligns]


def regions_for(row,group,n):
    result=[]
    for unit in units_for(row['events'],group):
        following=[e for e in row['events'] if e['event_key'] in (7,9) and e['event_index']>unit['event_index']]
        insert=min(following,key=lambda e:e['event_index']) if following else None
        upper=insert['start_frame']+n if group==GROUPS[1] and insert else unit['end_frame']+n
        result.append(dict(**unit,key_start=max(row['annotation_start'],unit['end_frame']-n),
            key_end=min(row['annotation_end'],upper),insert_start=insert['start_frame'] if insert else None,
            no_insert_fallback=group==GROUPS[1] and insert is None))
    return sorted(result,key=lambda r:(r['end_frame'],r['event_index']))


def frame_labels(row,regions):
    labels=np.zeros(row['total_frames'],dtype=np.int64)
    for region in regions:
        assert region['key_start']<=region['key_end']
        labels[region['key_start']:region['key_end']+1]=region['outcome']
    return labels


def window_labels(frames,timeline):
    rear=timeline[frames[:,HALF:]]
    present=(rear!=0).any(1)
    # Nearest preceding Key to the last sample; never consult the front half.
    last=HALF-1-np.argmax((rear!=0)[:,::-1],axis=1)
    return np.where(present,rear[np.arange(len(frames)),last],0).astype(np.int64)


def candidate_windows(cache,row,step):
    ticks=cache['ticks']; video=cache['video_frames']; indices=[];frames=[]; padded=[];missing=0
    # Tactile invalid gaps reset context. Repeated camera frames are resolved to the latest past tick.
    cuts=np.r_[0,np.flatnonzero(np.diff(ticks)!=1)+1,len(ticks)]
    for left,right in zip(cuts[:-1],cuts[1:]):
        mapping={int(video[i]):i for i in range(left,right)}
        first=min(mapping)
        for endpoint,index in sorted(mapping.items()):
            desired=endpoint-np.arange(WINDOW-1,-1,-1)*step
            actual=np.maximum(desired,first)
            if not all(int(f) in mapping for f in actual):
                missing+=1;continue
            local=np.array([mapping[int(f)] for f in actual],dtype=np.int64)
            assert (ticks[local]<=ticks[index]).all()
            indices.append(local);frames.append(actual);padded.append(int((desired<first).sum()))
    return dict(indices=np.array(indices,dtype=np.int64).reshape(-1,WINDOW),
        frames=np.array(frames,dtype=np.int64).reshape(-1,WINDOW),
        padded=np.array(padded,dtype=np.int64),missing_desired_camera_frames=missing)


def concat(parts):
    return {key:np.concatenate([p[key] for p in parts]) for key in ('indices','frames','ticks','labels','rollout_index','n','step','padded')}


def choose(data,positions):
    return {key:value[positions] for key,value in data.items()}


def counts(data):
    number=np.bincount(data['labels'],minlength=3)
    return dict(total=int(number.sum()),counts={c:int(number[i]) for i,c in enumerate(CLASSES)},
        percentages={c:float(number[i]/number.sum()*100) if number.sum() else 0 for i,c in enumerate(CLASSES)},
        rollouts=int(len(np.unique(data['rollout_index']))),padded_windows=int((data['padded']>0).sum()))


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data)


def csv_counts(path,records):
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def slow_label(frames,regions):
    for frame in reversed(frames[HALF:]):
        matches=[r for r in regions if r['key_start']<=frame<=r['key_end']]
        if matches:
            return max(matches,key=lambda r:(r['end_frame'],r['event_index']))['outcome']
    return 0


def verify(output,manifest,case_summaries,balanced):
    rows=manifest['rollouts'];checked=0
    # Boundary, front-half exclusion, nearest-Key rule and overlapped Key priority.
    timeline=np.zeros(100,dtype=np.int64);timeline[4:8]=2
    assert window_labels(np.arange(16)[None],timeline).tolist()==[0]
    timeline[8:10]=2;timeline[11:13]=1
    assert window_labels(np.arange(16)[None],timeline).tolist()==[1]  # last frame=0 -> reverse lookup finds success
    timeline[14]=2
    assert window_labels(np.arange(16)[None],timeline).tolist()==[2]
    split=json.loads((output/'split_manifest.json').read_text())
    assert len(set(split['train'])&set(split['val']))==len(set(split['train'])&set(split['test']))==len(set(split['val'])&set(split['test']))==0
    for group in GROUPS:
        for item in case_summaries:
            if item['group']!=group:continue
            for splitname in ('train','val','test'):
                path=output/'cases'/group/f"n{item['n']}_step{item['step']}"/(splitname+'.npz')
                with np.load(path) as a:
                    for ridx in np.unique(a['rollout_index']):
                        row=rows[int(ridx)];positions=np.flatnonzero(a['rollout_index']==ridx);frames=a['frames'][positions]
                        assert row['rollout_id'] in split[splitname]
                        assert frames.min()>=row['annotation_start'] and frames.max()<=row['annotation_end']
                        assert (np.diff(frames)>=0).all()
                        unpadded=a['padded'][positions]==0
                        assert (np.diff(frames[unpadded])==item['step']).all()
                        regions=regions_for(row,group,item['n']);timeline=frame_labels(row,regions)
                        assert np.array_equal(a['labels'][positions],window_labels(frames,timeline))
                        for position in positions[::max(1,len(positions)//10)]:
                            assert int(a['labels'][position])==slow_label(a['frames'][position].tolist(),regions)
                        with np.load(output/row['feature_path']) as feature:
                            local=a['indices'][positions]-row['feature_offset']
                            assert np.array_equal(feature['video_frames'][local],frames)
                            assert np.array_equal(feature['ticks'][local],a['ticks'][positions])
                            assert (a['ticks'][positions]<=a['ticks'][positions,-1,None]).all()
                    assert counts({k:a[k] for k in a.files})==item['splits'][splitname]
                    checked+=len(a['labels'])
        with np.load(output/'ready'/group/'train.npz') as a:
            c=np.bincount(a['labels'],minlength=3);assert c[0]==c[1]==c[2] and (c>0).all()
            identities=[(int(r),tuple(i)) for r,i in zip(a['rollout_index'],a['indices'])]
            assert len(identities)==len(set(identities))
            negatives=a['labels']==0
            assert (a['n'][negatives]==manifest['reference_n']).all() and (a['step'][negatives]==manifest['reference_step']).all()
            for i in range(len(a['labels'])):
                row=rows[int(a['rollout_index'][i])]
                assert int(a['labels'][i])==slow_label(a['frames'][i].tolist(),regions_for(row,group,int(a['n'][i])))
        for row in rows:
            units=units_for(row['events'],group)
            if group==GROUPS[1] and {e['event_key'] for e in row['events'] if e['event_key'] in (6,8)}=={6,8}:
                assert len(units)==1 and units[0]['outcome']==1 and units[0]['merged']
                assert units[0]['start_frame']==min(e['start_frame'] for e in row['events'] if e['event_key'] in (6,8))
                assert units[0]['end_frame']==max(e['end_frame'] for e in row['events'] if e['event_key'] in (6,8))
    for path,digest in manifest['source_hashes'].items():assert sha(ROOT/path)==digest
    assert sha(output/'split_manifest.json')==manifest['split_sha256']
    assert not list(output.rglob('best.pt')) and not list(output.rglob('history.json'))
    report=dict(status='PASS',checked_grid_windows=checked,groups=2,parameter_cases=len(case_summaries),
        hard_labels=[0,1,2],fixed_length=16,posterior_half_only=True,nearest_preceding_key=True,
        mixed_align_merged_to_success=True,global_annotation_bounds=True,train_balanced_without_replacement=True,
        no_progress_augmentation=True,split_and_source_hashes_unchanged=True,training_started=False)
    dump(output/'verification.json',report);return report


def write_reports(output,manifest,case_summaries,balanced,verification):
    lines=['# 修正后 Align 在线三分类数据集（仅生成，未训练）','',
        '标签：in_progress=0、success=1、failure=2。使用此前已提取并备份的标注快照；现 annotations 目录不参与重新标注，原数据与旧实验不覆盖。','',
        '## GT 与两组对照','',
        '窗口固定 16 个采样点，仅检查第 9–16 个实际采样帧是否属于 Key 区域。没有命中为 in_progress；命中则从末帧向前找到最近的 Key 类别，因此同时命中两类且末帧为 0 时也沿后半窗口向前倒推。同一帧的 Key 区域重叠时，时间上较后的 Align 标注优先。前半窗口命中不计。没有 Gaussian、soft label 或 ignore label。','',
        '| 组别 | Key 区域 | Align 处理 |','|---|---|',
        '| symmetric_separate | [align end−n, align end+n] | 保留原 Align failure/success 区间和类别 |',
        '| asymmetric_merged_success | [align end−n, 对应 insert start+n] | 同一 rollout 同时有 failure/success 时，从最早 Align start 到最晚 Align end 合并为连续 success interval，空隙纳入；不保留其中早期 failure Key |','',
        '非对称 Key 与合并 success 是同一组的联合改动。success-only/failure-only rollout 保留原类别；没有后续 Insert 时非对称上限退回 end+n（逐条记录 no_insert_fallback）。对应 Insert 按原标注 event 顺序取终态 Align 之后第一个 7/9。Insert 不作为独立分类目标，其起点用于 Key 上限，范围内传感器数据可作为窗口上下文。','',
        '每个 rollout 从所有有效原标注（不限 6/8）的最早 start 到最晚 end 构造窗口，所有采样点在闭区间内。沿此范围连续滑动，Align 切换或标注间空隙不重置；基础组的“保留区间”指保留各个 Key 和 GT 来源，而非在标注边界截断输入。','',
        '## 采样与冻结特征','',
        f"候选网格 n={manifest['ns']}、step={manifest['steps']}，两组共 {len(case_summaries)} 个参数组合。终点滑动步长为 1 相机帧，采样点为 [e−15×step,…,e]。不足长度时重复当前有效段首帧；相机重复索引取该帧最后的有效同步 tick，目标采样帧缺失就丢弃该窗口，不用邻帧替代；真正 tactile tick 缺口会分段。",
        'F6 每个特征仍由冻结 encoder 的稠密 16 tactile ticks 生成；为保证整个输入范围不越过首个标注，起始 15 ticks 的 F6 特征重新以范围内首个有效 tick 左补齐编码。Deform 与其余 F6 特征复用已有 frozen cache。没有重建检查，没有 optimizer/backward 或模型训练。','',
        '## 按类别增强与训练集平衡','',
        f"参考采样为 n={manifest['reference_n']}、step={manifest['reference_step']}。in_progress 只来自这一套参考采样，不叠加其他 n/step。success/failure 从全部 n/step 候选中收集，按实际 16 点 feature indices 去重；同一输入只保留一个 GT，优先保留参考版本，然后按固定参数顺序保留其他版本。与 positive augmentation 输入重合的 reference in_progress 剔除，避免同一输入具有冲突标签。",
        'ready/train 从三个池中无放回抽取相同数量（取三池最小容量），得到 1:1:1。没有复制稀有类制造重复样本；多余 in_progress 和多数 positive 可下采样。完整 positive pool、参考采样与全部候选网格保留，之后可选择其他平衡策略。增强/抽样只发生在 train，val/test 保留参考 n/step 的全部窗口及自然分布；不按它们的比例决定训练采样。','',
        '## Rollout / interval 分布','',
        '| Split | Rollout | 原 Align success | 原 Align failure | 合并后的 success units | 合并后的 failure units | mixed rollout |','|---|---:|---:|---:|---:|---:|---:|']
    for split in ('train','val','test'):
        rows=[r for r in manifest['rollouts']if r['split']==split];raw=[u for r in rows for u in units_for(r['events'],GROUPS[0])];merged=[u for r in rows for u in units_for(r['events'],GROUPS[1])]
        lines.append(f"| {split} | {len(rows)} | {sum(u['outcome']==1 for u in raw)} | {sum(u['outcome']==2 for u in raw)} | {sum(u['outcome']==1 for u in merged)} | {sum(u['outcome']==2 for u in merged)} | {sum(u['merged'] for u in merged)} |")
    lines+=['','## 窗口数量：参考采样 → 正类增强池 → ready','', '| Group | 数据阶段 | Split | in_progress | success | failure | 合计 |','|---|---|---|---:|---:|---:|---:|']
    for group in GROUPS:
        for phase in ('reference','positive_pool','ready'):
            for split in (('train',)if phase=='positive_pool'else('train','val','test')):
                c=balanced[group][phase][split];counts_=c['counts']
                lines.append(f"| {group} | {phase} | {split} | {counts_['in_progress']} | {counts_['success']} | {counts_['failure']} | {c['total']} |")
    lines+=['','逐参数、split 的原始计数及比例见 [class_distribution.csv](class_distribution.csv)；增强前后计数及比例见 [ready_distribution.csv](ready_distribution.csv)。窗口高度重叠，数量不等于独立 rollout 数。两组改变了 GT，候选输入 bank 相同但类别支持量可能不同，尚没有 ACC/BA/F1 或任何新训练结果。','',
        '## 文件与复现','',
        '- `features/<rollout>.npz`：F6 1280D / Deform 2560D 的范围内冻结特征与 tick、相机帧。',
        '- `cases/<group>/n<n>_step<step>/<split>.npz`：全部候选参数数据集。',
        '- `ready/<group>/<split>.npz`：建议下一阶段使用的数据；train 平衡，val/test 自然分布。',
        '- `positive_pools/<group>/train.npz`：去重后的 success/failure 增强池。',
        '- NPZ 每条记录含 `indices[16]`（全局 feature 索引）、`frames[16]`、`ticks[16]`、scalar `labels`、`rollout_index`、`n`、`step`、`padded`。',
        '- [dataset_manifest.json](dataset_manifest.json)：时间范围、原事件、merge provenance、Key 区域、特征路径/offset、原 split、源文件哈希。',
        '- [verification.json](verification.json)：实际帧命中、倒推规则、范围、合并、标签、平衡及无重复/未训练检查。','',
        '```bash','source ./project_env.sh',
        f"bash tools/run_sharpa_tactile_ablation.sh prepare_align_online --source {manifest['source']} --backup {manifest['backup']} --output {output.relative_to(ROOT)} --device cpu --steps {' '.join(map(str,manifest['steps']))} --ns {' '.join(map(str,manifest['ns']))} --reference-step {manifest['reference_step']} --reference-n {manifest['reference_n']} --seed {manifest['seed']}",
        '```','']
    (output/'README.md').write_text('\n'.join(lines))


def prepare(args):
    source=args.source.resolve();backup=args.backup.resolve();output=args.output.resolve()
    if output.exists():raise ValueError('Use a new output directory; never overwrite existing data')
    output.mkdir(parents=True);(output/'features').mkdir();(output/'annotations').mkdir()
    data=json.loads((source/'data_manifest.json').read_text());split=json.loads((source/'split_manifest.json').read_text())
    backup_manifest=json.loads((backup/'backup_manifest.json').read_text());lookup={r['source_record']:r for r in backup_manifest['records']}
    assert data['status']=='complete' and sha(ROOT/data['intervals'])==data['signature']['interval_sha256']
    assert args.reference_n in args.ns and args.reference_step in args.steps
    assert min(args.steps)>0 and min(args.ns)>=0
    shutil.copyfile(source/'split_manifest.json',output/'split_manifest.json')
    ids=[rid for splitname in ('train','val','test')for rid in split[splitname]]
    membership={rid:s for s in ('train','val','test')for rid in split[s]}
    source_hashes={str((source/'data_manifest.json').relative_to(ROOT)):sha(source/'data_manifest.json'),
                   str((ROOT/data['intervals']).relative_to(ROOT)):sha(ROOT/data['intervals'])}
    encoders=FrozenEncoders().to(args.device);assert all(not p.requires_grad for p in encoders.parameters())
    for modality,path in [('f6',encoders.f6_path),('deform',encoders.deform_path)]:
        assert sha(path)==data['signature'][modality+'_sha256'];source_hashes[str(path.relative_to(ROOT))]=sha(path)
    torch.set_num_threads(2);rows=[];banks={};offset=0;recomputed=0
    for rid in ids:
        events=[e for e in data['source_intervals']if e['rollout_id']==rid];source_record=events[0]['source_record']
        annotation=backup/lookup[source_record]['backup_record'];assert sha(annotation)==lookup[source_record]['sha256']
        source_hashes[str(annotation.relative_to(ROOT))]=sha(annotation);shutil.copy2(annotation,output/'annotations'/annotation.name)
        raw=json.loads(annotation.read_text());all_events=raw.get('failure_events') or [raw]
        bounds=[(int(e['causal_onset_frame']),int(e['observable_onset_frame']))for e in all_events if e.get('causal_onset_frame') is not None and e.get('observable_onset_frame') is not None]
        start=min(s for s,e in bounds);end=max(e for s,e in bounds);record=data['records'][rid]
        for event in events:
            original=all_events[event['event_index']]
            assert original['causal_onset_frame']==event['start_frame'] and original['observable_onset_frame']==event['end_frame']
        feature_source=source/'features'/(rid+'.npz');source_hashes[str(feature_source.relative_to(ROOT))]=sha(feature_source)
        with np.load(feature_source)as cached:
            selected=np.flatnonzero((cached['video_frames']>=start)&(cached['video_frames']<=end))
            assert len(selected)
            cache={k:cached[k][selected].copy()for k in ('f6','deform','ticks','video_frames')}
        first=int(cache['ticks'][0]);prefix=np.flatnonzero(cache['ticks']-first<15)
        arrays=episode_arrays(record,events,data['signature']['camera'],num_classes=3)
        for position in range(0,len(prefix),16):
            chosen=prefix[position:position+16]
            raw_f6=np.stack([arrays['f6'][np.maximum(np.arange(int(cache['ticks'][p])-15,int(cache['ticks'][p])+1),first)]for p in chosen])
            cache['f6'][chosen]=encoders.f6_features(torch.from_numpy(raw_f6).to(args.device)).cpu().numpy()
        assert np.isfinite(cache['f6']).all()and np.isfinite(cache['deform']).all()
        target=output/'features'/(rid+'.npz');save(target,cache)
        row=dict(rollout_id=rid,split=membership[rid],annotation_start=start,annotation_end=end,total_frames=record['total_frames'],
            all_annotation_intervals=[list(b)for b in bounds],events=events,feature_offset=offset,feature_rows=len(selected),
            feature_path=str(target.relative_to(output)),first_tick=first,f6_prefix_reencoded_ticks=len(prefix))
        row['units']={g:units_for(events,g)for g in GROUPS};row['key_regions']={g:{str(n):regions_for(row,g,n)for n in args.ns}for g in GROUPS}
        row['merge_gap_frames']=0
        if any(u['merged']for u in row['units'][GROUPS[1]]):
            u=row['units'][GROUPS[1]][0];covered=set()
            for e in events:
                if e['event_key']in(6,8):covered.update(range(e['start_frame'],e['end_frame']+1))
            row['merge_gap_frames']=u['end_frame']-u['start_frame']+1-len(covered)
        rows.append(row);recomputed+=len(prefix)
        for step in args.steps:banks[(len(rows)-1,step)]=candidate_windows(cache,row,step)
        offset+=len(selected);print('ONLINE_DATA_FEATURES',len(rows),len(ids),rid,'prefix',len(prefix),flush=True)
    del encoders
    manifest=dict(status='preparing',source=str(source.relative_to(ROOT)),backup=str(backup.relative_to(ROOT)),
        created_at=datetime.datetime.now().astimezone().isoformat(),steps=args.steps,ns=args.ns,
        reference_n=args.reference_n,reference_step=args.reference_step,seed=args.seed,
        class_mapping=list(CLASSES),window=16,posterior_indices=list(range(8,16)),
        split_sha256=sha(source/'split_manifest.json'),source_hashes=source_hashes,rollouts=rows,
        feature_rows=offset,f6_prefix_reencoded_ticks=recomputed,encoder_finetuning=False,training_started=False)
    summaries=[];distribution=[];cases={}
    for group in GROUPS:
        for n in args.ns:
            for step in args.steps:
                summary=dict(group=group,n=n,step=step,splits={})
                for splitname in ('train','val','test'):
                    parts=[]
                    for ridx,row in enumerate(rows):
                        if row['split']!=splitname:continue
                        bank=banks[(ridx,step)];labels=window_labels(bank['frames'],frame_labels(row,regions_for(row,group,n)))
                        local=bank['indices'];parts.append(dict(indices=local+row['feature_offset'],frames=bank['frames'],
                            ticks=np.load(output/row['feature_path'])['ticks'][local],labels=labels,
                            rollout_index=np.full(len(labels),ridx,dtype=np.int64),n=np.full(len(labels),n,dtype=np.int64),
                            step=np.full(len(labels),step,dtype=np.int64),padded=bank['padded']))
                    case=concat(parts);cases[(group,n,step,splitname)]=case
                    save(output/'cases'/group/f'n{n}_step{step}'/(splitname+'.npz'),case)
                    c=counts(case);summary['splits'][splitname]=c
                    distribution.append(dict(group=group,n=n,step=step,split=splitname,**c['counts'],total=c['total'],
                        **{k+'_percent':v for k,v in c['percentages'].items()}))
                summaries.append(summary)
    rng=np.random.default_rng(args.seed);balanced={};ready_rows=[]
    preferred_ns=[args.reference_n]+[n for n in sorted(args.ns,reverse=True)if n!=args.reference_n]
    preferred_steps=[args.reference_step]+[s for s in args.steps if s!=args.reference_step]
    for group in GROUPS:
        seen=set();pool_parts=[];conflicts=0
        for n in preferred_ns:
            for step in preferred_steps:
                case=cases[(group,n,step,'train')];selected=[]
                for i in np.flatnonzero(case['labels']!=0):
                    identity=(int(case['rollout_index'][i]),tuple(case['indices'][i]))
                    if identity in seen:continue
                    seen.add(identity);selected.append(i)
                if selected:pool_parts.append(choose(case,np.array(selected,dtype=np.int64)))
        pool=concat(pool_parts);save(output/'positive_pools'/group/'train.npz',pool)
        reference=cases[(group,args.reference_n,args.reference_step,'train')]
        negative_positions=[int(i)for i in np.flatnonzero(reference['labels']==0)if (int(reference['rollout_index'][i]),tuple(reference['indices'][i]))not in seen]
        negative=choose(reference,np.array(negative_positions,dtype=np.int64))
        pc=np.bincount(pool['labels'],minlength=3);target=min(len(negative['labels']),int(pc[1]),int(pc[2]));assert target>0
        choices=[choose(negative,rng.choice(len(negative['labels']),target,replace=False))]
        for cls in (1,2):choices.append(choose(pool,rng.choice(np.flatnonzero(pool['labels']==cls),target,replace=False)))
        ready_train=concat(choices);order=rng.permutation(len(ready_train['labels']));ready_train=choose(ready_train,order)
        balanced[group]=dict(reference={},positive_pool={'train':counts(pool)},ready={},
            negative_conflict_removed=int((reference['labels']==0).sum())-len(negative_positions),target_per_class=target)
        for splitname in ('train','val','test'):
            ref=cases[(group,args.reference_n,args.reference_step,splitname)];ready=ready_train if splitname=='train'else ref
            save(output/'ready'/group/(splitname+'.npz'),ready)
            balanced[group]['reference'][splitname]=counts(ref);balanced[group]['ready'][splitname]=counts(ready)
            for phase,c in [('reference',counts(ref)),('ready',counts(ready))]:
                ready_rows.append(dict(group=group,phase=phase,split=splitname,**c['counts'],total=c['total'],
                    **{k+'_percent':v for k,v in c['percentages'].items()}))
        c=counts(pool);ready_rows.append(dict(group=group,phase='positive_pool',split='train',**c['counts'],total=c['total'],
            **{k+'_percent':v for k,v in c['percentages'].items()}))
    manifest['status']='complete';manifest['balance']=balanced
    dump(output/'dataset_manifest.json',manifest);dump(output/'parameter_grid.json',summaries)
    csv_counts(output/'class_distribution.csv',distribution);csv_counts(output/'ready_distribution.csv',ready_rows)
    verification=verify(output,manifest,summaries,balanced)
    write_reports(output,manifest,summaries,balanced,verification)
    dump(output/'generation_record.json',dict(status='complete',training_started=False,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        code_sha256=sha(Path(__file__)),environment='repos/ProcVLM/.venv',device=args.device,
        command=' '.join(__import__('sys').argv)))
    print('ONLINE_DATASET_COMPLETE',json.dumps(balanced),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--backup',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--steps',nargs='+',type=int,default=[1,3,5,8,12])
    p.add_argument('--ns',nargs='+',type=int,default=[0,2,5,10]);p.add_argument('--reference-step',type=int,default=3)
    p.add_argument('--reference-n',type=int,default=5);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();prepare(a)


if __name__=='__main__':main()
