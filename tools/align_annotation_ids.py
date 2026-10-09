#!/usr/bin/env python3
"""Align live annotation IDs to reindexed recorder episodes; preserve history."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def encode(value):
    return json.dumps(value,indent=2,ensure_ascii=False)+'\n'


def atomic(path,text):
    temp=path.with_name(path.name+'.id_alignment.tmp')
    with temp.open('x') as stream:stream.write(text)
    temp.replace(path)


def run(args):
    manifest=args.manifest.resolve()
    records=jsonl(manifest)
    assert len({r['id'] for r in records})==len(records)
    episodes={}
    for record in records:
        episode=Path(record['source_database_path']).parent.name
        if episode in episodes:raise ValueError(f'Ambiguous episode {episode}')
        episodes[episode]=record
    mapping={};destinations={};plan=[];source_hashes={};outcome_diffs=[]
    groups=(('annotations/failrecovery/records','*.tactile.json'),
            ('annotations/tactile_canonical/v1/records','*.json'))
    id_pattern=re.compile(r'realrobot-failrecovery-(.+)-[0-9a-f]{10}')
    for directory,pattern in groups:
        for path in sorted((ROOT/directory).glob(pattern)):
            value=json.loads(path.read_text());old=value['rollout_id']
            match=id_pattern.fullmatch(old)
            if not match:raise ValueError(f'Unexpected source ID {old}')
            episode=match.group(1)
            if episode not in episodes:raise ValueError(f'No manifest episode {old}')
            record=episodes[episode];new=record['id']
            if value.get('total_frames')!=record['total_frames']:
                raise ValueError(f'Frame count mismatch {old}')
            if new in destinations and destinations[new]!=old:
                raise ValueError(f'Multiple old IDs map to {new}')
            destinations[new]=old;mapping[old]=new
            for field in ('intervals','tactile_intervals'):
                for event in value.get(field,[]):
                    assert event['rollout_id']==old
                    assert 0<=event['start_frame']<=event['end_frame']<record['total_frames']
            if directory==groups[0][0] and value.get('ground_truth_outcome')!=record['ground_truth_outcome']:
                outcome_diffs.append(dict(old_id=old,new_id=new,
                    annotation_outcome=value.get('ground_truth_outcome'),manifest_outcome=record['ground_truth_outcome']))
            target=path.with_name(path.name.replace(old,new,1))
            if target.exists():raise FileExistsError(target)
            plan.append((path,target,value));source_hashes[str(path.relative_to(ROOT))]=digest(path)
    # Historical provenance.source and migration audits stay unchanged. Only
    # identity fields and operational references to live canonical records move.
    def rewrite(value,key=None):
        if isinstance(value,dict):return {k:rewrite(v,k) for k,v in value.items()}
        if isinstance(value,list):return [rewrite(v,key) for v in value]
        if isinstance(value,str):
            if key=='rollout_id':return mapping[value]
            if key=='event_id':
                for old,new in mapping.items():
                    if value.startswith(old+':'):return new+value[len(old):]
            if key in ('source_record','migration_source'):
                for old,new in mapping.items():
                    if value.endswith('/'+old+'.json') and 'annotations/tactile_canonical/v1/records/' in value:
                        return value[:-len(old+'.json')]+new+'.json'
        return value
    index=ROOT/'annotations/tactile_canonical/v1/intervals.jsonl'
    original_rows=jsonl(index)
    rewritten_rows=[rewrite(row) for row in original_rows]
    source_hashes[str(index.relative_to(ROOT))]=digest(index)
    events_before=collections.Counter((r['rollout_id'],r['event_key'],r['start_frame'],r['end_frame']) for r in original_rows)
    events_after=collections.Counter((r['rollout_id'],r['event_key'],r['start_frame'],r['end_frame']) for r in rewritten_rows)
    assert events_after==collections.Counter({(mapping[r],k,a,b):n for (r,k,a,b),n in events_before.items()})
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    backup=output/'original_annotations_backup'
    for path,_,_ in plan:
        target=backup/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    shutil.copy2(index,backup/index.relative_to(ROOT))
    shutil.copy2(manifest,output/'rebuilt_manifest_snapshot.jsonl')
    (output/'source_hashes.json').write_text(encode(source_hashes))
    (output/'id_mapping.json').write_text(encode(mapping))
    with (output/'id_mapping.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['old_id','new_id']);writer.writeheader()
        writer.writerows(dict(old_id=k,new_id=v) for k,v in sorted(mapping.items()))
    (output/'outcome_differences.json').write_text(encode(outcome_diffs))
    # Check every source and prepare reviewable migrated documents before writes.
    stage=output/'aligned_annotations'
    for path,target,value in plan:
        staged=stage/target.relative_to(ROOT);staged.parent.mkdir(parents=True,exist_ok=True)
        staged.write_text(encode(rewrite(value)))
    new_index=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rewritten_rows)
    staged_index=stage/index.relative_to(ROOT);staged_index.write_text(new_index)
    assert digest(manifest)==digest(output/'rebuilt_manifest_snapshot.jsonl')
    for relative,expected in source_hashes.items():
        if digest(ROOT/relative)!=expected:raise RuntimeError(f'Concurrent annotation edit: {relative}')
        assert digest(backup/relative)==expected
    if args.apply:
        for path,target,_ in plan:atomic(target,(stage/target.relative_to(ROOT)).read_text())
        atomic(index,new_index)
        for path,_,_ in plan:path.unlink()  # exact bytes preserved in verified backup
    result=dict(status='applied' if args.apply else 'prepared',manifest=str(manifest.relative_to(ROOT)),
        manifest_sha256=digest(manifest),manifest_rollouts=len(records),mapped_rollouts=len(mapping),
        migrated_record_files=len(plan),canonical_intervals=len(original_rows),
        annotated_rollouts=len({r['rollout_id'] for r in original_rows}),
        empty_annotation_rollouts=len(mapping)-len({r['rollout_id'] for r in original_rows}),
        unmatched=0,ambiguous=0,frame_count_mismatches=0,labels_and_bounds_preserved=True,
        outcomes_preserved=True,manifest_outcome_differences=len(outcome_diffs),
        manifest_unknown_among_annotated=sum(r['manifest_outcome']=='unknown' for r in outcome_diffs),
        historical_provenance_preserved=True,backup_sha_verified=True,
        timestamp=datetime.datetime.now().astimezone().isoformat())
    if args.apply:
        for _,target,value in plan:
            migrated=json.loads(target.read_text());assert migrated==rewrite(value)
            assert migrated['rollout_id'] in destinations
        assert jsonl(index)==rewritten_rows
    (output/'alignment_report.json').write_text(encode(result))
    (output/'README.md').write_text('# Annotation ID alignment\n\n'+encode(result)+'\n'
       'Aligned live tactile sidecars, canonical per-rollout records and canonical intervals.jsonl to the rebuilt raw_failrecovery manifest. Episode directory basename uniquely identifies each recording; all frame counts agree. Only identity fields/event IDs and live canonical record references changed. Labels, bounds, outcomes and historical provenance preserved. Historical experiment outputs and LF3R source annotations unchanged.\n\n'
       'original_annotations_backup/ contains SHA-verified original bytes; aligned_annotations/ contains migrated documents. id_mapping.csv/json records the correspondence. outcome_differences.json lists pre-existing manifest/annotation outcome differences; no outcomes changed. New manifest unknown outcomes can prevent saving failure-only label3/4 in the WebUI; that is separate from ID alignment.\n')
    print(encode(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'datasets/raw_failrecovery/manifest.jsonl')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--apply',action='store_true')
    run(parser.parse_args())
