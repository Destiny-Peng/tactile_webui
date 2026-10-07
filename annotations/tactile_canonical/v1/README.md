# Canonical tactile annotation migration

The authority for rollout eligibility is `datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl`, field `ground_truth_outcome` (92 success / 33 failure USB rollouts). Interval labels never determine rollout outcome. Label 2 may therefore legitimately appear in an eventually successful rollout.

Inputs are the live USB annotation records, the verified 20261003 original annotation backup (needed to recover overwritten 6/7/8/9 intervals), and all three `sharpa_failure_relabel` snapshots with their verified native backups. This is a union of snapshots: distinct bounds are retained even when another revision is newer. Native backup and derived export copies both contribute provenance and raw-input counts; they do not multiply canonical intervals.

| Label | Raw source intervals before migration | After mapping, before filtering/dedup | Final rollouts | Final intervals |
|---|---:|---:|---:|---:|
| 1 (success) | 0 | 370 | 92 | 186 |
| 2 (failure) | 0 | 128 | 53 | 73 |
| 3 (dropped_object) | 157 | 157 | 24 | 33 |
| 4 (wrong_object) | 382 | 382 | 18 | 61 |

Historical raw-input counts: 6=124, 7=4, 8=187, 9=183. Mapping: 6/7→2; 8/9→1.

All 1037 input interval occurrences reconcile as 353 output intervals + 682 exact duplicate occurrences + 2 discarded success-rollout 3/4 occurrences + 0 background records. The two discarded occurrences both have label 4. There are 125 per-rollout records, 120 with intervals and 5 empty. Rollout counts per label overlap and must not be summed.

36 source event occurrences are separately audited as placeholders with both bounds null or failure types outside the historical tactile schema. For example, the old native `grasp_failure` hotkey 1 is not silently interpreted as canonical success 1. Partial bounds, unknown numeric labels, unknown outcomes and invalid frames fail migration. All actual output frames satisfy `0 <= start_frame <= end_frame < total_frames` with integer bounds.

Only `(rollout_id, mapped event_key, start_frame, end_frame)` duplicates are collapsed. Different labels or boundaries survive, including overlapping intervals. Every retained interval carries all original source paths, SHA-256 hashes, event indexes, labels and unchanged bounds. `stages` preserves Align/Insert provenance for stage experiments independently of annotation label. Original sources and old experiment outputs are unchanged.

- `intervals.jsonl`: canonical flat reader input, labels 1/2/3/4 only.
- `records/`: complete merged per-rollout annotations, authoritative outcome provenance, and total frames.
- `migration_summary.json`: before/after and per-source counts.
- `migration_audit.json`: filtered success 3/4, exact duplicates, background and excluded source events.
- `provenance.json`: all source SHA-256 hashes and output SHA-256 hashes.

Downstream `load_sources` preserves all four annotation labels and checks outcomes/bounds. Feature caches retain four independent `annotation_labels` channels; time overlaps can activate several channels. Existing success/failure probes explicitly project annotation1→target0 and annotation2→target1. They reject 3/4 as binary targets. The 3/4 probes now read this canonical input and select 3/4 as independent labels. Binary interval preparation also writes `independent_annotations.json`; no 3/4 label is folded into success/failure. Old 6/7/8/9 caches are rejected: generate new preparation outputs before training. No training was started during this migration.

Reproduce from the LF3R root (standard-library Python is sufficient for migration):

```bash
source ./project_env.sh
PYTHONPATH=tools python3 -m sharpa_tactile.canonical_annotations --output annotations/tactile_canonical/reproduced
PYTHONPATH=tools python3 -m sharpa_tactile.canonical_annotations --validate-only
PYTHONPATH=tools conda_envs/LF3R-ananlyse/bin/python -m unittest discover -s tools/tests -p test_tactile_canonical_annotations.py -v
```

The default output is `annotations/tactile_canonical/v1`; migration refuses to overwrite any existing output directory. The standalone tactile package resolves the linked manifest's LF3R dataset root; `TACTILE_DATA_ROOT` can explicitly override the data root.

Validation: one real failure-rollout CPU preparation smoke PASS (697 valid ticks; independent channel sums [0,165,5,93]); LF3R and standalone 3/4 probe imports PASS; six CPU regression/consumer tests PASS; source and output hash verification PASS; Python compilation and shell syntax PASS. Tests cover mapping, outcome filtering, exact dedup, distinct overlapping bounds, immutable sources, invalid bounds/unknown outcomes, independent four-channel labels and loading all 353 real intervals.

Verification recorded at 2026-10-07T18:43:05.283624+08:00; LF3R HEAD `57358ff76818aa3388284f28349995b62ad2a706` (working tree includes these changes).

Installed locally into tactile_webui: canonical files in annotations/tactile_canonical/v1; WebUI-native copies in annotations/failrecovery/records/*.tactile.json. LF3R original source files are unchanged.
