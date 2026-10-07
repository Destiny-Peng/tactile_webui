# Migrated tactile tools

All 82 LF3R tactile source files were moved with `mv` into this checkout. This added 27 missing SHARPA modules plus four launchers and the canonical annotation regression test. Existing standalone versions remain active; the 50 overlapping LF3R originals are preserved in `.migration_archive/20261007_204527/lf3r_tools` with their original SHA-256 hashes. The migration log is `logs/tools_migration/20261007_204527.json`. Unrelated LF3R tools were retained in LF3R.

The active package is `tools/sharpa_tactile`; media export, tactile plotting, encoder extraction, `run_trex.sh`, `run_sharpa_tactile_ablation.sh` and the early-warning/value-trend launchers live directly under `tools`.

`ROOT` now means this standalone checkout. Outputs, checkpoints and code use local paths. `DATA_ROOT` follows the existing manifest link for original sensor data; `TACTILE_DATA_ROOT` can override it. `project_path` resolves original raw paths through the existing dataset links and translates historical absolute LF3R output paths to the moved local outputs. External raw sources are recorded as absolute paths when needed. No dataset payload was copied or moved during this tool migration.

The shell launchers do not source LF3R/project_env.sh. `run_trex.sh` uses the existing ProcVLM Python environment found alongside the linked T-Rex repository, or python3 if unavailable; set `TREX_PYTHON` to explicitly choose an interpreter. No environments were installed or training started.

From the tactile_webui root:

```bash
bash tools/run_sharpa_tactile_ablation.sh --list
bash tools/run_sharpa_tactile_ablation.sh canonical_annotations --validate-only
bash tools/run_sharpa_tactile_ablation.sh early_warning --help
PYTHONPATH=tools bash tools/run_trex.sh python -m unittest discover -s tools/tests -v
```

The preserved generated LF3R canonical copy was already absent during verification. Its 130 original paths/hashes remain in `historical_canonical_artifacts` in the local canonical provenance file. Actual annotation source hashes and current local output hashes are still strictly validated.

Validation: ten annotation/path regression tests PASS; one real-rollout CPU preparation smoke PASS (697 ticks, four independent label channels); canonical and early-warning CLI startup PASS; internal module dependency audit, Python compilation and shell syntax PASS. Full Torch model initialization and training were not run.
