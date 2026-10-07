#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/tools"
RUN_DIR="${1:?output required}"
mkdir -p "$PROJECT_ROOT/logs/sharpa_early_warning_ba300"
PIDS=()
for PAIR in 0:0 8:1 15:2 30:2 45:2; do
    HORIZON="${PAIR%:*}"
    DEVICE_INDEX="${PAIR#*:}"
    bash "$PROJECT_ROOT/tools/run_trex.sh" python -u -m sharpa_tactile.early_warning_epoch_metrics --horizon "$HORIZON" --output "$RUN_DIR/shards/H${HORIZON}" --device "cuda:$DEVICE_INDEX" > "$PROJECT_ROOT/logs/sharpa_early_warning_ba300/$(basename "$RUN_DIR")_H${HORIZON}.log" 2>&1 &
    PIDS+=("$!")
done
FAILED=0
for TRAIN_PID in "${PIDS[@]}"; do wait "$TRAIN_PID" || FAILED=1; done
exit "$FAILED"
