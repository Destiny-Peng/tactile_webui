#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PROJECT_ROOT
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/tools"
RUN_DIR="${1:?output directory required}"
while [[ ! -f "$RUN_DIR/shared_complete.json" ]]; do sleep 3; done
mkdir -p "$PROJECT_ROOT/logs/sharpa_value_trend"
TRAIN_PIDS=()
for PAIR in 0:0 8:1 15:2 30:0 45:1; do
    HORIZON="${PAIR%:*}"
    DEVICE_INDEX="${PAIR#*:}"
    bash "$PROJECT_ROOT/tools/run_trex.sh" python -u -m sharpa_tactile.value_trend risk --output "$RUN_DIR" --horizon "$HORIZON" --device "cuda:$DEVICE_INDEX" > "$PROJECT_ROOT/logs/sharpa_value_trend/$(basename "$RUN_DIR")_H${HORIZON}.log" 2>&1 &
    TRAIN_PIDS+=("$!")
done
FAILED=0
for TRAIN_PID in "${TRAIN_PIDS[@]}"; do wait "$TRAIN_PID" || FAILED=1; done
exit "$FAILED"
