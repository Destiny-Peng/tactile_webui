#!/usr/bin/env bash
set -euo pipefail
WRIST_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$WRIST_PROJECT/../LF3R/project_env.sh"
cd "$WRIST_PROJECT"
export PYTHONPATH="$WRIST_PROJECT/tools${PYTHONPATH:+:$PYTHONPATH}"
WRIST_PYTHON="$PROJECT_ROOT/repos/ProcVLM/.venv/bin/python"
WRIST_OUTPUT="${1:-outputs/sharpa_multimodal_prefix_outcome_wrist/20261010_010000}"
WRIST_TAG="$(basename -- "$WRIST_OUTPUT")"
if [[ ! -f "$WRIST_OUTPUT/dataset_manifest.json" ]]; then
    "$WRIST_PYTHON" -u -m sharpa_tactile.prefix_outcome_wrist prepare --output "$WRIST_OUTPUT"
fi
WRIST_PIDS=()
for WRIST_GPU in 0 1 2; do
    "$WRIST_PYTHON" -u -m sharpa_tactile.prefix_outcome_data extract --output "$WRIST_OUTPUT" --modality rgb --shards 3 --shard-index "$WRIST_GPU" --device "cuda:$WRIST_GPU" >"logs/wrist_rgb_${WRIST_GPU}_${WRIST_TAG}.log" 2>&1 &
    WRIST_PIDS+=("$!")
done
for WRIST_PID in "${WRIST_PIDS[@]}"; do wait "$WRIST_PID"; done
echo WRIST_RGB_COMPLETE
WRIST_PIDS=()
WRIST_MODELS=(tactile_rgb tactile_rgb_pose rgb_pose)
for WRIST_GPU in 0 1 2; do
    WRIST_MODEL="${WRIST_MODELS[$WRIST_GPU]}"
    "$WRIST_PYTHON" -u -m sharpa_tactile.prefix_outcome_train --output "$WRIST_OUTPUT" --model "$WRIST_MODEL" --device "cuda:$WRIST_GPU" >"logs/wrist_train_${WRIST_MODEL}_${WRIST_TAG}.log" 2>&1 &
    WRIST_PIDS+=("$!")
done
for WRIST_PID in "${WRIST_PIDS[@]}"; do wait "$WRIST_PID"; done
"$WRIST_PYTHON" -u -m sharpa_tactile.prefix_outcome_wrist publish --output "$WRIST_OUTPUT"
echo WRIST_OUTCOME_COMPLETE
