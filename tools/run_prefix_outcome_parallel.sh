#!/usr/bin/env bash
set -euo pipefail

TACTILE_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$TACTILE_PROJECT/../LF3R/project_env.sh"
cd "$TACTILE_PROJECT"
export PYTHONPATH="$TACTILE_PROJECT/tools${PYTHONPATH:+:$PYTHONPATH}"
PREFIX_PYTHON="$PROJECT_ROOT/repos/ProcVLM/.venv/bin/python"
PREFIX_OUTPUT="${1:-outputs/sharpa_multimodal_prefix_outcome/20261009_223000}"
PREFIX_TAG="$(basename -- "$PREFIX_OUTPUT")"

if [[ ! -f "$PREFIX_OUTPUT/dataset_manifest.json" ]]; then
    "$PREFIX_PYTHON" -u -m sharpa_tactile.prefix_outcome_data prepare --output "$PREFIX_OUTPUT"
fi
"$PREFIX_PYTHON" -m sharpa_tactile.prefix_outcome_report configure --output "$PREFIX_OUTPUT"

"$PREFIX_PYTHON" -u -m sharpa_tactile.prefix_outcome_data extract --output "$PREFIX_OUTPUT" --modality tactile --device cuda:0 >"logs/prefix_tactile_${PREFIX_TAG}.log" 2>&1 &
PREFIX_TACTILE_PID=$!
"$PREFIX_PYTHON" -u -m sharpa_tactile.prefix_outcome_data extract --output "$PREFIX_OUTPUT" --modality rgb --device cuda:2 >"logs/prefix_rgb_${PREFIX_TAG}.log" 2>&1 &
PREFIX_RGB_PID=$!
wait "$PREFIX_TACTILE_PID"
echo "TACTILE_FEATURES_COMPLETE"

"$PREFIX_PYTHON" -u -m sharpa_tactile.prefix_outcome_train --output "$PREFIX_OUTPUT" --model tactile --device cuda:0 >"logs/prefix_train_tactile_${PREFIX_TAG}.log" 2>&1 &
PREFIX_TRAIN_T_PID=$!
"$PREFIX_PYTHON" -u -m sharpa_tactile.prefix_outcome_train --output "$PREFIX_OUTPUT" --model tactile_pose --device cuda:1 >"logs/prefix_train_tactile_pose_${PREFIX_TAG}.log" 2>&1 &
PREFIX_TRAIN_TP_PID=$!
wait "$PREFIX_RGB_PID"
echo "RGB_FEATURES_COMPLETE"

"$PREFIX_PYTHON" -u -m sharpa_tactile.prefix_outcome_train --output "$PREFIX_OUTPUT" --model rgb_pose --device cuda:2 >"logs/prefix_train_rgb_pose_${PREFIX_TAG}.log" 2>&1 &
PREFIX_TRAIN_RP_PID=$!
wait "$PREFIX_TRAIN_T_PID"
"$PREFIX_PYTHON" -u -m sharpa_tactile.prefix_outcome_train --output "$PREFIX_OUTPUT" --model tactile_rgb --device cuda:0 >"logs/prefix_train_tactile_rgb_${PREFIX_TAG}.log" 2>&1 &
PREFIX_TRAIN_TR_PID=$!
wait "$PREFIX_TRAIN_TP_PID"
"$PREFIX_PYTHON" -u -m sharpa_tactile.prefix_outcome_train --output "$PREFIX_OUTPUT" --model tactile_rgb_pose --device cuda:1 >"logs/prefix_train_tactile_rgb_pose_${PREFIX_TAG}.log" 2>&1 &
PREFIX_TRAIN_TRP_PID=$!
wait "$PREFIX_TRAIN_RP_PID"
wait "$PREFIX_TRAIN_TR_PID"
wait "$PREFIX_TRAIN_TRP_PID"
"$PREFIX_PYTHON" -u -m sharpa_tactile.prefix_outcome_report report --output "$PREFIX_OUTPUT"
echo "PREFIX_OUTCOME_COMPLETE"
