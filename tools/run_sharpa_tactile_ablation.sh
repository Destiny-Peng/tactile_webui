#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}/tools${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

case "${1:-}" in
    normalize_align_online|report_align_online|train_align_online|prepare_align_online|align_windows_data|align_windows|verify_align_windows|report_align_windows|publish_align_windows|prepare|train|verify|verify_online|report|prepare_three|train_three|verify_three|report_three|weight_seed_ablation|prepare_intervals|train_intervals|verify_intervals|report_intervals|causal_prefix|gaussian_online|merged_online|merged_interval_binary|align_gaussian_probability_curves|report_causal_prefix|report_gaussian_online|report_merged_online|report_merged_interval_binary|audit_gaussian_step_augmentation)
        SHARPA_COMMAND="$1"
        shift
        exec bash "${PROJECT_ROOT}/tools/run_trex.sh" python -m "sharpa_tactile.${SHARPA_COMMAND}" "$@"
        ;;
    *)
        echo "Usage: bash tools/run_sharpa_tactile_ablation.sh <sharpa_tactile module> [arguments...]" >&2
        echo "Known commands: prepare train verify report prepare_three train_three verify_three report_three prepare_intervals train_intervals verify_intervals report_intervals prepare_align_online train_align_online normalize_align_online report_align_online align_windows_data align_windows verify_align_windows report_align_windows publish_align_windows weight_seed_ablation causal_prefix gaussian_online merged_online merged_interval_binary align_gaussian_probability_curves audit_gaussian_step_augmentation" >&2
        exit 2
        ;;
esac
