#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
TREX_REPO="${TREX_REPO:-${PROJECT_ROOT}/repos/T-Rex}"
TREX_PYTHON="${TREX_PYTHON:-python}"

if [[ ! -d "${TREX_REPO}" ]]; then
    echo "T-Rex repo not found: ${TREX_REPO}" >&2
    echo "Set TREX_REPO or place T-Rex under repos/T-Rex." >&2
    exit 2
fi

export PROJECT_ROOT
export PYTHONPATH="${TREX_REPO}:${TREX_REPO}/dataset_quickstart/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-offline}"

case "${1:-}" in
    train|infer)
        TREX_COMMAND="$1"
        shift
        [[ "$TREX_COMMAND" == "infer" ]] && TREX_COMMAND=test
        exec "$TREX_PYTHON" "${TREX_REPO}/scripts/${TREX_COMMAND}.py" "$@"
        ;;
    python)
        shift
        exec "$TREX_PYTHON" "$@"
        ;;
    *)
        echo "Usage: bash tools/run_trex.sh {train|infer|python} [arguments...]" >&2
        exit 2
        ;;
esac
