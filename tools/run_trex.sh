#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
TREX_REPO="${TREX_REPO:-${PROJECT_ROOT}/repos/T-Rex}"
if [[ ! -d "${TREX_REPO}" ]]; then
    echo "T-Rex repo not found: ${TREX_REPO}" >&2
    echo "Set TREX_REPO or place T-Rex under repos/T-Rex." >&2
    exit 2
fi

if [[ -z "${TREX_PYTHON:-}" ]]; then
    TREX_SOURCE_ROOT="$(cd -- "${TREX_REPO}" && pwd -P)"
    TREX_ENV_PYTHON="${TREX_SOURCE_ROOT}/../ProcVLM/.venv/bin/python"
    if [[ -x "$TREX_ENV_PYTHON" ]]; then
        TREX_PYTHON="$TREX_ENV_PYTHON"
    else
        TREX_PYTHON="python3"
    fi
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
