#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}/tools${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: bash tools/run_sharpa_tactile_ablation.sh <module> [arguments...]"
    echo "Use --list to list available sharpa_tactile modules."
    exit 0
fi
if [[ "${1:-}" == "--list" ]]; then
    for MODULE_PATH in "${PROJECT_ROOT}"/tools/sharpa_tactile/*.py; do
        MODULE_NAME="${MODULE_PATH##*/}"
        [[ "$MODULE_NAME" == "__init__.py" ]] || echo "${MODULE_NAME%.py}"
    done
    exit 0
fi
SHARPA_COMMAND="${1:-}"
if [[ ! "$SHARPA_COMMAND" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]] || [[ ! -f "${PROJECT_ROOT}/tools/sharpa_tactile/${SHARPA_COMMAND}.py" ]]; then
    echo "Expected an existing sharpa_tactile module; use --list." >&2
    exit 2
fi
shift
exec bash "${PROJECT_ROOT}/tools/run_trex.sh" python -m "sharpa_tactile.${SHARPA_COMMAND}" "$@"
