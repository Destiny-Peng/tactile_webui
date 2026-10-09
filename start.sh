#!/usr/bin/env bash
# Launch the standalone tactile WebUI from any working directory.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ -n "${WEBUI_PYTHON:-}" ]]; then
  PYTHON_BIN="$WEBUI_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  printf 'Python interpreter not found. Set WEBUI_PYTHON=/path/to/python.\n' >&2
  exit 1
fi

cd "$ROOT"
exec "$PYTHON_BIN" "$ROOT/webui/server.py" --root "$ROOT" "$@"
