#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-$(dirname "$TARGET_ROOT")/LF3R}"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
note() { printf '[link] %s\n' "$*"; }

command -v python3 >/dev/null 2>&1 || fail "python3 is required"

SOURCE_ROOT="$(cd "$SOURCE_ROOT" 2>/dev/null && pwd)" || fail "source root not found: $SOURCE_ROOT"
[[ "$SOURCE_ROOT" != "$TARGET_ROOT" ]] || fail "source and target repositories must be different"

LEGACY_MANIFEST="$SOURCE_ROOT/datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl"
TRANSITIONAL_MANIFEST="$SOURCE_ROOT/datasets/lf3r_failure_rollouts/failrecovery_manifest.jsonl"
CANONICAL_MANIFEST="$SOURCE_ROOT/datasets/failrecovery/manifest.jsonl"

if [[ -f "$LEGACY_MANIFEST" ]]; then
  SOURCE_MANIFEST="$LEGACY_MANIFEST"
elif [[ -f "$TRANSITIONAL_MANIFEST" ]]; then
  SOURCE_MANIFEST="$TRANSITIONAL_MANIFEST"
elif [[ -f "$CANONICAL_MANIFEST" ]]; then
  SOURCE_MANIFEST="$CANONICAL_MANIFEST"
else
  fail "no failrecovery manifest found under $SOURCE_ROOT/datasets"
fi

link_path() {
  local source="$1"
  local target="$2"
  local label="$3"
  [[ -e "$source" || -L "$source" ]] || { note "$label missing; skipping"; return 0; }
  mkdir -p "$(dirname "$target")"
  if [[ -L "$target" ]]; then
    ln -sfn "$source" "$target"
    note "updated $label symlink"
  elif [[ -e "$target" ]]; then
    note "$label already exists locally; keeping it"
  else
    ln -s "$source" "$target"
    note "linked $label"
  fi
}

note "linking dataset manifest; fail-recovery payload will not be copied"
mkdir -p "$TARGET_ROOT/datasets/failrecovery"
TARGET_MANIFEST="$TARGET_ROOT/datasets/failrecovery/manifest.jsonl"
if [[ -L "$TARGET_MANIFEST" ]]; then
  ln -sfn "$SOURCE_MANIFEST" "$TARGET_MANIFEST"
elif [[ -e "$TARGET_MANIFEST" ]]; then
  note "local datasets/failrecovery/manifest.jsonl already exists; keeping local dataset"
  note "remove the local dataset first if you want to switch this checkout to symlink mode"
else
  ln -s "$SOURCE_MANIFEST" "$TARGET_MANIFEST"
fi

note "keeping annotations local and writable"
mkdir -p \
  "$TARGET_ROOT/annotations/failrecovery/records" \
  "$TARGET_ROOT/annotations/failrecovery/events"

note "linking optional large dependencies"
link_path \
  "$SOURCE_ROOT/checkpoints/T-Rex/encoders" \
  "$TARGET_ROOT/checkpoints/T-Rex/encoders" \
  "T-Rex encoders"
link_path \
  "$SOURCE_ROOT/repos/T-Rex" \
  "$TARGET_ROOT/repos/T-Rex" \
  "T-Rex repo"
link_path \
  "$SOURCE_ROOT/repos/sharpawave-deform-encoder" \
  "$TARGET_ROOT/repos/sharpawave-deform-encoder" \
  "SHARPA Wave encoder repo"

note "updating standalone settings"
python3 - "$TARGET_ROOT/.tactile_webui/settings.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = {}
if path.is_file():
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data.update(loaded)
    except Exception:
        pass

data["source_project_root"] = "."
data["annotations_path"] = "annotations/failrecovery/records"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

note "verifying symlink-aware source resolution"
python3 - "$TARGET_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from webui.server import TactileApplication

app = TactileApplication(root)
rows = app.load_rollouts(refresh=True)
if not rows:
    raise SystemExit("manifest loaded zero rollouts")

checked = 0
for row in rows:
    for value in (row.get("camera_video_paths") or {}).values():
        if value:
            path = app.source_file(value)
            if not path.is_file():
                raise SystemExit(f"missing video referenced by manifest: {value}")
            checked += 1
            break
    if checked:
        break

print(f"logical manifest entry: {root / 'datasets/failrecovery/manifest.jsonl'}")
print(f"resolved manifest: {app.manifest_path}")
print(f"resolved data root: {app.data_root}")
print(f"rollouts: {len(rows)}")
print(f"verified referenced videos: {checked}")
PY

note "link setup complete"
printf '\nLogical standalone layout:\n'
printf '  datasets/failrecovery/manifest.jsonl -> original LF3R manifest\n'
printf '  annotations/failrecovery/records/     (local writable annotations)\n'
printf '  annotations/failrecovery/events/      (local writable events)\n'
printf '\nNo fail-recovery video/tactile payload was copied.\n'
printf 'Static tactile PNGs are reused when present; packed streams remain the fallback.\n'
printf '\nStart with:\n'
printf '  python webui/server.py --root . --host 127.0.0.1 --port 8765\n'
