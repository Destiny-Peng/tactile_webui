#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${1:-/mnt/hdd/pyr/LF3R}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
note() { printf '[migrate] %s\n' "$*"; }

command -v rsync >/dev/null 2>&1 || fail "rsync is required"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"

SOURCE_ROOT="$(cd "$SOURCE_ROOT" 2>/dev/null && pwd)" || fail "source root not found: $SOURCE_ROOT"
[[ "$SOURCE_ROOT" != "$TARGET_ROOT" ]] || fail "source and target repositories must be different"

LEGACY_DATA="$SOURCE_ROOT/datasets/lf3r_failure_rollouts/v1"
FLAT_DATA="$SOURCE_ROOT/datasets/lf3r_failure_rollouts"

if [[ -f "$LEGACY_DATA/failrecovery_manifest.jsonl" ]]; then
  SOURCE_DATA="$LEGACY_DATA"
  SOURCE_PREFIX="datasets/lf3r_failure_rollouts/v1/"
elif [[ -f "$FLAT_DATA/failrecovery_manifest.jsonl" ]]; then
  SOURCE_DATA="$FLAT_DATA"
  SOURCE_PREFIX="datasets/lf3r_failure_rollouts/"
else
  fail "failrecovery_manifest.jsonl not found under $SOURCE_ROOT/datasets/lf3r_failure_rollouts"
fi

TARGET_DATA="$TARGET_ROOT/datasets/lf3r_failure_rollouts"
mkdir -p "$TARGET_DATA"

note "copying fail-recovery export"
[[ -d "$SOURCE_DATA/failrecovery" ]] || fail "missing exported episode directory: $SOURCE_DATA/failrecovery"
rsync -a --info=stats2 "$SOURCE_DATA/failrecovery/" "$TARGET_DATA/failrecovery/"

note "rewriting manifest into flat standalone layout"
python3 - "$SOURCE_DATA/failrecovery_manifest.jsonl" "$TARGET_DATA/failrecovery_manifest.jsonl" "$SOURCE_PREFIX" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
source_prefix = sys.argv[3]
flat_prefix = "datasets/lf3r_failure_rollouts/"

def rewrite(value):
    if isinstance(value, str):
        if value.startswith(source_prefix):
            return flat_prefix + value[len(source_prefix):]
        return value
    if isinstance(value, list):
        return [rewrite(item) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item) for key, item in value.items()}
    return value

rows = []
for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    row = json.loads(line)
    if not isinstance(row, dict):
        raise SystemExit(f"manifest line {line_number} is not an object")
    rows.append(rewrite(row))

target.write_text(
    "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
    encoding="utf-8",
)
print(f"rewrote {len(rows)} manifest rows")
PY

note "flattening annotations already present in tactile_webui"
if [[ -d "$TARGET_ROOT/annotations/failure_annotations/v1" ]]; then
  mkdir -p \
    "$TARGET_ROOT/annotations/failure_annotations/records" \
    "$TARGET_ROOT/annotations/failure_annotations/events"
  if [[ -d "$TARGET_ROOT/annotations/failure_annotations/v1/records" ]]; then
    rsync -a --ignore-existing \
      "$TARGET_ROOT/annotations/failure_annotations/v1/records/" \
      "$TARGET_ROOT/annotations/failure_annotations/records/"
  fi
  if [[ -d "$TARGET_ROOT/annotations/failure_annotations/v1/events" ]]; then
    rsync -a --ignore-existing \
      "$TARGET_ROOT/annotations/failure_annotations/v1/events/" \
      "$TARGET_ROOT/annotations/failure_annotations/events/"
  fi
  rm -rf "$TARGET_ROOT/annotations/failure_annotations/v1"
fi

note "migrating annotation records and save-event history"
SOURCE_ANN=""
if [[ -d "$SOURCE_ROOT/annotations/failure_annotations/v1" ]]; then
  SOURCE_ANN="$SOURCE_ROOT/annotations/failure_annotations/v1"
elif [[ -d "$SOURCE_ROOT/annotations/failure_annotations" ]]; then
  SOURCE_ANN="$SOURCE_ROOT/annotations/failure_annotations"
fi

if [[ -n "$SOURCE_ANN" ]]; then
  mkdir -p \
    "$TARGET_ROOT/annotations/failure_annotations/records" \
    "$TARGET_ROOT/annotations/failure_annotations/events"
  if [[ -d "$SOURCE_ANN/records" ]]; then
    rsync -a --ignore-existing \
      "$SOURCE_ANN/records/" \
      "$TARGET_ROOT/annotations/failure_annotations/records/"
  fi
  if [[ -d "$SOURCE_ANN/events" ]]; then
    rsync -a --ignore-existing \
      "$SOURCE_ANN/events/" \
      "$TARGET_ROOT/annotations/failure_annotations/events/"
  fi
else
  note "no annotation directory found; skipping"
fi

note "migrating USB interval seeds"
if [[ -d "$SOURCE_ROOT/outputs/usb_event_intervals" ]]; then
  mkdir -p "$TARGET_ROOT/outputs/usb_event_intervals"
  rsync -a --ignore-existing \
    "$SOURCE_ROOT/outputs/usb_event_intervals/" \
    "$TARGET_ROOT/outputs/usb_event_intervals/"
else
  note "no outputs/usb_event_intervals found; skipping"
fi

note "migrating frozen tactile encoders"
if [[ -d "$SOURCE_ROOT/checkpoints/T-Rex/encoders" ]]; then
  mkdir -p "$TARGET_ROOT/checkpoints/T-Rex/encoders"
  rsync -a --ignore-existing \
    "$SOURCE_ROOT/checkpoints/T-Rex/encoders/" \
    "$TARGET_ROOT/checkpoints/T-Rex/encoders/"
else
  note "no checkpoints/T-Rex/encoders found; skipping"
fi

note "migrating external tactile repositories"
for repo in T-Rex sharpawave-deform-encoder; do
  if [[ -d "$SOURCE_ROOT/repos/$repo" ]]; then
    mkdir -p "$TARGET_ROOT/repos/$repo"
    rsync -a \
      --exclude '.venv' \
      --exclude '__pycache__' \
      "$SOURCE_ROOT/repos/$repo/" \
      "$TARGET_ROOT/repos/$repo/"
  else
    note "no repos/$repo found; skipping"
  fi
done

note "updating standalone settings to use migrated local data"
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
data["annotations_path"] = "annotations/failure_annotations/records"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

note "verifying migrated manifest paths"
python3 - "$TARGET_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = root / "datasets/lf3r_failure_rollouts/failrecovery_manifest.jsonl"
legacy_prefix = "datasets/lf3r_failure_rollouts/v1/"
text = manifest.read_text(encoding="utf-8")

if legacy_prefix in text:
    raise SystemExit("legacy /v1/ path remains in migrated manifest")

missing = []
rows = 0
for line in text.splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    rows += 1

    for value in (row.get("camera_video_paths") or {}).values():
        if value and not (root / value).is_file():
            missing.append(value)

    for key in ("synchronized_frames_path", "tactile_events_path"):
        value = row.get(key)
        if value and not (root / value).is_file():
            missing.append(value)

    for entry in (row.get("tactile_stream_paths") or {}).values():
        if not isinstance(entry, dict):
            continue
        for value in entry.values():
            if value and not (root / value).is_file():
                missing.append(value)

if missing:
    preview = "\n  ".join(missing[:20])
    raise SystemExit(f"{len(missing)} referenced files are missing:\n  {preview}")

print(f"verified {rows} rollouts; every WebUI runtime file exists")
PY

note "migration complete"
printf '\nStandalone layout:\n'
printf '  datasets/lf3r_failure_rollouts/failrecovery_manifest.jsonl\n'
printf '  datasets/lf3r_failure_rollouts/failrecovery/\n'
printf '  annotations/failure_annotations/records/\n'
printf '  annotations/failure_annotations/events/\n'
printf '  outputs/usb_event_intervals/\n'
printf '  checkpoints/T-Rex/encoders/\n'
printf '  repos/T-Rex/\n'
printf '  repos/sharpawave-deform-encoder/\n'
printf '\nStart with:\n'
printf '  python webui/server.py --root . --host 127.0.0.1 --port 8765\n'
