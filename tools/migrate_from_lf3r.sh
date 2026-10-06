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

# Accept the original dissertation layout, the first standalone migration
# layout, and the new dataset-centric layout as migration inputs.
LEGACY_DATA="$SOURCE_ROOT/datasets/lf3r_failure_rollouts/v1"
TRANSITIONAL_DATA="$SOURCE_ROOT/datasets/lf3r_failure_rollouts"
CANONICAL_DATA="$SOURCE_ROOT/datasets/failrecovery"

if [[ -f "$LEGACY_DATA/failrecovery_manifest.jsonl" ]]; then
  SOURCE_EPISODES="$LEGACY_DATA/failrecovery"
  SOURCE_MANIFEST="$LEGACY_DATA/failrecovery_manifest.jsonl"
  SOURCE_PREFIX="datasets/lf3r_failure_rollouts/v1/failrecovery/"
elif [[ -f "$TRANSITIONAL_DATA/failrecovery_manifest.jsonl" ]]; then
  SOURCE_EPISODES="$TRANSITIONAL_DATA/failrecovery"
  SOURCE_MANIFEST="$TRANSITIONAL_DATA/failrecovery_manifest.jsonl"
  SOURCE_PREFIX="datasets/lf3r_failure_rollouts/failrecovery/"
elif [[ -f "$CANONICAL_DATA/manifest.jsonl" ]]; then
  SOURCE_EPISODES="$CANONICAL_DATA"
  SOURCE_MANIFEST="$CANONICAL_DATA/manifest.jsonl"
  SOURCE_PREFIX="datasets/failrecovery/"
else
  fail "no failrecovery manifest found under $SOURCE_ROOT/datasets"
fi

TARGET_DATA="$TARGET_ROOT/datasets/failrecovery"
mkdir -p "$TARGET_DATA"

note "copying fail-recovery dataset"
[[ -d "$SOURCE_EPISODES" ]] || fail "missing episode directory: $SOURCE_EPISODES"
rsync -a --info=stats2 --exclude 'manifest.jsonl' "$SOURCE_EPISODES/" "$TARGET_DATA/"

note "rewriting dataset-local manifest"
python3 - "$SOURCE_MANIFEST" "$TARGET_DATA/manifest.jsonl" "$SOURCE_PREFIX" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
source_prefix = sys.argv[3]
target_prefix = "datasets/failrecovery/"

def rewrite(value):
    if isinstance(value, str):
        if value.startswith(source_prefix):
            return target_prefix + value[len(source_prefix):]
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

note "migrating annotations into annotations/failrecovery"
TARGET_ANN="$TARGET_ROOT/annotations/failrecovery"
mkdir -p "$TARGET_ANN/records" "$TARGET_ANN/events"

# First migrate annotations that may already exist in this standalone repo.
for local_ann in \
  "$TARGET_ROOT/annotations/failure_annotations/v1" \
  "$TARGET_ROOT/annotations/failure_annotations"; do
  if [[ -d "$local_ann/records" ]]; then
    rsync -a --ignore-existing "$local_ann/records/" "$TARGET_ANN/records/"
  fi
  if [[ -d "$local_ann/events" ]]; then
    rsync -a --ignore-existing "$local_ann/events/" "$TARGET_ANN/events/"
  fi
done

SOURCE_ANN=""
if [[ -d "$SOURCE_ROOT/annotations/failrecovery" ]]; then
  SOURCE_ANN="$SOURCE_ROOT/annotations/failrecovery"
elif [[ -d "$SOURCE_ROOT/annotations/failure_annotations/v1" ]]; then
  SOURCE_ANN="$SOURCE_ROOT/annotations/failure_annotations/v1"
elif [[ -d "$SOURCE_ROOT/annotations/failure_annotations" ]]; then
  SOURCE_ANN="$SOURCE_ROOT/annotations/failure_annotations"
fi
if [[ -n "$SOURCE_ANN" ]]; then
  [[ ! -d "$SOURCE_ANN/records" ]] || rsync -a --ignore-existing "$SOURCE_ANN/records/" "$TARGET_ANN/records/"
  [[ ! -d "$SOURCE_ANN/events" ]] || rsync -a --ignore-existing "$SOURCE_ANN/events/" "$TARGET_ANN/events/"
else
  note "no annotation directory found; skipping source annotations"
fi

note "migrating USB interval seeds"
if [[ -d "$SOURCE_ROOT/outputs/usb_event_intervals" ]]; then
  mkdir -p "$TARGET_ROOT/outputs/usb_event_intervals"
  rsync -a --ignore-existing "$SOURCE_ROOT/outputs/usb_event_intervals/" "$TARGET_ROOT/outputs/usb_event_intervals/"
else
  note "no outputs/usb_event_intervals found; skipping"
fi

note "migrating frozen tactile encoders"
if [[ -d "$SOURCE_ROOT/checkpoints/T-Rex/encoders" ]]; then
  mkdir -p "$TARGET_ROOT/checkpoints/T-Rex/encoders"
  rsync -a --ignore-existing "$SOURCE_ROOT/checkpoints/T-Rex/encoders/" "$TARGET_ROOT/checkpoints/T-Rex/encoders/"
else
  note "no checkpoints/T-Rex/encoders found; skipping"
fi

note "migrating external tactile repositories"
for repo in T-Rex sharpawave-deform-encoder; do
  if [[ -d "$SOURCE_ROOT/repos/$repo" ]]; then
    mkdir -p "$TARGET_ROOT/repos/$repo"
    rsync -a --exclude '.venv' --exclude '__pycache__' "$SOURCE_ROOT/repos/$repo/" "$TARGET_ROOT/repos/$repo/"
  else
    note "no repos/$repo found; skipping"
  fi
done

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

note "verifying dataset-local manifest paths"
python3 - "$TARGET_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = root / "datasets/failrecovery/manifest.jsonl"
text = manifest.read_text(encoding="utf-8")
for legacy in (
    "datasets/lf3r_failure_rollouts/v1/",
    "datasets/lf3r_failure_rollouts/",
):
    if legacy in text:
        raise SystemExit(f"legacy path remains in migrated manifest: {legacy}")

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

# Remove only the obsolete standalone layout after the canonical copy verifies.
rm -rf "$TARGET_ROOT/datasets/lf3r_failure_rollouts"
rm -rf "$TARGET_ROOT/annotations/failure_annotations"

note "migration complete"
printf '\nStandalone layout:\n'
printf '  datasets/failrecovery/manifest.jsonl\n'
printf '  datasets/failrecovery/<episode files/directories...>\n'
printf '  annotations/failrecovery/records/\n'
printf '  annotations/failrecovery/events/\n'
printf '  outputs/usb_event_intervals/\n'
printf '  checkpoints/T-Rex/encoders/\n'
printf '  repos/T-Rex/\n'
printf '  repos/sharpawave-deform-encoder/\n'
printf '\nStart with:\n'
printf '  python webui/server.py --root . --host 127.0.0.1 --port 8765\n'
