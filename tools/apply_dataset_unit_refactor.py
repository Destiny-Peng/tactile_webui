from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding="utf-8")


def replace(path: str, old: str, new: str, required: bool = True) -> None:
    text = read(path)
    if old not in text:
        if required:
            raise SystemExit(f"missing expected text in {path}: {old!r}")
        return
    write(path, text.replace(old, new))


# WebUI: datasets/<dataset>/manifest.jsonl and annotations/<dataset>/...
replace(
    "webui/server.py",
    '"annotations_path": "annotations/failure_annotations/records",',
    '"annotations_path": "annotations/failrecovery/records",',
)
old_server = '''        dataset_root = self.source_root / "datasets" / "lf3r_failure_rollouts"\n        flat_manifest = dataset_root / "failrecovery_manifest.jsonl"\n        legacy_manifest = dataset_root / "v1" / "failrecovery_manifest.jsonl"\n        # Prefer the standalone flat layout, while still allowing an old\n        # dissertation root to be used before migration.\n        self.manifest_path = (\n            flat_manifest\n            if flat_manifest.is_file() or not legacy_manifest.is_file()\n            else legacy_manifest\n        )\n        self.tactile = FailRecoveryTactileService(self.source_root, self.manifest_path)'''
new_server = '''        dataset_root = self.source_root / "datasets" / "failrecovery"\n        manifest = dataset_root / "manifest.jsonl"\n        transitional_manifest = (\n            self.source_root\n            / "datasets"\n            / "lf3r_failure_rollouts"\n            / "failrecovery_manifest.jsonl"\n        )\n        legacy_manifest = (\n            self.source_root\n            / "datasets"\n            / "lf3r_failure_rollouts"\n            / "v1"\n            / "failrecovery_manifest.jsonl"\n        )\n        # New standalone layout is dataset-centric. The two old paths are\n        # read-only compatibility inputs for repositories not migrated yet.\n        if manifest.is_file():\n            self.manifest_path = manifest\n        elif transitional_manifest.is_file():\n            self.manifest_path = transitional_manifest\n        else:\n            self.manifest_path = legacy_manifest\n        self.tactile = FailRecoveryTactileService(self.source_root, self.manifest_path)'''
replace("webui/server.py", old_server, new_server)

replace(
    "webui/tactile_service.py",
    '            / "datasets"\n            / "lf3r_failure_rollouts"\n            / "failrecovery_manifest.jsonl"',
    '            / "datasets"\n            / "failrecovery"\n            / "manifest.jsonl"',
)

# Standalone experiment/helper defaults.
for path in (
    "tools/sharpa_tactile/prepare.py",
    "tools/plot_failrecovery_tactile.py",
    "tools/sharpa_tactile/align_gaussian_probability_curves.py",
):
    text = read(path)
    text = text.replace(
        "datasets/lf3r_failure_rollouts/failrecovery_manifest.jsonl",
        "datasets/failrecovery/manifest.jsonl",
    )
    write(path, text)

text = read("tools/export_failrecovery_media.py")
text = text.replace(
    'DATASET_ROOT = PROJECT_ROOT / "datasets/lf3r_failure_rollouts"',
    'DATASET_ROOT = PROJECT_ROOT / "datasets/failrecovery"',
)
text = text.replace(
    'MANIFEST_PATH = DATASET_ROOT / "failrecovery_manifest.jsonl"',
    'MANIFEST_PATH = DATASET_ROOT / "manifest.jsonl"',
)
write("tools/export_failrecovery_media.py", text)

# Tests use the canonical dataset-centric target while keeping the explicit
# legacy-source compatibility case intact.
text = read("webui/tests/test_workspace.py")
text = text.replace(
    'root / "datasets/lf3r_failure_rollouts/failrecovery_manifest.jsonl"',
    'root / "datasets/failrecovery/manifest.jsonl"',
)
text = text.replace(
    'app.root / "annotations/failure_annotations/records"',
    'app.root / "annotations/failrecovery/records"',
)
text = text.replace(
    'app.root / "annotations/failure_annotations/v1/records"',
    'app.root / "annotations/failrecovery/records"',
)
write("webui/tests/test_workspace.py", text)

text = read("webui/tests/test_tactile_service.py")
text = text.replace(
    "self.root / 'datasets/lf3r_failure_rollouts/failrecovery_manifest.jsonl'",
    "self.root / 'datasets/failrecovery/manifest.jsonl'",
)
write("webui/tests/test_tactile_service.py", text)

# README: describe dataset/annotation units rather than LF3R historical roots.
text = read("README.md")
text = text.replace(
    '''datasets/\n  lf3r_failure_rollouts/\n    failrecovery_manifest.jsonl\n    failrecovery/\n    ...''',
    '''datasets/\n  failrecovery/\n    manifest.jsonl\n    failrecovery/\n    ...''',
)
text = text.replace(
    "The WebUI also accepts the legacy `datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl` there",
    "The WebUI uses `datasets/failrecovery/manifest.jsonl` in this standalone repository. For an unmigrated dissertation source, it also accepts the legacy `datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl`",
)
text = text.replace(
    "annotations/failure_annotations/records/<rollout-id>.tactile.json",
    "annotations/failrecovery/records/<rollout-id>.tactile.json",
)
text = text.replace(
    "copies the exported fail-recovery dataset into the flat layout",
    "copies the exported fail-recovery dataset into `datasets/failrecovery/`",
)
write("README.md", text)

migration = r'''#!/usr/bin/env bash
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
  SOURCE_DATA="$LEGACY_DATA"
  SOURCE_MANIFEST="$LEGACY_DATA/failrecovery_manifest.jsonl"
  SOURCE_PREFIX="datasets/lf3r_failure_rollouts/v1/"
elif [[ -f "$TRANSITIONAL_DATA/failrecovery_manifest.jsonl" ]]; then
  SOURCE_DATA="$TRANSITIONAL_DATA"
  SOURCE_MANIFEST="$TRANSITIONAL_DATA/failrecovery_manifest.jsonl"
  SOURCE_PREFIX="datasets/lf3r_failure_rollouts/"
elif [[ -f "$CANONICAL_DATA/manifest.jsonl" ]]; then
  SOURCE_DATA="$CANONICAL_DATA"
  SOURCE_MANIFEST="$CANONICAL_DATA/manifest.jsonl"
  SOURCE_PREFIX="datasets/failrecovery/"
else
  fail "no failrecovery manifest found under $SOURCE_ROOT/datasets"
fi

TARGET_DATA="$TARGET_ROOT/datasets/failrecovery"
mkdir -p "$TARGET_DATA"

note "copying fail-recovery dataset"
[[ -d "$SOURCE_DATA/failrecovery" ]] || fail "missing episode directory: $SOURCE_DATA/failrecovery"
rsync -a --info=stats2 "$SOURCE_DATA/failrecovery/" "$TARGET_DATA/failrecovery/"

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
printf '  datasets/failrecovery/failrecovery/\n'
printf '  annotations/failrecovery/records/\n'
printf '  annotations/failrecovery/events/\n'
printf '  outputs/usb_event_intervals/\n'
printf '  checkpoints/T-Rex/encoders/\n'
printf '  repos/T-Rex/\n'
printf '  repos/sharpawave-deform-encoder/\n'
printf '\nStart with:\n'
printf '  python webui/server.py --root . --host 127.0.0.1 --port 8765\n'
'''
write("tools/migrate_from_lf3r.sh", migration)

print("dataset-unit refactor applied")
