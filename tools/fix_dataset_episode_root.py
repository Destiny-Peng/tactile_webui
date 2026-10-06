from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def patch(path: str, old: str, new: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"missing expected text in {path}: {old!r}")
    p.write_text(text.replace(old, new), encoding="utf-8")


patch(
    "tools/export_failrecovery_media.py",
    'EPISODE_ROOT = DATASET_ROOT / "failrecovery"',
    'EPISODE_ROOT = DATASET_ROOT',
)

patch(
    "README.md",
    '''datasets/\n  failrecovery/\n    manifest.jsonl\n    failrecovery/\n    ...''',
    '''datasets/\n  failrecovery/\n    manifest.jsonl\n    <episode files/directories...>''',
)

path = ROOT / "tools/migrate_from_lf3r.sh"
text = path.read_text(encoding="utf-8")
old = '''if [[ -f "$LEGACY_DATA/failrecovery_manifest.jsonl" ]]; then
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
'''
new = '''if [[ -f "$LEGACY_DATA/failrecovery_manifest.jsonl" ]]; then
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
'''
if old not in text:
    raise SystemExit("migration source/copy block not found")
text = text.replace(old, new)
text = text.replace("printf '  datasets/failrecovery/failrecovery/\\n'\n", "printf '  datasets/failrecovery/<episode files/directories...>\\n'\n")
path.write_text(text, encoding="utf-8")

print("flattened failrecovery episode root")
