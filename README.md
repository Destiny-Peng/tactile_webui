# Tactile WebUI

Standalone tactile annotation, synchronized visualization, and SHARPA experiment tooling split from `Destiny-Peng/dissertation`.

The repository keeps the familiar WebUI workspace shape while removing LF3R-specific repair and simulation dependencies. The top-level browser workspace has five pages:

- **Annotate** — real-robot video + synchronized five-finger tactile preview and interval annotation;
- **Results** — the full tactile viewer with raw/deform sprites and F6 history curves;
- **Runs** — read-only discovery of SHARPA experiment outputs under `outputs/`;
- **Analysis** — annotation coverage and the four tactile event distributions;
- **Settings** — standalone paths, display label registry, default camera/image type, and theme.

There is intentionally no Repair page and no LIBERO, world-model, Robo-Dopamine, SAFE, or ProcVLM backend.

## Layout

```text
webui/
  server.py                 standalone HTTP/API server
  tactile_service.py        synchronized tactile reader / sprite service
  static/
    index.html              Annotate → Settings workspace shell
    shell.js
    shell.css
    tactile/                 detailed tactile Results viewer
  tests/
    test_tactile_service.py
    test_workspace.py

tools/
  sharpa_tactile/           complete SHARPA tactile experiment package
  export_failrecovery_media.py
  plot_failrecovery_tactile.py
  extract_trex_encoders.py
  run_trex.sh
  run_sharpa_tactile_ablation.sh
```

Large data, model checkpoints, generated outputs and third-party repositories are intentionally not copied into source control.

## Data sources: peer manifests + original SQLite

The WebUI automatically discovers all `datasets/*/manifest.jsonl` peer manifests, including symlinks. It checks duplicate rollout IDs across manifests, shows manifest provenance, and resolves relative media paths against each dataset's own data root. The existing LF3R fail-recovery manifest location is supported as a fallback.

**Existing LF3R manifest:** keep your symlink. If a row has a valid `source_database_path`, the WebUI prefers the original `episode.sqlite3` and original `videos/realsense_color.mp4` / `videos/wrist_right.mp4`. Nothing is extracted. If the original recorder database is unavailable, the older JSONL + `.u8` export reader still works.

**New original SQLite dataset:** index its recording directory once:

```bash
python tools/index_sqlite_dataset.py /mnt/hdd/qiuxia/datasets/failrecovery --name raw_failrecovery
./start.sh
```

This generates only `datasets/raw_failrecovery/manifest.jsonl` and a small summary JSON in this repository; it never copies or modifies recorder SQLite files, raw/deform BLOBs or MP4s. Each source episode is expected to have `episode.sqlite3`, `manifest.json`, and `videos/realsense_color.mp4` plus `videos/wrist_right.mp4`. The indexer requires `ffprobe`. Specify a different unique `--name` for each dataset and `--overwrite` to explicitly refresh a locally generated manifest. Do not index the same episodes a second time if your existing linked LF3R manifest already exposes their original database; peer manifests with duplicate rollout IDs are marked invalid rather than silently overwritten.

The online viewer loads `synchronized_frames` and the small non-BLOB event fields (including F6) from SQLite when an episode is opened. Each requested tactile image reads only that `event_id`'s Raw or Deform BLOB and encodes a grayscale PNG in memory, with bounded LRU caches. The original SQLite is opened read-only. No `frames.jsonl`, `events.jsonl`, `.u8`, or pre-generated PNG is required. Original MP4 files are served directly when the codec is browser-compatible. For MPEG-4 Part 2 and other unsupported source codecs, use **Runs → Utilities → Batch H.264 transcode**: choose the loaded manifests (all selected by default), then click **Transcode selected manifests**. The task skips already-compatible videos, deduplicates source paths, shows live counts and a persistent log, and writes H.264/yuv420p files under `cache/videos/h264/`. Original SQLite and MP4 files are never moved or replaced. The WebUI automatically serves cached versions on subsequent requests; if the original file changes, its old cached copy is no longer used. Requires `ffmpeg` and `ffprobe` on PATH and available disk space. Converted videos persist across WebUI restarts; a batch can be started again to process only unfinished items. Process status and logs are kept under `logs/webui/transcode/`.

## Link an existing LF3R manifest

```bash
bash tools/migrate_from_lf3r.sh /mnt/hdd/qiuxia/pyr/LF3R
```

This script updates the standalone manifest symlink and optional encoder repository symlinks. New annotations stay local; it does not copy the dataset or rewrite the source manifest.

## Run the WebUI

The WebUI backend uses only the Python standard library. From the repository root:

```bash
./start.sh
```

The launcher works from any working directory, uses `.venv/bin/python` when available (otherwise the current `python3`), and runs the server in the foreground. To use a different interpreter or bind address/port:

```bash
WEBUI_PYTHON=/path/to/python ./start.sh --host 0.0.0.0 --port 8766
```

Then open `http://127.0.0.1:8765/` for the default host/port. Runtime logs remain under `logs/webui/server/` and `logs/webui/client/`.

The synchronized tactile diagnostic viewer remains directly available at `http://127.0.0.1:8765/tactile`. The main **Results** page is now the online-detection review surface.

**Annotate** has a collapsible rollout queue (arrow beside the task title) and LF3R-style camera view buttons when multiple video views exist; changing cameras preserves the current frame and resumes playback if needed. Its tactile preview combines five-finger deform/raw images with **current F6 values for each finger**, plus a selectable finger's **six-channel F6 history** and a video-synchronized playhead. Raw/deform is selected via Settings; the F6 display is independent of image kind.

## Annotation schema

The annotator uses a project-local label registry in `config/annotation_labels.json` (editable in **Settings → Annotation labels**). Names, descriptions, colors and active status drive Annotate, Results, and Analysis. Label IDs are stable: editing their presentation does not rewrite annotation records or change experiment targets.

The initial labels are **1 Success**, **2 Failure**, **3 Dropped object**, and **4 Wrong object**. Labels 3/4 are restricted to rollouts with authoritative `ground_truth_outcome=failure`. The other two describe local action intervals, not necessarily the final rollout outcome.

Add new label IDs through Settings and choose `All` or `Failure only` before first save. Existing IDs and their eligibility scopes cannot be removed or changed; turn off **Active** to hide a label from new annotation choices while retaining previously saved intervals. An unregistered historical ID is shown rather than coerced into Success, but cannot be saved until registered or explicitly reassigned. Avoid repurposing an existing ID for a new semantic concept.

The backend exposes `GET/POST /api/labels`. The label registry is checked by the server when saving intervals. This is a UI annotation schema, not an automatic mapping to training targets: downstream 0/1/2 online GT or other model labels remain defined by experiment code.

Each saved record retains the experiment-compatible fields `rollout_id`, `event_index`, `event_key`, `start_frame`, and `end_frame`.

Standalone annotations are stored one rollout per file under:

```text
annotations/failrecovery/records/<rollout-id>.tactile.json
```

Existing `<rollout-id>.json` LF3R records can coexist beside the tactile sidecars and are never overwritten. If a tactile sidecar does not yet exist, the WebUI looks for the latest historical source matching:

```text
outputs/usb_event_intervals/*/intervals.jsonl
```

That historical file is loaded as a **read-only seed**. On the first edit, all seed records are copied into the standalone annotation target and only the new target is modified. The original experiment input is not overwritten.

Keyboard controls on Annotate:

```text
1 / 2 / 3 / 4   select the corresponding label IDs
Q / W           set active interval start / end to the current frame
↑ / ↓           previous / next active interval
, / .           previous / next rollout
← / →           previous / next video frame
Space           play / pause
```

## Results and diagnostics

The main Results page reviews online detection against the robot video on one shared frame axis. It shows saved numeric annotation intervals, numeric GT labels, numeric predicted labels, and the three model probability curves (`p0`, `p1`, `p2`). Existing SHARPA `test_predictions.csv` files below the configured Runs root are discovered automatically.

The legacy `/tactile` route remains a low-level synchronized tactile diagnostic viewer.

WebUI stability/performance telemetry is written only below:

```text
logs/webui/server/
logs/webui/client/
```

The Settings → Diagnostics panel reports server/video latency, HTTP errors, browser video stalls/errors and the active log paths. Successful high-frequency tactile requests are sampled so monitoring does not add material I/O load.

## One-episode Deform compression benchmark

To compare lossy WebP with the existing PNG approach using an **original** recorder episode (no export/copy of the dataset):

```bash
python -m pip install Pillow
python tools/benchmark_deform_webp.py /path/to/episode.sqlite3
```

A path to the episode directory also works. The script reads `events` and `tactile_frames.deform_blob` directly from SQLite in read-only mode; by default it tests every right-hand Deform frame at the original resolution. It measures PNG level 3, WebP Q60/Q70/Q80/Q90, sampled PSNR/decode latency, and a real indexed Q80 cache SQLite file. Raw and compressed random lookup times are also compared.

Results are saved automatically under `outputs/tactile_webp_benchmark/<episode>_<timestamp>/`:

- `report.md`: brief Markdown report you can paste into ChatGPT;
- `results.json`: full machine-readable metrics;
- `comparison.png`: raw versus all WebP qualities, on three sampled frames.

The temporary Q80 cache is removed after timing. Pass `--keep-cache` to retain it, or `--max-images 300` for a quick non-representative subset. The original SQLite, video, annotations and manifest are never rewritten. Benchmarking requires Pillow, but running the WebUI itself does not.

## Tests

```bash
python -m unittest discover -s webui/tests -v
```

The repository CI also runs Python compilation and JavaScript syntax checks.

## SHARPA tactile experiments

The experiment package retains its root-relative layout. Put the external repositories under:

```text
repos/T-Rex/
repos/sharpawave-deform-encoder/
```

Set the Python interpreter explicitly if needed:

```bash
export TREX_PYTHON=/path/to/trex/python
```

`tools/run_trex.sh` adds the required T-Rex source paths to `PYTHONPATH`, and `tools/run_sharpa_tactile_ablation.sh` dispatches the experiment modules.

Example:

```bash
bash tools/run_sharpa_tactile_ablation.sh prepare --output outputs/example --device cuda:0 --batch-size 8
bash tools/run_sharpa_tactile_ablation.sh verify --output outputs/example
bash tools/run_sharpa_tactile_ablation.sh train --output outputs/example --device cuda:0
```

## Encoder extraction

`tools/extract_trex_encoders.py` extracts and verifies the frozen F6 and deform encoders from the pinned T-Rex source checkpoint. By default it expects:

```text
checkpoints/T-Rex/midtrain_source/
```

and writes standalone encoder weights under:

```text
checkpoints/T-Rex/encoders/
```

## Dataset helpers

`tools/export_failrecovery_media.py` and `tools/plot_failrecovery_tactile.py` produce and inspect the synchronized tactile representation consumed by both the WebUI and SHARPA experiments.

## Intentionally excluded

This split does not include LIBERO, repair/world-model code, general LF3R baseline jobs, Robo-Dopamine/SAFE/ProcVLM integrations, raw tactile datasets, checkpoints, or vendored third-party repositories.

## Legacy static image compatibility

Existing exported datasets with materialized PNGs still work. The original SQLite path takes priority when present; otherwise the legacy reader uses its old PNG and packed `.u8` fallback. You do not need to regenerate those exported images for the WebUI.

## Migrated SHARPA tools

The complete tactile experiment toolchain and launch scripts now live under `tools/`. Run `bash tools/run_sharpa_tactile_ablation.sh --list` to discover modules. See [tool migration and validation](tools/MIGRATION.md) for data/output path resolution, interpreter selection, and preserved original versions.

## Rebuilt-manifest annotation ID alignment (2026-10-09)

Live tactile sidecars and canonical records/index now use `raw_failrecovery--<episode-directory>` IDs. 125 record pairs were migrated; 353 intervals on120 annotated rollouts are visible under the rebuilt152-rollout catalog, with5 empty records preserved. Labels and frame bounds unchanged; frame counts match in every case. Historical provenance and original LF3R annotations/results are retained.

Backup, correspondence and audit: [annotation_id_alignment/20261009_200000](outputs/annotation_id_alignment/20261009_200000/README.md). The new manifest has80 corresponding rollout outcomes marked unknown; annotation outcomes were retained, and no manifest outcomes were inferred or changed. Failure-only labels3/4 still require a manifest failure outcome when saved.

## Multimodal causal prefix outcome experiment (2026-10-09)

Tactile / Tactile+RGB / Tactile+Pose / Tactile+RGB+Pose / RGB+Pose, seeds42–46. Full rollout final-outcome targets, rollout-meanBCE, causalGRU128. Includes frozenDeform/DINOv2ViT-S14 caches, measuredPose34D,87/19/19rolloutsplit,25checkpoints,fullper-framepredictions and125rolloutcurves. [Results and configuration](outputs/sharpa_multimodal_prefix_outcome/20261009_223000/README.md).

Wrist camera 对照（15 个 RGB 模型重新训练，seeds42–46）：[实验报告](outputs/sharpa_multimodal_prefix_outcome_wrist/20261010_010000/README.md)。保持原 split 和非 RGB 输入，原 high camera 结果保留。
