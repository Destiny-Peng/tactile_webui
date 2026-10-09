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

## Expected data layout

The WebUI and SHARPA scripts retain the fail-recovery layout used by the dissertation project:

```text
datasets/
  failrecovery/
    manifest.jsonl
    <episode files/directories...>
```

Manifest entries use the exported fields `camera_video_paths`, `synchronized_frames_path`, `tactile_events_path`, and `tactile_stream_paths`.

The fail-recovery payload does not need to be copied. `datasets/failrecovery/manifest.jsonl` may be a symlink to the original LF3R manifest. When it is linked, the WebUI resolves the symlink target, inspects the project-relative paths stored in that manifest, and automatically infers the original source-project root. Video, synchronization, tactile-event, tactile-stream, and historical interval-seed paths therefore continue to resolve against the original LF3R tree while new annotations stay local to this repository.

## One-command migration

To connect this repository to an existing dissertation/LF3R data tree without copying the payload, run:

```bash
bash tools/migrate_from_lf3r.sh /mnt/hdd/pyr/LF3R
```

The script copies the exported fail-recovery dataset into `datasets/failrecovery/`, rewrites manifest paths that still contain `/v1/`, migrates annotation records/events and USB interval seeds, and copies the tactile encoder checkpoints plus the two external tactile repositories when present. It intentionally does not copy virtual environments, raw recorder SQLite data, or historical generated SHARPA runs.

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

## Static tactile images

The WebUI uses materialized tactile PNGs for display instead of encoding PNGs on every frame request. Images are grouped by kind and finger inside each episode:

```text
tactile/images/
  deform/
    thumb/
    index/
    middle/
    ring/
    pinky/
  raw/
    thumb/
    index/
    middle/
    ring/
    pinky/
```

Filenames use the per-finger `sample_index` (`000000.png`, `000001.png`, ...), so repeated video frames that reference the same tactile event do not duplicate images. `tools/migrate_from_lf3r.sh` materializes these files automatically. For an existing canonical dataset, run:

```bash
python tools/materialize_tactile_images.py --root .
```

The server retains a compatibility fallback to the packed `.u8` streams when a static PNG is missing, but migrated and newly exported episodes use the static path.

## Migrated SHARPA tools

The complete tactile experiment toolchain and launch scripts now live under `tools/`. Run `bash tools/run_sharpa_tactile_ablation.sh --list` to discover modules. See [tool migration and validation](tools/MIGRATION.md) for data/output path resolution, interpreter selection, and preserved original versions.
