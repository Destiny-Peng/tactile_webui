# Tactile WebUI

Standalone tactile annotation, synchronized visualization, and SHARPA experiment tooling split from `Destiny-Peng/dissertation`.

The repository keeps the familiar WebUI workspace shape while removing LF3R-specific repair and simulation dependencies. The top-level browser workspace has five pages:

- **Annotate** — real-robot video + synchronized five-finger tactile preview and interval annotation;
- **Results** — the full tactile viewer with raw/deform sprites and F6 history curves;
- **Runs** — read-only discovery of SHARPA experiment outputs under `outputs/`;
- **Analysis** — annotation coverage and the four tactile event distributions;
- **Settings** — standalone paths, default camera/image type, and theme.

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
  lf3r_failure_rollouts/
    v1/
      failrecovery_manifest.jsonl
      failrecovery/
      ...
```

Manifest entries use the exported fields `camera_video_paths`, `synchronized_frames_path`, `tactile_events_path`, and `tactile_stream_paths`.

Do not symlink the dataset tree when the manifest contains paths relative to the original project root. In **Settings → Source project root**, point directly to that original project (for example `/mnt/hdd/pyr/LF3R`). The WebUI reads `datasets/lf3r_failure_rollouts/v1/failrecovery_manifest.jsonl` there and resolves every exported relative video/tactile path against the same source root. The standalone repository keeps its own annotations and experiment outputs.

## Run the WebUI

The WebUI backend uses only the Python standard library.

```bash
python webui/server.py --root . --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765/`.

The detailed Results viewer is also directly available at `http://127.0.0.1:8765/tactile`.

## Annotation schema

The standalone annotator edits only the four USB tactile interval types already consumed by `tools/sharpa_tactile`:

| event_key | label | outcome |
|---:|---|---|
| 6 | Align failure | failure |
| 7 | Insert failure | failure |
| 8 | Align success | success |
| 9 | Insert success | success |

Each saved record keeps the experiment-compatible fields `rollout_id`, `event_index`, `event_key`, `start_frame`, and `end_frame`.

The default writable annotation file is:

```text
annotations/tactile_intervals.jsonl
```

If that file does not yet exist, the WebUI looks for the latest historical source matching:

```text
outputs/usb_event_intervals/*/intervals.jsonl
```

That historical file is loaded as a **read-only seed**. On the first edit, all seed records are copied into the standalone annotation target and only the new target is modified. The original experiment input is not overwritten.

Keyboard controls on Annotate:

```text
1 / 2 / 3 / 4   select keys 6 / 7 / 8 / 9
Q / W           set active interval start / end to the current frame
↑ / ↓           previous / next active interval
, / .           previous / next rollout
← / →           previous / next video frame
Space           play / pause
```

## Tactile Results viewer

Results keeps the optimized synchronized path from the dissertation WebUI:

- synchronized robot video;
- five-finger raw/deform tactile sprites;
- F6 values/history;
- one five-finger sprite per synchronized frame;
- small forward sprite prefetch window;
- de-duplicated tactile-series requests.

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
