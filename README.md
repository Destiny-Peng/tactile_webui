# LF3R Tactile

Standalone tactile tooling split from `Destiny-Peng/dissertation`.

This repository contains two related pieces:

- a standalone browser UI for synchronized robot video + five-finger SHARPA tactile streams;
- the SHARPA tactile representation / failure-prediction experiment code previously under `tools/sharpa_tactile`.

The source split was prepared from dissertation commit `6d211285ed33d6e8cab0a83f48893e531f90c07f`.

## Layout

```text
webui/
  server.py                 minimal standalone HTTP server
  tactile_service.py        synchronized tactile reader / sprite service
  static/tactile/           standalone tactile UI
  tests/test_tactile_service.py

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

The WebUI and SHARPA scripts retain the dataset layout used by the dissertation project:

```text
datasets/
  lf3r_failure_rollouts/
    v1/
      failrecovery_manifest.jsonl
      failrecovery/
      ...
```

Manifest entries use the existing exported fields, including `camera_video_paths`, `synchronized_frames_path`, `tactile_events_path`, and `tactile_stream_paths`.

Copy the dataset into this repository or symlink `datasets/lf3r_failure_rollouts` to the existing dataset location.

## Run the tactile WebUI

The WebUI backend uses only the Python standard library.

```bash
python webui/server.py --root . --host 127.0.0.1 --port 8765
```

Then open `http://127.0.0.1:8765/`.

The standalone UI keeps the optimized synchronized rendering path from the dissertation WebUI: one five-finger sprite per synchronized frame, a small sprite prefetch window, and de-duplicated tactile-series requests.

Service test:

```bash
python -m unittest webui.tests.test_tactile_service -v
```

## SHARPA tactile experiments

The experiment package retains its original root assumptions. Put the external repositories under:

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

`tools/export_failrecovery_media.py` and `tools/plot_failrecovery_tactile.py` are included because they produce and inspect the synchronized tactile representation consumed by both the WebUI and the SHARPA experiments.

## Intentionally excluded

This split does not include LIBERO, repair/world-model code, general LF3R annotation/results pages, Robo-Dopamine/SAFE/ProcVLM integrations, generated experiment outputs, raw tactile datasets, model checkpoints, or vendored third-party repositories.
