from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: Path, marker: str, addition: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    path.write_text(text.rstrip() + "\n\n" + addition.rstrip() + "\n", encoding="utf-8")


# ---- tactile service: prefer pre-materialized per-finger PNGs ----
service = ROOT / "webui/tactile_service.py"
replace_once(
    service,
    '''    def image(self, rollout_id: str, finger: str, event_id: str, kind: str) -> bytes:\n        if finger not in FINGERS or kind not in KINDS:\n            raise KeyError("unknown tactile image")\n        episode = self._episode(rollout_id)\n        event = self._event(episode, finger, event_id)\n        stream = episode.streams.get(finger, {}).get(kind)\n        if event is None or stream is None:\n            raise KeyError("tactile image not found")\n\n        offset = int(event[kind + "_offset_bytes"])\n        length = int(event[kind + "_length_bytes"])\n        shape = event[kind + "_shape"]\n        if offset < 0 or length <= 0:\n            raise ValueError("invalid tactile image byte range")\n        with stream.open("rb") as handle:\n            handle.seek(offset)\n            data = handle.read(length)\n        if len(data) != length:\n            raise ValueError("short tactile image read")\n        return self._png(data, shape)\n''',
    '''    def static_image_path(self, rollout_id: str, finger: str, event_id: str, kind: str) -> Path | None:\n        if finger not in FINGERS or kind not in KINDS:\n            raise KeyError("unknown tactile image")\n        episode = self._episode(rollout_id)\n        event = self._event(episode, finger, event_id)\n        stream = episode.streams.get(finger, {}).get(kind)\n        if event is None or stream is None:\n            raise KeyError("tactile image not found")\n        sample_index = event.get("sample_index")\n        if sample_index is None:\n            return None\n        path = stream.parent / "images" / kind / finger / f"{int(sample_index):06d}.png"\n        return path if path.is_file() else None\n\n    def image(self, rollout_id: str, finger: str, event_id: str, kind: str) -> bytes:\n        if finger not in FINGERS or kind not in KINDS:\n            raise KeyError("unknown tactile image")\n        episode = self._episode(rollout_id)\n        event = self._event(episode, finger, event_id)\n        stream = episode.streams.get(finger, {}).get(kind)\n        if event is None or stream is None:\n            raise KeyError("tactile image not found")\n\n        static_path = self.static_image_path(rollout_id, finger, event_id, kind)\n        if static_path is not None:\n            return static_path.read_bytes()\n\n        # Compatibility fallback for an old export that has not been\n        # materialized yet. Canonical migrated/new exports use static PNGs.\n        offset = int(event[kind + "_offset_bytes"])\n        length = int(event[kind + "_length_bytes"])\n        shape = event[kind + "_shape"]\n        if offset < 0 or length <= 0:\n            raise ValueError("invalid tactile image byte range")\n        with stream.open("rb") as handle:\n            handle.seek(offset)\n            data = handle.read(length)\n        if len(data) != length:\n            raise ValueError("short tactile image read")\n        return self._png(data, shape)\n''',
    "static tactile image preference",
)

# ---- Annotate: five stable per-finger static images ----
index = ROOT / "webui/static/index.html"
replace_once(
    index,
    '''              <section class="tactile-preview-card">\n                <div><p class="eyebrow">CURRENT TACTILE FRAME</p><span id="annotateTactileStatus" class="muted">Waiting for synchronized tactile…</span></div>\n                <img id="annotateTactileSprite" alt="Five-finger tactile sprite">\n              </section>''',
    '''              <section class="tactile-preview-card">\n                <div><p class="eyebrow">CURRENT TACTILE FRAME</p><span id="annotateTactileStatus" class="muted">Waiting for synchronized tactile…</span></div>\n                <div id="annotateTactileGrid" class="annotate-tactile-grid" aria-label="Five-finger tactile images"></div>\n              </section>''',
    "annotate tactile grid",
)

shell_css = ROOT / "webui/static/shell.css"
append_once(
    shell_css,
    ".annotate-tactile-grid",
    '''.tactile-preview-card { display: block; }\n.tactile-preview-card > div:first-child { margin-bottom: 9px; }\n.annotate-tactile-grid { display: grid; grid-template-columns: repeat(5,minmax(0,1fr)); gap: 7px; }\n.annotate-tactile-finger { min-width: 0; padding: 5px; background: #f7f9fc; border: 1px solid var(--line-soft); border-radius: 8px; }\n.annotate-tactile-finger strong { display: block; margin-bottom: 4px; color: #5d6c7f; font: 700 .55rem/1 ui-monospace,monospace; text-align: center; text-transform: uppercase; }\n.annotate-tactile-finger img { display: block; width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #eef2f6; border-radius: 5px; }\n.annotate-tactile-finger .tactile-missing { display: grid; width: 100%; aspect-ratio: 1 / 1; place-items: center; color: var(--muted); background: #eef2f6; border-radius: 5px; font-size: .52rem; }''',
)

shell_js = ROOT / "webui/static/shell.js"
replace_once(
    shell_js,
    'var LABEL_KEYS = [6, 7, 8, 9];\n',
    'var LABEL_KEYS = [6, 7, 8, 9];\nvar TACTILE_FINGERS = ["thumb", "index", "middle", "ring", "pinky"];\n',
    "tactile finger constant",
)
replace_once(
    shell_js,
    '''  tactileGeneration: 0,\n  results: {''',
    '''  tactileGeneration: 0,\n  tactileSeries: null,\n  tactileSeriesKey: "",\n  tactileSeriesPromise: null,\n  tactileAppliedKey: "",\n  results: {''',
    "annotate tactile state",
)
replace_once(
    shell_js,
    '''  state.dirty = false;\n  byId("annotateEmpty").classList.add("hidden");''',
    '''  state.dirty = false;\n  state.tactileSeries = null;\n  state.tactileSeriesKey = "";\n  state.tactileSeriesPromise = null;\n  state.tactileAppliedKey = "";\n  byId("annotateEmpty").classList.add("hidden");''',
    "reset tactile series on rollout",
)
old_update = '''function updateAnnotateTactile(frame) {\n  var record = selectedRollout();\n  if (!record) return;\n  var camera = byId("annotateCamera").value || chooseCamera(record);\n  var kind = state.settings && state.settings.default_tactile_kind || "deform";\n  var generation = ++state.tactileGeneration;\n  var image = byId("annotateTactileSprite");\n  image.onload = function () { if (generation === state.tactileGeneration) byId("annotateTactileStatus").textContent = camera + " · frame " + frame + " · " + kind; };\n  image.onerror = function () { if (generation === state.tactileGeneration) byId("annotateTactileStatus").textContent = "No synchronized tactile at this frame"; };\n  image.src = "/api/tactile/" + encodeURIComponent(record.id) + "/sprite?camera=" + encodeURIComponent(camera) + "&frame=" + frame + "&kind=" + encodeURIComponent(kind);\n  for (var offset = 1; offset <= 2; offset += 1) {\n    var preload = new Image();\n    preload.src = "/api/tactile/" + encodeURIComponent(record.id) + "/sprite?camera=" + encodeURIComponent(camera) + "&frame=" + clampFrame(frame + offset) + "&kind=" + encodeURIComponent(kind);\n  }\n}\n'''
new_update = '''function annotateTactileImageUrl(record, finger, eventId, kind) {\n  return "/api/tactile/" + encodeURIComponent(record.id) + "/image?finger=" + encodeURIComponent(finger)\n    + "&event_id=" + encodeURIComponent(eventId) + "&kind=" + encodeURIComponent(kind);\n}\nfunction nearestAnnotateTactileSync(series, frame) {\n  var rows = series && Array.isArray(series.sync_frames) ? series.sync_frames : [];\n  if (!rows.length) return null;\n  var low = 0, high = rows.length - 1;\n  while (low <= high) {\n    var mid = (low + high) >> 1;\n    var value = Number(rows[mid].frame);\n    if (value < frame) low = mid + 1; else if (value > frame) high = mid - 1; else return rows[mid];\n  }\n  if (low <= 0) return rows[0];\n  if (low >= rows.length) return rows[rows.length - 1];\n  return Math.abs(frame - Number(rows[low - 1].frame)) <= Math.abs(Number(rows[low].frame) - frame) ? rows[low - 1] : rows[low];\n}\nfunction ensureAnnotateTactileSeries(record, camera) {\n  var key = record.id + "|" + camera;\n  if (state.tactileSeriesKey === key && state.tactileSeries) return Promise.resolve(state.tactileSeries);\n  if (state.tactileSeriesKey === key && state.tactileSeriesPromise) return state.tactileSeriesPromise;\n  state.tactileSeriesKey = key;\n  state.tactileSeries = null;\n  state.tactileAppliedKey = "";\n  state.tactileSeriesPromise = jsonRequest("/api/tactile/" + encodeURIComponent(record.id) + "/series?camera=" + encodeURIComponent(camera))\n    .then(function (payload) {\n      if (state.tactileSeriesKey !== key) return null;\n      state.tactileSeries = payload.tactile || null;\n      return state.tactileSeries;\n    }).catch(function (error) {\n      if (state.tactileSeriesKey === key) byId("annotateTactileStatus").textContent = "Tactile timeline unavailable: " + String(error.message || error);\n      return null;\n    }).finally(function () {\n      if (state.tactileSeriesKey === key) state.tactileSeriesPromise = null;\n    });\n  return state.tactileSeriesPromise;\n}\nasync function updateAnnotateTactile(frame) {\n  var record = selectedRollout();\n  if (!record) return;\n  var camera = byId("annotateCamera").value || chooseCamera(record);\n  var kind = state.settings && state.settings.default_tactile_kind || "deform";\n  var generation = ++state.tactileGeneration;\n  var series = await ensureAnnotateTactileSeries(record, camera);\n  if (generation !== state.tactileGeneration || !series) return;\n  var sync = nearestAnnotateTactileSync(series, frame);\n  if (!sync) { byId("annotateTactileStatus").textContent = "No synchronized tactile at this frame"; return; }\n  var fingerRows = sync.fingers || {};\n  var key = record.id + "|" + camera + "|" + sync.frame + "|" + kind + "|"\n    + TACTILE_FINGERS.map(function (finger) { return fingerRows[finger] && fingerRows[finger].event_id != null ? fingerRows[finger].event_id : "-"; }).join(",");\n  if (key === state.tactileAppliedKey) return;\n  byId("annotateTactileStatus").textContent = camera + " · video " + frame + " → tactile " + sync.frame + " · loading " + kind;\n  var cells = [];\n  var loads = [];\n  TACTILE_FINGERS.forEach(function (finger) {\n    var info = fingerRows[finger] || {};\n    if (info.event_id == null) { cells.push({ finger: finger, missing: true }); return; }\n    var url = annotateTactileImageUrl(record, finger, info.event_id, kind);\n    cells.push({ finger: finger, url: url });\n    loads.push(new Promise(function (resolve, reject) {\n      var image = new Image();\n      image.onload = resolve;\n      image.onerror = function () { reject(new Error(finger + " tactile image failed")); };\n      image.src = url;\n    }));\n  });\n  try {\n    await Promise.all(loads);\n  } catch (error) {\n    if (generation !== state.tactileGeneration) return;\n    byId("annotateTactileStatus").textContent = "Tactile image load failed; keeping previous frame";\n    queueTelemetry({ event: "tactile_image_error", level: "error", rollout_id: record.id, camera: camera, frame: frame, matched_frame: sync.frame, kind: kind, message: String(error.message || error) });\n    return;\n  }\n  if (generation !== state.tactileGeneration) return;\n  byId("annotateTactileGrid").innerHTML = cells.map(function (cell) {\n    return '<div class="annotate-tactile-finger"><strong>' + escapeHtml(cell.finger) + '</strong>'\n      + (cell.missing ? '<div class="tactile-missing">No sample</div>' : '<img src="' + escapeHtml(cell.url) + '" alt="' + escapeHtml(cell.finger + " " + kind) + '">') + '</div>';\n  }).join("");\n  state.tactileAppliedKey = key;\n  byId("annotateTactileStatus").textContent = camera + " · video " + frame + " → tactile " + sync.frame + " · " + kind;\n}\n'''
replace_once(shell_js, old_update, new_update, "stable annotate tactile loader")
replace_once(
    shell_js,
    '''  byId("annotateCamera").addEventListener("change", function () { loadAnnotateVideo(); seekFrame(0); });''',
    '''  byId("annotateCamera").addEventListener("change", function () { state.tactileSeries = null; state.tactileSeriesKey = ""; state.tactileSeriesPromise = null; state.tactileAppliedKey = ""; loadAnnotateVideo(); seekFrame(0); });''',
    "camera tactile reset",
)

# ---- Detailed tactile viewer: direct per-finger image endpoint, no runtime sprite ----
tactile_js = ROOT / "webui/static/tactile/app.js"
replace_once(
    tactile_js,
    '''  function renderFinger(finger, data, kind) {\n''',
    '''  function tactileImageUrl(record, finger, eventId, kind) {\n    return "/api/tactile/" + encodeURIComponent(record.id) + "/image"\n      + "?finger=" + encodeURIComponent(finger)\n      + "&event_id=" + encodeURIComponent(eventId)\n      + "&kind=" + encodeURIComponent(kind);\n  }\n\n  function renderFinger(record, finger, data, kind) {\n''',
    "tactile viewer image url",
)
replace_once(
    tactile_js,
    '''    var fingerIndex = FINGERS.indexOf(finger);\n    var image = hasImage\n      ? '<div class="tactile-image tactile-sprite finger-' + fingerIndex + '" role="img" aria-label="'\n        + escapeHtml(LABELS[finger]) + ' tactile ' + escapeHtml(kind) + '"></div>'\n      : '<div class="tactile-image placeholder">No image</div>';''',
    '''    var image = hasImage\n      ? '<img class="tactile-image" loading="eager" src="' + escapeHtml(tactileImageUrl(record, finger, data.event_id, kind)) + '" alt="'\n        + escapeHtml(LABELS[finger]) + ' tactile ' + escapeHtml(kind) + '">'\n      : '<div class="tactile-image placeholder">No image</div>';''',
    "direct per-finger tactile image",
)
replace_once(
    tactile_js,
    '''      return renderFinger(finger, currentFingerData(syncRow, finger), kind);''',
    '''      return renderFinger(record, finger, currentFingerData(syncRow, finger), kind);''',
    "render finger call",
)
replace_once(
    tactile_js,
    '''    state.appliedSpriteKey = "";\n    applySprite(record, camera, Number(syncRow.frame), match.index, kind);\n    updateCurvePlayhead(frame);''',
    '''    state.appliedSpriteKey = "";\n    updateCurvePlayhead(frame);''',
    "remove dynamic sprite application",
)

tactile_css = ROOT / "webui/static/tactile/styles.css"
append_once(
    tactile_css,
    ".tactile-image-static-mode",
    '''.tactile-image-static-mode { display: none; }\nimg.tactile-image { object-fit: contain; }''',
)

# ---- migration automatically backfills static raw/deform images ----
migrate = ROOT / "tools/migrate_from_lf3r.sh"
replace_once(
    migrate,
    '''PY\n\nnote "migrating annotations into annotations/failrecovery"''',
    '''PY\n\nnote "materializing static tactile images"\npython3 "$TARGET_ROOT/tools/materialize_tactile_images.py" --root "$TARGET_ROOT" --manifest datasets/failrecovery/manifest.jsonl\n\nnote "migrating annotations into annotations/failrecovery"''',
    "migration tactile materialization",
)

# ---- future exports directly emit the same static PNG hierarchy ----
exporter = ROOT / "tools/export_failrecovery_media.py"
replace_once(
    exporter,
    '''from typing import Any\n\n\nPROJECT_ROOT''',
    '''from typing import Any\n\ntry:\n    from materialize_tactile_images import image_path as tactile_image_path, png_bytes\nexcept ImportError:\n    from tools.materialize_tactile_images import image_path as tactile_image_path, png_bytes\n\n\nPROJECT_ROOT''',
    "exporter materializer import",
)
replace_once(
    exporter,
    '''            raw_file.write(raw)\n            deform_file.write(deform)\n            index.write(json_line(record))''',
    '''            raw_file.write(raw)\n            deform_file.write(deform)\n            for kind, data, shape, stream_file in (\n                ("raw", raw, raw_shape, raw_file),\n                ("deform", deform, deform_shape, deform_file),\n            ):\n                image_target = tactile_image_path(Path(stream_file.name), kind, finger, counts[finger])\n                image_target.parent.mkdir(parents=True, exist_ok=True)\n                image_target.write_bytes(png_bytes(data, shape, compression_level=3))\n            index.write(json_line(record))''',
    "export static tactile images",
)
replace_once(
    exporter,
    '''            "tactile_encoding": "uint8 row-major; byte offsets and shapes in tactile/events.jsonl",\n            "tactile_fingers": list(FINGERS),''',
    '''            "tactile_encoding": "uint8 row-major; byte offsets and shapes in tactile/events.jsonl",\n            "tactile_image_layout": "tactile/images/{raw,deform}/<finger>/<sample_index:06d>.png",\n            "tactile_fingers": list(FINGERS),''',
    "export metadata image layout",
)

# ---- README ----
readme = ROOT / "README.md"
append_once(
    readme,
    "## Static tactile images",
    '''## Static tactile images\n\nThe WebUI uses materialized tactile PNGs for display instead of encoding PNGs on every frame request. Images are grouped by kind and finger inside each episode:\n\n```text\ntactile/images/\n  deform/\n    thumb/\n    index/\n    middle/\n    ring/\n    pinky/\n  raw/\n    thumb/\n    index/\n    middle/\n    ring/\n    pinky/\n```\n\nFilenames use the per-finger `sample_index` (`000000.png`, `000001.png`, ...), so repeated video frames that reference the same tactile event do not duplicate images. `tools/migrate_from_lf3r.sh` materializes these files automatically. For an existing canonical dataset, run:\n\n```bash\npython tools/materialize_tactile_images.py --root .\n```\n\nThe server retains a compatibility fallback to the packed `.u8` streams when a static PNG is missing, but migrated and newly exported episodes use the static path.''' ,
)
