"use strict";

var EVENT_TYPES = {
  6: { name: "align_failure", label: "Align failure", outcome: "failure" },
  7: { name: "insert_failure", label: "Insert failure", outcome: "failure" },
  8: { name: "align_success", label: "Align success", outcome: "success" },
  9: { name: "insert_success", label: "Insert success", outcome: "success" }
};

var state = {
  view: "annotate",
  rollouts: [],
  filtered: [],
  selectedId: null,
  currentFrame: 0,
  events: [],
  activeEvent: null,
  dirty: false,
  settings: null,
  annotationSource: null,
  annotationTarget: null,
  videoGeneration: 0,
  tactileGeneration: 0
};

function byId(id) { return document.getElementById(id); }
function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function taskLabel(record) {
  return String(record.task_description || record.instruction || record.task_key || ("Task " + (record.task_id == null ? "" : record.task_id)) || "Tactile task");
}
function formatTime(seconds) {
  var value = Math.max(0, Number(seconds) || 0);
  var minutes = Math.floor(value / 60);
  var secs = Math.floor(value % 60);
  var millis = Math.floor((value - Math.floor(value)) * 1000);
  return String(minutes).padStart(2, "0") + ":" + String(secs).padStart(2, "0") + "." + String(millis).padStart(3, "0");
}
function formatPercent(value) { return (100 * (Number(value) || 0)).toFixed(1) + "%"; }
function formatDate(value) {
  if (!value) return "";
  var date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}
function isTypingTarget(target) {
  if (!target) return false;
  if (target.isContentEditable) return true;
  return ["INPUT", "TEXTAREA", "SELECT"].indexOf(target.tagName) >= 0;
}
function badge(label, cssClass) {
  return '<span class="badge ' + escapeHtml(cssClass || "") + '">' + escapeHtml(label) + "</span>";
}

async function jsonRequest(url, options) {
  var response = await fetch(url, Object.assign({ cache: "no-store" }, options || {}));
  var payload = await response.json();
  if (!response.ok) throw new Error(payload.error || ("Request failed: " + response.status));
  return payload;
}

function applyTheme(theme) {
  document.body.classList.toggle("theme-dark", theme === "dark");
}

function setGlobalStatus(ok, text) {
  byId("globalStatus").classList.toggle("online", ok === true);
  byId("globalStatus").classList.toggle("error", ok === false);
  byId("globalStatusText").textContent = text;
}

function viewTitle(view) {
  return view.charAt(0).toUpperCase() + view.slice(1);
}

function switchView(view, updateHash) {
  if (!byId("view-" + view)) return;
  if (state.view === "annotate" && view !== "annotate" && state.dirty) {
    if (!window.confirm("Discard unsaved interval changes?")) return;
    state.dirty = false;
  }
  state.view = view;
  document.querySelectorAll(".view").forEach(function (node) {
    node.classList.toggle("active-view", node.id === "view-" + view);
  });
  document.querySelectorAll(".nav-item").forEach(function (button) {
    button.classList.toggle("active", button.dataset.view === view);
  });
  byId("pageTitle").textContent = viewTitle(view);
  if (updateHash !== false) history.replaceState(null, "", "#" + view);
  if (view === "analysis") loadAnalysis();
  if (view === "runs") loadRuns();
  if (view === "settings") loadSettings();
}

function selectedRollout() {
  return state.rollouts.find(function (record) { return record.id === state.selectedId; }) || null;
}

function updateHeader() {
  var complete = state.rollouts.filter(function (record) { return record.annotation_status === "complete"; }).length;
  byId("datasetStatus").textContent = state.rollouts.length + " tactile rollouts";
  byId("annotationStatus").textContent = complete + " / " + state.rollouts.length + " annotated";
}

function populateTaskFilter() {
  var select = byId("annotateTaskFilter");
  var current = select.value || "all";
  var tasks = Array.from(new Set(state.rollouts.map(taskLabel))).sort();
  select.innerHTML = '<option value="all">All tasks</option>' + tasks.map(function (task) {
    return '<option value="' + escapeHtml(task) + '">' + escapeHtml(task) + "</option>";
  }).join("");
  select.value = tasks.indexOf(current) >= 0 ? current : "all";
}

function filterRollouts() {
  var query = byId("annotateSearch").value.trim().toLowerCase();
  var task = byId("annotateTaskFilter").value;
  var review = byId("annotateReviewFilter").value;
  state.filtered = state.rollouts.filter(function (record) {
    var haystack = [record.id, taskLabel(record), record.task_key, record.task_id, record.episode_index].join(" ").toLowerCase();
    return (!query || haystack.indexOf(query) >= 0)
      && (task === "all" || taskLabel(record) === task)
      && (review === "all" || record.annotation_status === review);
  });
  byId("annotateVisibleCount").textContent = state.filtered.length;
  renderRolloutList();
}

function renderRolloutList() {
  var container = byId("annotateRolloutList");
  if (!state.filtered.length) {
    container.innerHTML = '<div class="empty-card" style="min-height:120px;padding:15px"><span>No matching rollouts.</span></div>';
    return;
  }
  container.innerHTML = state.filtered.map(function (record) {
    var events = record.annotation_events || [];
    var success = events.filter(function (event) { return Number(event.event_key) >= 8; }).length;
    var failure = events.filter(function (event) { return Number(event.event_key) <= 7; }).length;
    return '<button type="button" class="rollout-card' + (record.id === state.selectedId ? " active" : "") + '" data-rollout-id="' + escapeHtml(record.id) + '">'
      + '<div class="badge-row">'
      + badge(record.annotation_status || "unreviewed", record.annotation_status || "")
      + (failure ? badge(failure + " fail", "failure") : "")
      + (success ? badge(success + " success", "success") : "")
      + '</div><div class="rollout-title">' + escapeHtml(taskLabel(record)) + "</div>"
      + '<div class="rollout-meta"><span>' + escapeHtml(record.id) + '</span><span>' + escapeHtml(record.total_frames || "?") + 'f</span></div>'
      + "</button>";
  }).join("");
  container.querySelectorAll("[data-rollout-id]").forEach(function (button) {
    button.addEventListener("click", function () { maybeSelectRollout(button.dataset.rolloutId); });
  });
}

function maybeSelectRollout(id) {
  if (state.dirty && id !== state.selectedId && !window.confirm("Discard unsaved interval changes?")) return;
  selectRollout(id);
}

function cameraKeys(record) {
  var paths = record && record.camera_video_paths;
  return paths && typeof paths === "object" ? Object.keys(paths) : [];
}

function chooseCamera(record) {
  var keys = cameraKeys(record);
  if (!keys.length) return "";
  var preferred = state.settings && state.settings.default_camera;
  if (preferred && keys.indexOf(preferred) >= 0) return preferred;
  if (record.observation_key && keys.indexOf(record.observation_key) >= 0) return record.observation_key;
  if (keys.indexOf("cam_high") >= 0) return "cam_high";
  return keys[0];
}

function selectRollout(id) {
  var record = state.rollouts.find(function (item) { return item.id === id; });
  if (!record) return;
  state.selectedId = id;
  state.currentFrame = 0;
  state.events = (record.annotation_events || []).map(function (event) { return Object.assign({}, event); });
  state.activeEvent = state.events.length ? 0 : null;
  state.dirty = false;
  state.videoGeneration += 1;
  byId("annotateEmpty").classList.add("hidden");
  byId("annotateContent").classList.remove("hidden");
  renderRolloutList();

  var cameraSelect = byId("annotateCamera");
  var keys = cameraKeys(record);
  cameraSelect.innerHTML = keys.map(function (key) { return '<option value="' + escapeHtml(key) + '">' + escapeHtml(key) + "</option>"; }).join("");
  cameraSelect.value = chooseCamera(record);
  byId("annotateTaskTitle").textContent = taskLabel(record);
  byId("annotateRecordMeta").textContent = record.id + " · " + (record.total_frames || "?") + " frames · " + (record.fps || "?") + " fps";
  byId("annotateBadges").innerHTML = badge("real robot", "") + badge(record.annotation_status || "unreviewed", record.annotation_status || "");

  var slider = byId("annotateFrameSlider");
  slider.max = Math.max(0, Number(record.total_frames || 1) - 1);
  slider.value = "0";
  loadAnnotateVideo();
  renderIntervals();
  updateFrameUi(0, true);
  updateNavigationButtons();
  setAnnotationMessage("", "");
}

function loadAnnotateVideo() {
  var record = selectedRollout();
  if (!record) return;
  var camera = byId("annotateCamera").value || chooseCamera(record);
  var video = byId("annotateVideo");
  state.videoGeneration += 1;
  var generation = state.videoGeneration;
  video.pause();
  video.src = "/api/videos/" + encodeURIComponent(record.id) + "?camera=" + encodeURIComponent(camera);
  video.load();
  byId("annotatePlay").textContent = "Play";
  if (typeof video.requestVideoFrameCallback === "function") {
    var watch = function (_now, metadata) {
      if (generation !== state.videoGeneration) return;
      if (!video.paused && !video.ended) {
        var fps = Number(record.fps) || 30;
        updateFrameUi(Math.round((metadata.mediaTime || video.currentTime) * fps), false);
      }
      video.requestVideoFrameCallback(watch);
    };
    video.requestVideoFrameCallback(watch);
  }
}

function clampFrame(frame) {
  var record = selectedRollout();
  var max = Math.max(0, Number(record && record.total_frames || 1) - 1);
  return Math.max(0, Math.min(max, Math.round(Number(frame) || 0)));
}

function seekFrame(frame) {
  var record = selectedRollout();
  if (!record) return;
  var value = clampFrame(frame);
  var fps = Number(record.fps) || 30;
  state.currentFrame = value;
  byId("annotateVideo").currentTime = value / fps;
  updateFrameUi(value, true);
}

function updateFrameUi(frame, forceTactile) {
  var record = selectedRollout();
  if (!record) return;
  var value = clampFrame(frame);
  var changed = value !== state.currentFrame;
  state.currentFrame = value;
  byId("annotateFrameSlider").value = String(value);
  byId("annotateFrameText").textContent = "Frame " + value + " / " + Math.max(0, Number(record.total_frames || 1) - 1);
  byId("annotateTimeText").textContent = formatTime(value / (Number(record.fps) || 30));
  if (changed || forceTactile) updateAnnotateTactile(value);
}

function updateAnnotateTactile(frame) {
  var record = selectedRollout();
  if (!record) return;
  var camera = byId("annotateCamera").value || chooseCamera(record);
  var kind = state.settings && state.settings.default_tactile_kind || "deform";
  var generation = ++state.tactileGeneration;
  var image = byId("annotateTactileSprite");
  image.onload = function () {
    if (generation === state.tactileGeneration) byId("annotateTactileStatus").textContent = camera + " · frame " + frame + " · " + kind;
  };
  image.onerror = function () {
    if (generation === state.tactileGeneration) byId("annotateTactileStatus").textContent = "No synchronized tactile at this frame";
  };
  image.src = "/api/tactile/" + encodeURIComponent(record.id) + "/sprite?camera=" + encodeURIComponent(camera)
    + "&frame=" + frame + "&kind=" + encodeURIComponent(kind);
  for (var offset = 1; offset <= 2; offset += 1) {
    var preload = new Image();
    preload.src = "/api/tactile/" + encodeURIComponent(record.id) + "/sprite?camera=" + encodeURIComponent(camera)
      + "&frame=" + clampFrame(frame + offset) + "&kind=" + encodeURIComponent(kind);
  }
}

function togglePlay() {
  var video = byId("annotateVideo");
  if (video.paused || video.ended) {
    video.play().then(function () { byId("annotatePlay").textContent = "Pause"; }).catch(function () {});
  } else {
    video.pause();
    byId("annotatePlay").textContent = "Play";
  }
}

function eventOptions(selected) {
  return Object.keys(EVENT_TYPES).map(function (key) {
    var spec = EVENT_TYPES[key];
    return '<option value="' + key + '"' + (Number(selected) === Number(key) ? " selected" : "") + '>' + key + " · " + escapeHtml(spec.label) + "</option>";
  }).join("");
}

function normalizeLocalEvents() {
  state.events.forEach(function (event, index) {
    event.event_index = index;
    event.event_key = Number(event.event_key) || 6;
    event.start_frame = clampFrame(event.start_frame);
    event.end_frame = clampFrame(event.end_frame);
    if (event.end_frame < event.start_frame) event.end_frame = event.start_frame;
  });
}

function renderIntervals() {
  normalizeLocalEvents();
  var container = byId("intervalList");
  if (!state.events.length) {
    container.innerHTML = '<div class="empty-card" style="min-height:105px;padding:14px"><span>No intervals yet. Add one at the current frame.</span></div>';
    return;
  }
  container.innerHTML = state.events.map(function (event, index) {
    return '<div class="interval-row' + (index === state.activeEvent ? " active" : "") + '" data-event-index="' + index + '">'
      + '<div class="interval-index">' + (index + 1) + "</div>"
      + '<label class="interval-field"><span>Type</span><select class="select-input event-type">' + eventOptions(event.event_key) + "</select></label>"
      + '<label class="interval-field"><span>Start</span><input class="interval-number event-start" type="number" min="0" value="' + event.start_frame + '"><div class="interval-frame-tools"><button type="button" class="use-start">Use current (Q)</button></div></label>'
      + '<label class="interval-field"><span>End</span><input class="interval-number event-end" type="number" min="0" value="' + event.end_frame + '"><div class="interval-frame-tools"><button type="button" class="use-end">Use current (W)</button></div></label>'
      + '<div class="row-actions interval-actions"><button type="button" class="button secondary jump-start">Start</button><button type="button" class="button secondary jump-end">End</button><button type="button" class="button danger delete-event">Delete</button></div>'
      + "</div>";
  }).join("");
  container.querySelectorAll(".interval-row").forEach(function (row) {
    var index = Number(row.dataset.eventIndex);
    row.addEventListener("click", function () {
      state.activeEvent = index;
      container.querySelectorAll(".interval-row").forEach(function (node) { node.classList.toggle("active", Number(node.dataset.eventIndex) === index); });
    });
    row.querySelector(".event-type").addEventListener("change", function (event) { state.events[index].event_key = Number(event.target.value); markDirty(); });
    row.querySelector(".event-start").addEventListener("change", function (event) { state.events[index].start_frame = clampFrame(event.target.value); if (state.events[index].end_frame < state.events[index].start_frame) state.events[index].end_frame = state.events[index].start_frame; markDirty(); renderIntervals(); });
    row.querySelector(".event-end").addEventListener("change", function (event) { state.events[index].end_frame = Math.max(state.events[index].start_frame, clampFrame(event.target.value)); markDirty(); renderIntervals(); });
    row.querySelector(".use-start").addEventListener("click", function () { state.activeEvent = index; state.events[index].start_frame = state.currentFrame; if (state.events[index].end_frame < state.currentFrame) state.events[index].end_frame = state.currentFrame; markDirty(); renderIntervals(); });
    row.querySelector(".use-end").addEventListener("click", function () { state.activeEvent = index; state.events[index].end_frame = Math.max(state.events[index].start_frame, state.currentFrame); markDirty(); renderIntervals(); });
    row.querySelector(".jump-start").addEventListener("click", function () { seekFrame(state.events[index].start_frame); });
    row.querySelector(".jump-end").addEventListener("click", function () { seekFrame(state.events[index].end_frame); });
    row.querySelector(".delete-event").addEventListener("click", function () { state.events.splice(index, 1); if (!state.events.length) state.activeEvent = null; else state.activeEvent = Math.min(index, state.events.length - 1); markDirty(); renderIntervals(); });
  });
}

function addInterval(key) {
  var eventKey = Number(key) || 6;
  state.events.push({ event_key: eventKey, start_frame: state.currentFrame, end_frame: state.currentFrame, notes: "" });
  state.activeEvent = state.events.length - 1;
  markDirty();
  renderIntervals();
}

function setActiveEventType(key) {
  if (state.activeEvent == null || !state.events[state.activeEvent]) {
    addInterval(key);
    return;
  }
  state.events[state.activeEvent].event_key = Number(key);
  markDirty();
  renderIntervals();
}

function setActiveBoundary(which) {
  if (state.activeEvent == null || !state.events[state.activeEvent]) return;
  var event = state.events[state.activeEvent];
  if (which === "start") {
    event.start_frame = state.currentFrame;
    if (event.end_frame < event.start_frame) event.end_frame = event.start_frame;
  } else {
    event.end_frame = Math.max(event.start_frame, state.currentFrame);
  }
  markDirty();
  renderIntervals();
}

function markDirty() {
  state.dirty = true;
  setAnnotationMessage("Unsaved changes", "");
}
function setAnnotationMessage(text, kind) {
  var node = byId("annotationMessage");
  node.textContent = text || "";
  node.className = "message" + (kind ? " " + kind : "");
}

async function saveIntervals() {
  var record = selectedRollout();
  if (!record) return;
  normalizeLocalEvents();
  byId("saveIntervals").disabled = true;
  setAnnotationMessage("Saving…", "");
  try {
    var payload = await jsonRequest("/api/annotations/" + encodeURIComponent(record.id), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ events: state.events })
    });
    state.events = (payload.events || []).map(function (event) { return Object.assign({}, event); });
    record.annotation_events = state.events.map(function (event) { return Object.assign({}, event); });
    record.annotation_status = state.events.length ? "complete" : "unreviewed";
    state.annotationTarget = payload.annotation_target || state.annotationTarget;
    state.dirty = false;
    renderIntervals();
    renderRolloutList();
    updateHeader();
    setAnnotationMessage("Saved to " + (state.annotationTarget || "annotation file"), "success");
  } catch (error) {
    setAnnotationMessage(error.message, "error");
  } finally {
    byId("saveIntervals").disabled = false;
  }
}

function updateNavigationButtons() {
  var index = state.filtered.findIndex(function (record) { return record.id === state.selectedId; });
  byId("annotatePrevious").disabled = index <= 0;
  byId("annotateNext").disabled = index < 0 || index >= state.filtered.length - 1;
}
function navigateRollout(delta) {
  var index = state.filtered.findIndex(function (record) { return record.id === state.selectedId; });
  var target = state.filtered[index + delta];
  if (target) maybeSelectRollout(target.id);
}

async function loadRollouts() {
  var payload = await jsonRequest("/api/rollouts");
  state.rollouts = payload.rollouts || [];
  state.annotationSource = payload.annotation_source || null;
  state.annotationTarget = payload.annotation_target || null;
  populateTaskFilter();
  filterRollouts();
  updateHeader();
  if (state.filtered.length) selectRollout(state.filtered[0].id);
}

async function loadAnalysis() {
  var cards = byId("analysisCards");
  cards.innerHTML = '<div class="metric-card"><span class="muted">Loading…</span></div>';
  try {
    var data = await jsonRequest("/api/analysis");
    cards.innerHTML = [
      ["Rollouts", data.rollouts],
      ["Annotated", data.reviewed_rollouts],
      ["Coverage", formatPercent(data.coverage)],
      ["Intervals", data.events]
    ].map(function (item) {
      return '<div class="metric-card"><div class="eyebrow">' + escapeHtml(item[0]) + '</div><div class="metric-value">' + escapeHtml(item[1]) + "</div></div>";
    }).join("");
    byId("eventAnalysisTable").innerHTML = '<table class="data-table"><thead><tr><th>Key</th><th>Type</th><th>Outcome</th><th>Count</th><th>Mean duration</th></tr></thead><tbody>'
      + (data.event_types || []).map(function (row) {
        return "<tr><td>" + row.event_key + "</td><td>" + escapeHtml(row.label) + "</td><td>" + badge(row.outcome, row.outcome) + "</td><td>" + row.count + "</td><td>" + (row.mean_duration_frames == null ? "—" : row.mean_duration_frames.toFixed(1) + " f") + "</td></tr>";
      }).join("") + "</tbody></table>";
    byId("taskAnalysisTable").innerHTML = '<table class="data-table"><thead><tr><th>Task</th><th>Rollouts</th><th>Annotated</th><th>Coverage</th><th>Intervals</th></tr></thead><tbody>'
      + (data.tasks || []).map(function (row) {
        return "<tr><td>" + escapeHtml(row.task) + "</td><td>" + row.rollouts + "</td><td>" + row.reviewed + "</td><td>" + formatPercent(row.rollouts ? row.reviewed / row.rollouts : 0) + "</td><td>" + row.events + "</td></tr>";
      }).join("") + "</tbody></table>";
  } catch (error) {
    cards.innerHTML = '<div class="metric-card"><span class="message error">' + escapeHtml(error.message) + "</span></div>";
  }
}

async function loadRuns() {
  var container = byId("runsList");
  container.innerHTML = '<div class="muted">Loading tactile experiment outputs…</div>';
  try {
    var payload = await jsonRequest("/api/runs");
    byId("runsRoot").textContent = "Scanning: " + payload.runs_root;
    var runs = payload.runs || [];
    if (!runs.length) {
      container.innerHTML = '<div class="empty-card" style="min-height:140px"><span>No SHARPA output directories found under the configured runs root.</span></div>';
      return;
    }
    container.innerHTML = runs.map(function (run) {
      return '<div class="run-row"><div><div class="run-path">' + escapeHtml(run.path) + '</div><div class="muted" style="font-size:10px;margin-top:3px">' + escapeHtml(formatDate(run.modified_at)) + '</div></div><div class="run-markers">'
        + (run.markers || []).map(function (marker) { return badge(marker, ""); }).join("")
        + '</div><span class="badge">tactile</span></div>';
    }).join("");
  } catch (error) {
    container.innerHTML = '<div class="message error">' + escapeHtml(error.message) + "</div>";
  }
}

async function loadSettings() {
  try {
    var payload = await jsonRequest("/api/settings");
    var settings = payload.settings || {};
    state.settings = settings;
    applyTheme(settings.theme);
    byId("settingsManifest").value = payload.manifest_path || "";
    byId("settingsAnnotations").value = settings.annotations_path || "";
    byId("settingsSeedGlob").value = settings.annotation_seed_glob || "";
    byId("settingsRunsRoot").value = settings.runs_root || "";
    byId("settingsCamera").value = settings.default_camera || "cam_high";
    byId("settingsKind").value = settings.default_tactile_kind || "deform";
    byId("settingsTheme").value = settings.theme || "light";
    byId("settingsSource").textContent = payload.annotation_source || "None";
  } catch (error) {
    var node = byId("settingsMessage"); node.textContent = error.message; node.className = "message error";
  }
}

async function saveSettings(event) {
  event.preventDefault();
  var payload = {
    annotations_path: byId("settingsAnnotations").value.trim(),
    annotation_seed_glob: byId("settingsSeedGlob").value.trim(),
    runs_root: byId("settingsRunsRoot").value.trim(),
    default_camera: byId("settingsCamera").value.trim(),
    default_tactile_kind: byId("settingsKind").value,
    theme: byId("settingsTheme").value
  };
  var message = byId("settingsMessage");
  try {
    var result = await jsonRequest("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
    });
    state.settings = result.settings;
    applyTheme(state.settings.theme);
    message.textContent = "Settings saved."; message.className = "message success";
    await loadSettings();
  } catch (error) {
    message.textContent = error.message; message.className = "message error";
  }
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach(function (button) {
    button.addEventListener("click", function () { switchView(button.dataset.view); });
  });
  byId("annotateSearch").addEventListener("input", filterRollouts);
  byId("annotateTaskFilter").addEventListener("change", filterRollouts);
  byId("annotateReviewFilter").addEventListener("change", filterRollouts);
  byId("annotatePrevious").addEventListener("click", function () { navigateRollout(-1); });
  byId("annotateNext").addEventListener("click", function () { navigateRollout(1); });
  byId("annotateCamera").addEventListener("change", function () { loadAnnotateVideo(); seekFrame(0); });
  byId("annotatePlay").addEventListener("click", togglePlay);
  byId("annotateStepBack").addEventListener("click", function () { seekFrame(state.currentFrame - 1); });
  byId("annotateStepForward").addEventListener("click", function () { seekFrame(state.currentFrame + 1); });
  byId("annotateFrameSlider").addEventListener("input", function (event) { seekFrame(event.target.value); });
  byId("annotateVideo").addEventListener("play", function () { byId("annotatePlay").textContent = "Pause"; });
  byId("annotateVideo").addEventListener("pause", function () { byId("annotatePlay").textContent = "Play"; });
  byId("annotateVideo").addEventListener("timeupdate", function () {
    var record = selectedRollout(); if (!record) return;
    updateFrameUi(Math.round(byId("annotateVideo").currentTime * (Number(record.fps) || 30)), false);
  });
  byId("addInterval").addEventListener("click", function () { addInterval(6); });
  byId("saveIntervals").addEventListener("click", saveIntervals);
  byId("refreshRuns").addEventListener("click", loadRuns);
  byId("refreshAnalysis").addEventListener("click", loadAnalysis);
  byId("settingsForm").addEventListener("submit", saveSettings);
  window.addEventListener("beforeunload", function (event) {
    if (state.dirty) { event.preventDefault(); event.returnValue = ""; }
  });
  window.addEventListener("keydown", function (event) {
    if (state.view !== "annotate" || isTypingTarget(event.target)) return;
    if (event.key === " ") { event.preventDefault(); togglePlay(); return; }
    if (event.key === ",") { event.preventDefault(); navigateRollout(-1); return; }
    if (event.key === ".") { event.preventDefault(); navigateRollout(1); return; }
    if (event.key === "ArrowLeft") { event.preventDefault(); seekFrame(state.currentFrame - 1); return; }
    if (event.key === "ArrowRight") { event.preventDefault(); seekFrame(state.currentFrame + 1); return; }
    if (event.key === "ArrowUp" && state.events.length) { event.preventDefault(); state.activeEvent = state.activeEvent == null ? 0 : Math.max(0, state.activeEvent - 1); renderIntervals(); return; }
    if (event.key === "ArrowDown" && state.events.length) { event.preventDefault(); state.activeEvent = state.activeEvent == null ? 0 : Math.min(state.events.length - 1, state.activeEvent + 1); renderIntervals(); return; }
    if (["1", "2", "3", "4"].indexOf(event.key) >= 0) { event.preventDefault(); setActiveEventType(5 + Number(event.key)); return; }
    if (event.key.toLowerCase() === "q") { event.preventDefault(); setActiveBoundary("start"); return; }
    if (event.key.toLowerCase() === "w") { event.preventDefault(); setActiveBoundary("end"); }
  });
}

async function init() {
  bindEvents();
  var requested = (location.hash || "#annotate").slice(1);
  if (!byId("view-" + requested)) requested = "annotate";
  switchView(requested, false);
  try {
    var settingsPayload = await jsonRequest("/api/settings");
    state.settings = settingsPayload.settings || {};
    state.annotationSource = settingsPayload.annotation_source || null;
    state.annotationTarget = settingsPayload.annotation_target || null;
    applyTheme(state.settings.theme);
    await loadRollouts();
    setGlobalStatus(true, "Dataset online");
  } catch (error) {
    setGlobalStatus(false, error.message);
    byId("datasetStatus").textContent = "Dataset unavailable";
  }
}

init();
