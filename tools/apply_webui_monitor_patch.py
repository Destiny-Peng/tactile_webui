from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "webui/server.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


text = SERVER.read_text(encoding="utf-8")
text = replace_once(text, "import tempfile\nimport threading\n", "import tempfile\nimport threading\nimport time\n", "time import")
text = replace_once(
    text,
    '''try:  # package import for tests\n    from .tactile_service import FailRecoveryTactileService\nexcept ImportError:  # direct ``python webui/server.py`` execution\n    from tactile_service import FailRecoveryTactileService\n''',
    '''try:  # package import for tests\n    from .monitoring import WebUIMonitor\n    from .results_service import OnlineResultsService\n    from .tactile_service import FailRecoveryTactileService\nexcept ImportError:  # direct ``python webui/server.py`` execution\n    from monitoring import WebUIMonitor\n    from results_service import OnlineResultsService\n    from tactile_service import FailRecoveryTactileService\n''',
    "service imports",
)
text = replace_once(
    text,
    '''        self._annotation_lock = threading.RLock()\n        self._configure_data_source()\n''',
    '''        self._annotation_lock = threading.RLock()\n        self.monitor = WebUIMonitor(self.root / "logs" / "webui")\n        self.results = OnlineResultsService(self.root)\n        self._configure_data_source()\n''',
    "app services",
)
text = replace_once(
    text,
    '''    CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)\n\n    def _safe_write(self, data: bytes) -> bool:\n''',
    '''    CLIENT_DISCONNECT_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)\n\n    def _begin_request(self, method: str, path: str) -> None:\n        self._request_started = time.perf_counter()\n        self._request_method = method\n        self._request_path = path\n        self._request_status = 500\n        self._request_resource = "other"\n        self._response_bytes = None\n        self._client_disconnected = False\n        self._video_total_bytes = None\n        self._video_range = None\n        self._request_recorded = False\n\n    def send_response(self, code: int, message: str | None = None) -> None:\n        self._request_status = int(code)\n        super().send_response(code, message)\n\n    def finish(self) -> None:\n        try:\n            super().finish()\n        finally:\n            started = getattr(self, "_request_started", None)\n            if started is None or getattr(self, "_request_recorded", False):\n                return\n            self._request_recorded = True\n            try:\n                self.app.monitor.record_request(\n                    method=getattr(self, "_request_method", "?"),\n                    path=getattr(self, "_request_path", "?"),\n                    status=getattr(self, "_request_status", 500),\n                    resource=getattr(self, "_request_resource", "other"),\n                    duration_ms=round((time.perf_counter() - started) * 1000.0, 3),\n                    response_bytes=getattr(self, "_response_bytes", None),\n                    client_disconnected=bool(getattr(self, "_client_disconnected", False)),\n                    range=getattr(self, "_video_range", None),\n                    video_total_bytes=getattr(self, "_video_total_bytes", None),\n                )\n            except OSError:\n                pass\n\n    def log_message(self, format: str, *args: Any) -> None:\n        # Structured request logging is handled by WebUIMonitor in finish().\n        return\n\n    def _safe_write(self, data: bytes) -> bool:\n''',
    "handler monitoring",
)
text = replace_once(
    text,
    '''        except self.CLIENT_DISCONNECT_ERRORS:\n            self.close_connection = True\n            return False\n''',
    '''        except self.CLIENT_DISCONNECT_ERRORS:\n            self.close_connection = True\n            self._client_disconnected = True\n            return False\n''',
    "disconnect flag",
)
text = replace_once(
    text,
    '''    def json_response(self, status: int, payload: Any) -> None:\n        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")\n''',
    '''    def json_response(self, status: int, payload: Any) -> None:\n        self._request_resource = "api"\n        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")\n        self._response_bytes = len(body)\n''',
    "json metrics",
)
text = replace_once(
    text,
    '''    def serve_static(self, path: Path) -> None:\n        if not path.is_file():\n''',
    '''    def serve_static(self, path: Path) -> None:\n        self._request_resource = "static"\n        if not path.is_file():\n''',
    "static metrics",
)
text = replace_once(text, '''        body = path.read_bytes()\n        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"\n''', '''        body = path.read_bytes()\n        self._response_bytes = len(body)\n        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"\n''', "static size")
text = replace_once(
    text,
    '''    def serve_png(self, body: bytes) -> None:\n        self.send_response(HTTPStatus.OK)\n''',
    '''    def serve_png(self, body: bytes) -> None:\n        self._request_resource = "tactile"\n        self._response_bytes = len(body)\n        self.send_response(HTTPStatus.OK)\n''',
    "png metrics",
)
text = replace_once(
    text,
    '''    def serve_video(self, path: Path) -> None:\n        if not path.is_file():\n''',
    '''    def serve_video(self, path: Path) -> None:\n        self._request_resource = "video"\n        if not path.is_file():\n''',
    "video metrics",
)
text = replace_once(
    text,
    '''        length = end - start + 1\n        self.send_response(status)\n''',
    '''        length = end - start + 1\n        self._response_bytes = length\n        self._video_total_bytes = size\n        self._video_range = [start, end] if range_header else None\n        self.send_response(status)\n''',
    "video sizes",
)
text = replace_once(
    text,
    '''    def do_GET(self) -> None:\n        parsed = urlparse(self.path)\n        path = unquote(parsed.path)\n        query = parse_qs(parsed.query, keep_blank_values=True)\n        try:\n''',
    '''    def do_GET(self) -> None:\n        parsed = urlparse(self.path)\n        path = unquote(parsed.path)\n        query = parse_qs(parsed.query, keep_blank_values=True)\n        self._begin_request("GET", path)\n        try:\n''',
    "GET begin",
)
text = replace_once(
    text,
    '''            if path == "/api/runs":\n                self.json_response(HTTPStatus.OK, {"runs": self.app.list_runs(), "runs_root": self.app.settings["runs_root"]})\n                return\n            if path == "/api/settings":\n''',
    '''            if path == "/api/runs":\n                self.json_response(HTTPStatus.OK, {"runs": self.app.list_runs(), "runs_root": self.app.settings["runs_root"]})\n                return\n            if path == "/api/results/sources":\n                self.json_response(\n                    HTTPStatus.OK,\n                    {"sources": self.app.results.list_sources(self.app.settings["runs_root"])},\n                )\n                return\n            if path == "/api/results/curve":\n                source = str(query.get("source", [""])[0] or "").strip()\n                rollout_id = str(query.get("rollout_id", [""])[0] or "").strip()\n                if not source or not rollout_id:\n                    raise ValueError("source and rollout_id are required")\n                if rollout_id not in self.app.rollout_map():\n                    self.json_error(HTTPStatus.NOT_FOUND, "Unknown rollout")\n                    return\n                result = self.app.results.curve(self.app.settings["runs_root"], source, rollout_id)\n                result["annotation_events"] = self.app.annotations_by_rollout().get(rollout_id, [])\n                self.json_response(HTTPStatus.OK, result)\n                return\n            if path == "/api/diagnostics":\n                snapshot = self.app.monitor.snapshot()\n                snapshot.update(\n                    {\n                        "manifest_exists": self.app.manifest_path.is_file(),\n                        "manifest": self.app._display_path(self.app.manifest_path, self.app.source_root),\n                        "source_project_root": str(self.app.source_root),\n                    }\n                )\n                self.json_response(HTTPStatus.OK, snapshot)\n                return\n            if path == "/api/settings":\n''',
    "GET result/diagnostics routes",
)
text = replace_once(
    text,
    '''    def do_POST(self) -> None:\n        parsed = urlparse(self.path)\n        path = unquote(parsed.path)\n        try:\n            payload = self._read_json_body()\n            if path == "/api/settings":\n''',
    '''    def do_POST(self) -> None:\n        parsed = urlparse(self.path)\n        path = unquote(parsed.path)\n        self._begin_request("POST", path)\n        try:\n            payload = self._read_json_body()\n            if path == "/api/telemetry":\n                events = payload.get("events")\n                if not isinstance(events, list):\n                    raise ValueError("events must be a list")\n                accepted = self.app.monitor.record_client_events(events)\n                self.json_response(HTTPStatus.OK, {"accepted": accepted})\n                return\n            if path == "/api/settings":\n''',
    "POST telemetry route",
)
text = replace_once(
    text,
    '''    print(f"Annotation target: {app.annotation_target_path()}", flush=True)\n    try:\n''',
    '''    print(f"Annotation target: {app.annotation_target_path()}", flush=True)\n    print(f"WebUI logs: {app.monitor.root}", flush=True)\n    app.monitor.record_server_event("server_start", host=args.host, port=args.port)\n    try:\n''',
    "startup log",
)
SERVER.write_text(text, encoding="utf-8")

# Keep runtime logs namespaced and out of source control.
gitignore = ROOT / ".gitignore"
ignore = gitignore.read_text(encoding="utf-8")
if "logs/\n" not in ignore:
    ignore += "\n# Runtime diagnostics / logs\nlogs/\n"
    gitignore.write_text(ignore, encoding="utf-8")

# The visual timeline uses a nested track area so frame percentages are exact
# and unaffected by the left-side lane labels.
css = ROOT / "webui/static/shell.css"
css_text = css.read_text(encoding="utf-8")
needle = '.precision-timeline.compact-timeline { min-height: 142px; }\n'
insert = needle + '.timeline-track-area { position: relative; min-height: 134px; margin-left: 34px; }\n.compact-timeline .timeline-track-area { min-height: 96px; }\n'
if needle not in css_text:
    raise SystemExit("timeline CSS insertion point missing")
css_text = css_text.replace(needle, insert, 1)
css.write_text(css_text, encoding="utf-8")

# Validate the newly-added Python modules in CI as well as the migration shell.
workflow = ROOT / ".github/workflows/validate-tactile-webui.yml"
workflow_text = workflow.read_text(encoding="utf-8")
if "webui/monitoring.py" not in workflow_text:
    workflow_text = workflow_text.replace("      - 'webui/**'\n", "      - 'webui/**'\n      - 'tools/**'\n")
workflow.write_text(workflow_text, encoding="utf-8")
