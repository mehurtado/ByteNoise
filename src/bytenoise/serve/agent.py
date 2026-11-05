"""HTTP agent + UI host (stdlib only). Binds loopback by default.

Security model: it can start training processes and read run files, so
  * default bind is 127.0.0.1; expose it to the tailnet with `tailscale serve` (never funnel);
  * Host must be in the allowlist (blocks DNS rebinding), Origin if present must match Host (blocks cross-site POSTs);
  * mutating requests must be application/json (a cross-site form or no-cors fetch cannot send that without a preflight we never answer);
  * if a token is configured every /api call needs `Authorization: Bearer <token>`; with `--tailnet-users` a request from
    loopback carrying an allowlisted Tailscale-User-Login header (set by `tailscale serve`) is also accepted;
  * names from the network are only ever resolved through allowlisted lookups (configs/<kind>/<name>.yaml, runs/<id>), never as paths.
"""
import hmac
import json
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import __version__
from ..core.config import list_refs
from ..datasets.store import list_datasets
from ..evalx import report
from ..evalx.harness import list_modes, run_suite
from ..evalx.protocol import load_protocol
from ..models.loading import load_run_model
from ..models.registry import get_model_class, list_models
from ..runstore import read_metrics, refresh, safe_run_path
from . import jobs

UI_DIR = Path(__file__).resolve().parent / "ui"
UI_FILES = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
CTYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
MAX_BODY = 1 << 20
_WHICH = ("ema", "raw")
_DEVICES = ("cpu", "cuda")


class ApiError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code, self.msg = code, msg


class Models:
    """One-slot model cache. Model operations are serialised: this box has one GPU and 32 GB of RAM."""

    def __init__(self):
        self.lock = threading.Lock()
        self.key = None
        self.model = None

    def get(self, run_id, which, device):
        key = (run_id, which, device)
        if key != self.key:
            self.model = None  # free before loading the next
            self.model = load_run_model(safe_run_path(run_id), which, device)
            self.key = key
        return self.model


class State:
    def __init__(self, token=None, allow_hosts=(), tailnet_users=()):
        self.token = token
        self.allow_hosts = {h.lower() for h in allow_hosts}
        self.tailnet_users = {u.lower() for u in tailnet_users}
        self.models = Models()


def _need(body, key, typ=str):
    v = body.get(key)
    if not isinstance(v, typ) or isinstance(v, bool) and typ is not bool:
        raise ApiError(400, f"`{key}` must be {typ.__name__}")
    return v


def _opt_choice(body, key, choices, default):
    v = body.get(key, default)
    if v not in choices:
        raise ApiError(400, f"`{key}` must be one of {list(choices)}")
    return v


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"bytenoise/{__version__}"

        def log_message(self, fmt, *args):
            pass

        # --- plumbing ---
        def _send(self, code, payload, ctype="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if ctype.startswith("text/html"):
                self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'")
            self.end_headers()
            self.wfile.write(data)

        def _host_ok(self):
            host = (self.headers.get("Host") or "").lower()
            if host not in state.allow_hosts:
                return False
            origin = self.headers.get("Origin")
            if origin:
                o = urllib.parse.urlparse(origin)
                return o.netloc.lower() == host
            return True

        def _authed(self):
            if state.token is None and not state.tailnet_users:
                return True
            auth = self.headers.get("Authorization") or ""
            if state.token and auth.startswith("Bearer ") and hmac.compare_digest(auth[7:], state.token):
                return True
            user = (self.headers.get("Tailscale-User-Login") or "").lower()
            return bool(user and user in state.tailnet_users and self.client_address[0] in ("127.0.0.1", "::1"))

        def _body(self):
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                raise ApiError(415, "content-type must be application/json")
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ApiError(413, "body too large")
            try:
                v = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                raise ApiError(400, "invalid json")
            if not isinstance(v, dict):
                raise ApiError(400, "body must be an object")
            return v

        def _dispatch(self, method):
            url = urllib.parse.urlparse(self.path)
            q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
            try:
                if not self._host_ok():
                    raise ApiError(403, "host/origin not allowed")
                if method == "GET" and url.path in UI_FILES:
                    f = UI_DIR / UI_FILES[url.path]
                    return self._send(200, f.read_bytes(), CTYPES[f.suffix])
                if not url.path.startswith("/api/"):
                    raise ApiError(404, "not found")
                if not self._authed():
                    raise ApiError(401, "unauthorised")
                body = self._body() if method == "POST" else {}
                out = route(state, method, url.path[5:].strip("/").split("/"), q, body)
                self._send(200, out)
            except ApiError as e:
                self._send(e.code, {"error": e.msg})
            except (FileNotFoundError, ValueError, KeyError, NotImplementedError) as e:
                self._send(400, {"error": f"{type(e).__name__}: {e}"})
            except Exception as e:  # keep the server alive; the detail stays server-side except the type
                self._send(500, {"error": f"internal error ({type(e).__name__})"})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    return Handler


def route(state, method, parts, q, body):
    head = parts[0]
    n = len(parts)
    if method == "GET":
        if head == "health":
            return {"ok": True, "version": __version__}
        if head == "models":
            return [{"family": f, "capabilities": sorted(get_model_class(f).capabilities), "sampler": get_model_class(f).sampler_schema()} for f in list_models()]
        if head == "runs" and n == 1:
            infos, stats = refresh()
            infos.sort(key=lambda i: i.mtime, reverse=True)
            return {"stats": stats, "runs": [i.to_dict() for i in infos]}
        if head == "runs" and n == 2:
            from ..runstore import describe_run
            p = safe_run_path(parts[1])
            info = describe_run(p)
            if info is None:
                raise ApiError(404, "not a run directory")
            cfg_f = (p / "config.resolved.json") if (p / "config.resolved.json").exists() else None
            return {"info": info.to_dict(), "config": json.loads(cfg_f.read_text(encoding="utf-8")) if cfg_f else None,
                    "evals": report.list_results(p.name)}
        if head == "runs" and n == 3 and parts[2] == "metrics":
            return read_metrics(safe_run_path(parts[1]), since=int(q.get("since", 0)), max_points=min(int(q.get("max", 1500)), 5000))
        if head == "datasets":
            return list_datasets()
        if head == "protocols":
            return list_refs("protocols")
        if head == "experiments":
            return list_refs("experiments")
        if head == "modes":
            return list_modes()
        if head == "evals" and n == 1:
            return report.list_results(q.get("run"))
        if head == "evals" and n == 3:
            return report.load(parts[1], parts[2])
        if head == "gpu":
            return jobs.gpu_info()
        if head == "jobs" and n == 1:
            return jobs.list_jobs()
        if head == "jobs" and n == 2:
            return jobs.get(parts[1])
        if head == "jobs" and n == 3 and parts[2] == "log":
            return {"log": jobs.log_tail(parts[1])}
    else:
        if head == "jobs" and n == 1:
            sets = body.get("sets", [])
            if not isinstance(sets, list) or not all(isinstance(s, str) for s in sets):
                raise ApiError(400, "`sets` must be a list of 'key=value' strings")
            job = jobs.submit(_need(body, "experiment"), sets, _opt_choice(body, "device", ("cpu", "cuda", "auto"), "cpu"))
            blocked = jobs.schedule()
            out = jobs.get(job["id"])
            out["blocked"] = blocked if out["status"] == "queued" else None
            return out
        if head == "jobs" and n == 3 and parts[2] == "stop":
            out = jobs.stop(parts[1])
            jobs.schedule()
            return out
        if head == "schedule":
            return {"blocked": jobs.schedule()}
        if head == "generate":
            run, which = _need(body, "run"), _opt_choice(body, "which", _WHICH, "ema")
            device = _opt_choice(body, "device", _DEVICES, "cpu")
            with state.models.lock:
                m = state.models.get(run, which, device)
                prompt = _need(body, "prompt").encode("utf-8") if "prompt" in body else b""
                n_bytes = int(body.get("n_bytes", 128))
                if not 1 <= n_bytes <= 4096:
                    raise ApiError(400, "n_bytes must be in [1, 4096]")
                out = m.generate(prompt, n_bytes, seed=int(body.get("seed", 0)), **(body.get("sampler") or {}))
            return {"text": out.decode("utf-8", "replace"), "repr": ascii(out), "model": m.describe()}
        if head == "eval":
            run, which = _need(body, "run"), _opt_choice(body, "which", _WHICH, "ema")
            device = _opt_choice(body, "device", _DEVICES, "cpu")
            proto = load_protocol(_need(body, "protocol"), body.get("data"))
            with state.models.lock:
                m = state.models.get(run, which, device)
                res = run_suite(m, proto, device, only=body.get("modes"))
            path = report.save(res, safe_run_path(run).name, which)
            return {"saved": path.name, "result": res}
    raise ApiError(404, "no such endpoint")


def make_server(host="127.0.0.1", port=8765, token=None, allow_hosts=(), tailnet_users=()):
    """Bind and return (httpd, state); port 0 picks a free port."""
    if host not in ("127.0.0.1", "::1", "localhost") and not token and not tailnet_users:
        raise SystemExit("refusing to bind a non-loopback address without --token or --tailnet-users")
    state = State(token, (), tailnet_users)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    real = httpd.server_address[1]
    state.allow_hosts = {f"127.0.0.1:{real}", f"localhost:{real}", *(h.lower() for h in allow_hosts)}
    return httpd, state


def serve(host="127.0.0.1", port=8765, token=None, allow_hosts=(), tailnet_users=(), poll_s=5.0):
    httpd, state = make_server(host, port, token, allow_hosts, tailnet_users)

    def pump():  # start queued jobs as the GPU frees up
        while True:
            try:
                jobs.schedule()
            except Exception:
                pass
            time.sleep(poll_s)

    threading.Thread(target=pump, daemon=True).start()
    print(f"bytenoise agent on http://{host}:{port}  (allowed hosts: {sorted(state.allow_hosts)})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
