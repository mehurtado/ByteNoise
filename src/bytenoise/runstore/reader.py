"""Read run directories of every vintage without modifying them. Never loads checkpoints."""
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..core.paths import runs_dir

ACTIVE_WINDOW_S = 300
NATIVE_STALE_S = 1800   # native runs may log rarely; only call them dead after 30 minutes of silence
RESUME_SLACK_S = 5      # metrics newer than status.json by more than this means the run was resumed after it "stopped"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MARKERS = ("run.json", "config.json", "metrics.jsonl", "status.json", "meta.json")


@dataclass
class RunInfo:
    id: str
    kind: str = "legacy"          # native | legacy
    status: str = "unknown"       # running | done | failed | stopped | incomplete | unknown
    family: str = ""
    arch: str = ""
    denoiser: str = ""
    data: str = ""
    step: int | None = None
    target_steps: int | None = None
    elapsed_s: float | None = None
    params: int | None = None
    last_loss: float | None = None
    git: str = ""
    parent: str = ""
    experiment: str = ""
    has_checkpoint: bool = False
    mtime: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def safe_run_path(run_id):
    """Resolve a run id (from an HTTP request or CLI) to a directory inside runs/, or raise."""
    if not isinstance(run_id, str) or not _ID.match(run_id) or run_id in (".", ".."):
        raise ValueError(f"invalid run id {run_id!r}")
    p = (runs_dir() / run_id).resolve()
    if p.parent != runs_dir().resolve() or not p.is_dir():
        raise FileNotFoundError(f"no such run {run_id!r}")
    return p


def payload_dir(path):
    """Where trainer files (checkpoint.pt, metrics.jsonl, config.json, meta.json) live: <run>/train for native runs."""
    path = Path(path)
    sub = path / "train"
    return sub if (path / "run.json").exists() and sub.is_dir() else path


def _json(path):
    try:
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        return v if isinstance(v, dict) else {}
    except (OSError, ValueError):
        return {}


def tail_rows(path, n=5, chunk=1 << 16):
    """Last n parseable json rows of a jsonl file, reading only the tail."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - chunk))
            raw = f.read()
    except OSError:
        return []
    lines = raw.splitlines()
    if size > chunk:
        lines = lines[1:]  # first line may be cut
    rows = []
    for ln in reversed(lines):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if isinstance(r, dict):
            rows.append(r)
            if len(rows) >= n:
                break
    return rows[::-1]


def _step(row):
    for k in ("update", "step"):
        if isinstance(row.get(k), (int, float)):
            return int(row[k])
    return None


def describe_run(path):
    """RunInfo for a directory, or None if it has no recognisable run files."""
    path = Path(path)
    top = path
    path = payload_dir(path)
    present = {m: (path / m) for m in MARKERS if (path / m).exists()}
    if (top / "run.json").exists():
        present["run.json"] = top / "run.json"
    if not present:
        return None
    info = RunInfo(id=top.name)
    manifest = _json(top / "run.json") if "run.json" in present else {}
    cfg = _json(path / "config.json")
    meta = _json(path / "meta.json")
    status = _json(path / "status.json")
    rows = tail_rows(path / "metrics.jsonl") if "metrics.jsonl" in present else []
    train_rows = [r for r in rows if not r.get("eval")]
    last = train_rows[-1] if train_rows else {}

    if manifest:
        info.kind = "native"
        info.status = manifest.get("status", "unknown")
        info.family = manifest.get("model", {}).get("family", "")
        info.parent = manifest.get("parent") or ""
        info.experiment = manifest.get("experiment", "")
        info.git = manifest.get("git", "")
    info.arch = cfg.get("arch", "")
    info.denoiser = (cfg.get("model") or {}).get("denoiser", "") if info.arch == "diffusion" else ""
    train_cfg = cfg.get("train") or {}
    info.data = train_cfg.get("data") or train_cfg.get("data_dir") or meta.get("dataset", "")
    info.target_steps = train_cfg.get("max_updates") if isinstance(train_cfg.get("max_updates"), int) else None
    info.params = meta.get("param_count") or meta.get("params")
    info.git = info.git or meta.get("git_revision", "")
    info.step = _step(last) if last and _step(last) is not None else status.get("updates")
    info.elapsed_s = last.get("elapsed") if isinstance(last.get("elapsed"), (int, float)) else status.get("elapsed")
    loss = last.get("loss")
    info.last_loss = float(loss) if isinstance(loss, (int, float)) else None
    info.has_checkpoint = (path / "checkpoint.pt").exists() or any(path.glob("state_*.pt")) or any(path.glob("model_*.pt"))

    newest = max((p.stat().st_mtime for p in present.values()), default=0.0)
    info.mtime = newest
    now = time.time()
    metrics_m = present["metrics.jsonl"].stat().st_mtime if "metrics.jsonl" in present else 0.0
    status_m = present["status.json"].stat().st_mtime if "status.json" in present else 0.0
    if info.kind == "native":
        started = manifest.get("started") or manifest.get("created") or 0.0
        if info.status in ("running", "created") and now - max(metrics_m, started) > NATIVE_STALE_S:
            info.status = "incomplete"  # trainer was killed without updating its manifest
    elif info.kind == "legacy":
        if "stopped" in status and metrics_m - status_m < RESUME_SLACK_S:
            info.status = {"max_updates": "done", "max_seconds": "done"}.get(status["stopped"], "stopped")
            if status["stopped"] == "max_seconds" and info.target_steps and (info.step or 0) < info.target_steps:
                info.status = "timed_out"  # wall-clock budget ended it before max_updates
        elif "metrics.jsonl" in present and now - newest < ACTIVE_WINDOW_S:
            info.status = "running"
        else:
            info.status = "incomplete"
    return info


def scan_runs():
    """(infos, skipped) over runs/. Skipped = names that are not run directories, with a reason."""
    root = runs_dir()
    infos, skipped = [], []
    if not root.exists():
        return infos, skipped
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            skipped.append((p.name, "not a directory"))
            continue
        try:
            info = describe_run(p)
        except Exception as e:  # one broken run must not hide the others
            skipped.append((p.name, f"unreadable: {type(e).__name__}"))
            continue
        if info is None:
            skipped.append((p.name, "no run files"))
        else:
            infos.append(info)
    return infos, skipped


def _num(v):
    """float, or None for bools / non-numbers / NaN / inf (the API must emit valid JSON when a run diverges)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _flatten(r):
    prefix = "eval." if r.get("eval") else ""
    flat = {}
    for k, v in r.items():
        if k in ("update", "step", "eval"):
            continue
        if isinstance(v, list):
            flat.update({f"{prefix}{k}.{i}": _num(x) for i, x in enumerate(v) if x is None or _num(x) is not None})
        elif _num(v) is not None:
            flat[prefix + k] = _num(v)
    return flat


def _thin(rows, n):
    if len(rows) <= n:
        return rows
    stride = len(rows) / n
    keep = {int(i * stride) for i in range(n)} | {len(rows) - 1}
    return [rows[i] for i in sorted(keep)]


def read_metrics(path, since=0, max_points=1500):
    """Streaming read of metrics.jsonl -> {'steps': [...], 'series': {name: [...]}}.

    List-valued fields are exploded to name.0, name.1, ... (null entries are gaps). Eval rows (eval=true) become series
    prefixed 'eval.' on the same step as the training row they follow. Training rows are thinned evenly to max_points;
    eval rows are sparse and always kept (up to max_points).
    """
    if not isinstance(max_points, int) or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    f = payload_dir(path) / "metrics.jsonl"
    train, evals = [], []
    if f.exists():
        with open(f, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                s = _step(r) if isinstance(r, dict) else None
                if s is not None and s >= since:
                    (evals if r.get("eval") else train).append((s, _flatten(r)))
    rows = sorted(_thin(train, max_points) + _thin(evals, max_points), key=lambda x: x[0])  # stable: eval after train at equal step
    merged = []
    for s, flat in rows:
        if merged and merged[-1][0] == s:
            merged[-1][1].update(flat)
        else:
            merged.append((s, dict(flat)))
    steps, series = [], {}
    for idx, (s, flat) in enumerate(merged):
        steps.append(s)
        for k in set(series) | set(flat):
            series.setdefault(k, [None] * idx).append(flat.get(k))
    return {"steps": steps, "series": series}
