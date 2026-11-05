"""Serial training-job queue. Each job is a detached `python -m bytenoise.train run` process.

State is files under <cache>/jobs: <id>.json (written only by the agent), <id>.runs (appended by the child, one run id
per line), <id>.exit (written once by the child), <id>.log. Splitting writers this way avoids read-modify-write races,
and jobs survive agent restarts because status is re-derived from these files and the pid.
"""
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from ..core.paths import REPO_ROOT, cache_dir

_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
_lock = threading.RLock()
SRC = str(Path(__file__).resolve().parents[2])


def jobs_dir():
    d = cache_dir() / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _check_id(job_id):
    if not isinstance(job_id, str) or not _ID.match(job_id):
        raise ValueError(f"invalid job id {job_id!r}")
    return job_id


def _atomic(path, text):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def pid_alive(pid):
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


# --- called by the child process ---

def record_run(job_id, run_id):
    with open(jobs_dir() / f"{_check_id(job_id)}.runs", "a", encoding="utf-8") as f:
        f.write(run_id + "\n")


def finish(job_id, code):
    _atomic(jobs_dir() / f"{_check_id(job_id)}.exit", json.dumps({"code": code, "t": time.time()}))


# --- agent side ---

def submit(experiment, sets=(), device="cpu"):
    if device not in ("cpu", "cuda", "auto"):
        raise ValueError(f"invalid device {device!r}")
    from ..train.experiment import load_experiment
    load_experiment(experiment, sets)  # validate now, not when the job reaches the front of the queue
    job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
    job = {"id": job_id, "experiment": experiment, "sets": list(sets), "device": device, "created": time.time(),
           "pid": None, "started": None, "stopped": False}
    _atomic(jobs_dir() / f"{job_id}.json", json.dumps(job))
    return get(job_id)


def _raw(job_id):
    return json.loads((jobs_dir() / f"{_check_id(job_id)}.json").read_text(encoding="utf-8"))


def get(job_id):
    job = _raw(job_id)
    d = jobs_dir()
    runs_f, exit_f = d / f"{job_id}.runs", d / f"{job_id}.exit"
    job["runs"] = runs_f.read_text(encoding="utf-8").split() if runs_f.exists() else []
    if job.get("stopped"):
        job["status"] = "stopped"
    elif exit_f.exists():
        code = json.loads(exit_f.read_text(encoding="utf-8"))["code"]
        job["status"], job["exit_code"] = ("done" if code == 0 else "failed"), code
    elif job.get("started"):
        job["status"] = "running" if pid_alive(job["pid"]) else "failed"
        if job["status"] == "failed":
            job["note"] = "process vanished without an exit record"
    else:
        job["status"] = "queued"
    return job


def list_jobs(limit=50):
    ids = sorted((p.stem for p in jobs_dir().glob("*.json") if _ID.match(p.stem)), reverse=True)[:limit]
    return [get(i) for i in ids]


def log_tail(job_id, nbytes=8000):
    f = jobs_dir() / f"{_check_id(job_id)}.log"
    if not f.exists():
        return ""
    size = f.stat().st_size
    with open(f, "rb") as fh:
        fh.seek(max(0, size - nbytes))
        return fh.read().decode("utf-8", "replace")


def gpu_foreign_python(own_pids=()):
    """Names of python processes on the GPU other than ours (mirrors the launch_*.ps1 preflight). None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    found = []
    for ln in out.stdout.splitlines():
        pid, _, name = ln.partition(",")
        if re.search(r"python", name, re.I) and pid.strip().isdigit() and int(pid) not in set(own_pids):
            found.append(f"{pid.strip()}:{name.strip()}")
    return found


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        if out.returncode != 0:
            return {"available": False}
        name, used, total, util = [x.strip() for x in out.stdout.splitlines()[0].split(",")]
        return {"available": True, "name": name, "mem_used_mb": int(used), "mem_total_mb": int(total), "util_pct": int(util),
                "foreign_python": gpu_foreign_python()}
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError):
        return {"available": False}


def _spawn(job):
    d = jobs_dir()
    cmd = [sys.executable, "-m", "bytenoise.train", "run", job["experiment"], "--device", job["device"], "--job-id", job["id"]]
    for s in job["sets"]:
        cmd += ["--set", s]
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW
    else:
        kw["start_new_session"] = True
    log = open(d / f"{job['id']}.log", "ab")
    try:
        p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True, **kw)
    finally:
        log.close()
    job["pid"], job["started"] = p.pid, time.time()
    _atomic(d / f"{job['id']}.json", json.dumps({k: job[k] for k in ("id", "experiment", "sets", "device", "created", "pid", "started", "stopped")}))


def schedule():
    """Start the oldest queued job if nothing is running. Returns a short reason string when it is blocked, else None."""
    with _lock:
        jobs = list_jobs(200)
        if any(j["status"] == "running" for j in jobs):
            return "a job is running"
        queued = sorted((j for j in jobs if j["status"] == "queued"), key=lambda j: j["created"])
        if not queued:
            return None
        job = queued[0]
        if job["device"] != "cpu":
            foreign = gpu_foreign_python()
            if foreign is None:
                return "cannot verify GPU ownership (nvidia-smi unavailable)"
            if foreign:
                return f"GPU in use by {', '.join(foreign)}"
        _spawn(_raw(job["id"]))
        return None


def stop(job_id):
    with _lock:
        job = get(job_id)
        if job["status"] in ("done", "failed", "stopped"):
            return job
        raw = _raw(job_id)
        raw["stopped"] = True
        _atomic(jobs_dir() / f"{job_id}.json", json.dumps(raw))
        pid = job.get("pid")
        if pid and pid_alive(pid):
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30)
            else:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
        for rid in job["runs"]:
            _mark_stopped(rid)
        return get(job_id)


def _mark_stopped(run_id):
    from ..runstore.reader import safe_run_path
    from ..runstore.writer import RunWriter
    try:
        w = RunWriter(safe_run_path(run_id))
        if w.read_manifest().get("status") in ("running", "created"):
            w.update(status="stopped", finished=time.time())
            w.event("stopped")
    except (FileNotFoundError, ValueError, OSError):
        pass
