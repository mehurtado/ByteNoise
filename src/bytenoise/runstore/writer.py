"""Native run layout: runs/<YYYYMMDD>-<slug>-<4hex>/{run.json, events.jsonl, config.resolved.json, ...}."""
import json
import os
import re
import secrets
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ..core.fsutil import atomic_write
from ..core.paths import REPO_ROOT, runs_dir

SCHEMA = 1


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "run"


def git_state():
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


class RunWriter:
    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def create(cls, slug, manifest):
        """Make a fresh run directory (never reuses one) and write the initial manifest."""
        root = runs_dir()
        root.mkdir(parents=True, exist_ok=True)
        while True:
            path = root / f"{datetime.now():%Y%m%d}-{slugify(slug)}-{secrets.token_hex(2)}"
            try:
                path.mkdir()
                break
            except FileExistsError:
                continue
        w = cls(path)
        w.write_manifest({"schema": SCHEMA, "id": path.name, "status": "created", "created": time.time(),
                          "git": git_state(), "python": sys.version.split()[0], **manifest})
        w.event("created")
        return w

    def read_manifest(self):
        return json.loads((self.path / "run.json").read_text(encoding="utf-8"))

    def write_manifest(self, manifest):
        atomic_write(self.path / "run.json", json.dumps(manifest, indent=2, default=str))

    def update(self, **fields):
        m = self.read_manifest()
        m.update(fields)
        self.write_manifest(m)
        return m

    def event(self, kind, **data):
        with open(self.path / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "event": kind, **data}, default=str) + "\n")

    def write_json(self, rel, obj):
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(p, json.dumps(obj, indent=2, default=str))
        return p
