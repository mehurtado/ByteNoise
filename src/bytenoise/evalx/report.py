"""Persist, render and compare eval results."""
import json
import os
import re
from pathlib import Path

from ..core.fsutil import atomic_write
from ..core.paths import evals_dir

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@-]*$")


def _fmt(v):
    return "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))


def flatten(obj, prefix=""):
    out = {}
    for k, v in obj.items():
        if k in ("samples", "spec", "sampler"):
            continue
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        elif v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)):
            out[f"{prefix}{k}"] = v
    return out


def save(result, run_id, tag="ema"):
    """var/evals/<run_id>/<protocol>@<tag>.json (atomic)."""
    for part in (run_id, tag, result["protocol"]["name"]):
        if not _SAFE.match(part):
            raise ValueError(f"unsafe path component {part!r}")
    d = evals_dir() / run_id
    d.mkdir(parents=True, exist_ok=True)
    name = result["protocol"]["name"]
    path = d / f"{name}@{tag}.json"
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            old = None
        if old and old["protocol"]["hash"] != result["protocol"]["hash"]:
            path = d / f"{name}.{result['protocol']['hash'][:8]}@{tag}.json"  # a different protocol/data: keep both
        elif old:
            result = {**result, "results": {**old.get("results", {}), **result["results"]}}  # same protocol: partial runs accumulate
    atomic_write(path, json.dumps(result, indent=1))
    return path


def list_results(run_id=None):
    root = evals_dir()
    if not root.exists():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (run_id is None or p.name == run_id)):
        for f in sorted(d.glob("*.json")):
            out.append({"run": d.name, "file": f.name, "mtime": f.stat().st_mtime})
    return out


def load(run_id, file):
    if not _SAFE.match(run_id) or not _SAFE.match(file.removesuffix(".json")):
        raise ValueError("bad eval reference")
    f = evals_dir() / run_id / file
    if not f.exists():
        raise FileNotFoundError(f"no eval result {run_id}/{file}; run `bn eval run {run_id} --protocol <name>` first")
    return json.loads(f.read_text(encoding="utf-8"))


def to_markdown(result, title=""):
    m, p = result["model"], result["protocol"]
    L = [f"## {title or m['name']} ({m['family']}) - protocol {p['name']} [{p['hash']}]", ""]
    for mode, res in result["results"].items():
        L.append(f"### {mode}")
        if "skipped" in res:
            L += [f"skipped: {res['skipped']}", ""]
            continue
        L += ["| metric | value |", "|---|---|"] + [f"| {k} | {_fmt(v)} |" for k, v in flatten(res).items()]
        for s in res.get("samples", [])[:3]:
            L += ["", f"sample: `{s['repr'][:160]}`"]
        L.append("")
    return "\n".join(L)


def compare(results, force=False):
    """results: {label: result}. Refuses mismatched protocol hashes unless force."""
    hashes = {r["protocol"]["hash"] for r in results.values()}
    if len(hashes) > 1 and not force:
        raise ValueError(f"protocols differ ({sorted(hashes)}); comparing them is not meaningful (force=True to override)")
    flat = {k: flatten(r["results"]) for k, r in results.items()}
    names = list(flat)
    keys = [k for k in flat[names[0]] if all(k in f for f in flat.values()) and not k.endswith(".seconds")]
    L = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    L += [f"| {k} | " + " | ".join(_fmt(flat[n][k]) for n in names) + " |" for k in keys]
    if not keys:
        L.append("\n(no metrics in common: these models/modes are not comparable)")
    if len(hashes) > 1:
        L.append("\n(warning: protocol hashes differ)")
    return "\n".join(L) + "\n"
