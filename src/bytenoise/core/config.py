"""yaml loading with deep overrides (`a.b.c=value`) and a stable content hash."""
import hashlib
import json
import re
from pathlib import Path

import yaml

from .paths import config_dir


def list_refs(kind):
    d = config_dir() / kind
    return sorted(f.stem for f in d.glob("*.yaml")) if d.exists() else []


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def resolve_ref(kind, ref, allow_path=False):
    """configs/<kind>/<ref>.yaml. allow_path additionally accepts an explicit file (CLI only, never network input)."""
    if allow_path and Path(ref).suffix in (".yaml", ".yml") and Path(ref).exists():
        return Path(ref)
    if not isinstance(ref, str) or not NAME.match(ref):
        raise ValueError(f"invalid {kind} name {ref!r}")
    q = config_dir() / kind / f"{ref}.yaml"
    if not q.exists():
        raise FileNotFoundError(f"no {kind} config {ref!r} (looked for {q})")
    return q


def deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def parse_scalar(s):
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def apply_sets(cfg, items):
    """Apply ['train.max_updates=50', ...] to a nested dict. Unknown intermediate keys are created."""
    out = json.loads(json.dumps(cfg))
    for item in items or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} must look like key.sub=value")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot descend into {p!r} of {key!r}")
        node[parts[-1]] = parse_scalar(val)
    return out


def stable_hash(obj, n=16):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]
