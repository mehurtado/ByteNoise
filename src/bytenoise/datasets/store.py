"""Byte datasets: <home>/data/<name>/{train,val,test}.bin (uint8), optionally a virtual mix.json."""
import json
from pathlib import Path

import numpy as np
import torch

from ..core.paths import data_dir

SPLITS = ("train", "val", "test")


def dataset_path(name):
    p = (data_dir() / name).resolve()
    if data_dir().resolve() not in p.parents:
        raise ValueError(f"dataset name {name!r} escapes the data directory")
    return p


def load_split(name, split):
    """uint8 tensor of a split. A dataset with mix.json is the concatenation of its sources' splits."""
    d = dataset_path(name)
    mix = d / "mix.json"
    if mix.exists():
        sources = json.loads(mix.read_text(encoding="utf-8"))["sources"]
        return torch.cat([load_split(s["dataset"], split) for s in sources])
    f = d / f"{split}.bin"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing")
    return torch.from_numpy(np.fromfile(f, dtype=np.uint8))


def load_prefix(name, split, max_bytes):
    """First max_bytes of a split as raw bytes without reading the whole file. Mix datasets use their first source."""
    d = dataset_path(name)
    if (d / "mix.json").exists():
        name = json.loads((d / "mix.json").read_text(encoding="utf-8"))["sources"][0]["dataset"]
        d = dataset_path(name)
    f = d / f"{split}.bin"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing")
    return np.fromfile(f, dtype=np.uint8, count=int(max_bytes)).tobytes()


def windows(blob, size, stride, count=None):
    """(n, size) stack of strided windows over a 1-D tensor."""
    n = max(0, (len(blob) - size) // stride + 1)
    if count is not None:
        n = min(n, count)
    return torch.stack([blob[i * stride:i * stride + size] for i in range(n)]) if n else blob.new_zeros((0, size))


def list_datasets():
    """Cheap listing (stat only, never reads the .bin files)."""
    out = []
    root = data_dir()
    if not root.exists():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
        sizes = {s: (d / f"{s}.bin").stat().st_size for s in SPLITS if (d / f"{s}.bin").exists()}
        is_mix = (d / "mix.json").exists()
        if not sizes and not is_mix:
            continue
        out.append({"name": d.name, "bytes": sizes, "mix": is_mix, "manifest": (d / "manifest.json").exists()})
    return out
