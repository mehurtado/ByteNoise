"""Where bulk bytes live. Everything resolves from BYTENOISE_HOME (absolute), never from the working directory."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def home():
    h = os.environ.get("BYTENOISE_HOME")
    if not h:
        raise RuntimeError("BYTENOISE_HOME is not set; point it at the directory holding data/ and runs/")
    p = Path(h)
    if not p.is_absolute():
        raise RuntimeError(f"BYTENOISE_HOME must be absolute, got {h!r}")
    return p


def data_dir():
    return home() / "data"


def runs_dir():
    return home() / "runs"


def cache_dir():
    """Rebuildable state (sqlite index, job table). Deliberately outside runs/ so it never mixes with run files."""
    p = Path(os.environ.get("BYTENOISE_CACHE") or (home() / "var" / "cache"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def evals_dir():
    """Eval results for every run, native or legacy. Kept outside runs/ so evaluating never mutates a run directory."""
    return home() / "var" / "evals"


def config_dir():
    return Path(os.environ.get("BYTENOISE_CONFIGS") or (REPO_ROOT / "configs"))
