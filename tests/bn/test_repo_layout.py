"""Root-clutter guard: the repo root holds only the legacy core modules, packaging files and a fixed set of top-level docs."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_FILES = {
    ".gitignore", "README.md", "pyproject.toml", "requirements.txt", "requirements-lock.txt",
    # legacy core, imported by name by the bytenoise adapters and every experiment script
    "benchmark.py", "config.py", "data.py", "eval.py", "gpu_smoke.py", "main.py", "models_lm.py", "tasks.py", "train.py",
}
ALLOWED_DIRS = {".jules", "dataset", "configs", "data", "design", "docs", "experiments", "scripts", "src", "tests"}


def test_no_new_clutter_at_repo_root():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    top = {f.split("/")[0] for f in files if f}
    extra = sorted(t for t in top if t not in ALLOWED_FILES | ALLOWED_DIRS)
    assert not extra, f"new top-level entries {extra}: put docs under docs/, scripts under experiments/, code under src/bytenoise/"
