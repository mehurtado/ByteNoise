"""Path shim so tests run without installing the package (editable installs from a worktree go stale)."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TEXT = (b"the quick brown fox jumps over the lazy dog. pack my box with five dozen liquor jugs.\n"
        b"how vexingly quick daft zebras jump! sphinx of black quartz, judge my vow.\n")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Isolated BYTENOISE_HOME with a tiny dummy-v1 dataset; CPU only."""
    h = tmp_path / "home"
    d = h / "data" / "dummy-v1"
    d.mkdir(parents=True)
    (d / "train.bin").write_bytes(TEXT * 400)
    (d / "val.bin").write_bytes(TEXT * 40)
    (h / "runs").mkdir()
    monkeypatch.setenv("BYTENOISE_HOME", str(h))
    monkeypatch.setenv("BYTENOISE_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("BYTENOISE_CONFIGS", raising=False)
    return h
