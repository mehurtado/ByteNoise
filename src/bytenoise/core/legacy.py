"""Strangler shim: the original root modules (config, main, train, models_lm, tasks) are wrapped, not rewritten.

This is the only place that touches sys.path; delete it when those modules move into bytenoise.models.
"""
import sys

from .paths import REPO_ROOT, data_dir


def ensure():
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import data as legacy_data  # root data.py
    legacy_data.DATA_ROOT = str(data_dir()) if _has_home() else legacy_data.DATA_ROOT


def _has_home():
    import os
    return bool(os.environ.get("BYTENOISE_HOME"))
