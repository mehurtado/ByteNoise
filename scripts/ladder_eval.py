"""Compatibility alias: this module moved to experiments/ladder_eval.py (phase 5). Untracked scripts in scripts/ still import it by name."""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules[__name__] = importlib.import_module("experiments.ladder_eval")
