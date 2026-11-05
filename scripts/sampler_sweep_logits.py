"""Compatibility alias: this module moved to experiments/sampler_sweep_logits.py (phase 5). Untracked scripts in scripts/ still import it by name."""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules[__name__] = importlib.import_module("experiments.sampler_sweep_logits")
