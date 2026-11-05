"""Experiment yaml -> resolved stage configs.

    name: debug-smoke
    model: debug            # configs/models/<ref>.yaml  ({family, preset, set})
    recipe: debug           # configs/recipes/<ref>.yaml ({train: {...}}); optional
    data: tiny              # dataset name -> train.data; optional
    set: {train: {seed: 3}} # overrides on the resolved legacy config; optional
    stages:
      - {name: main, train: {max_updates: 20}}
      - {name: lowlr, init_from: previous, train: {max_updates: 10, learning_rate: 1.0e-4}}
"""
import re
from dataclasses import dataclass

from ..core.config import apply_sets, deep_merge, load_yaml, resolve_ref, stable_hash
from ..models.legacy_adapters import legacy_config

_DATASET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SECTIONS = ("train", "model", "diffusion", "lm")
TOP_KEYS = {"name", "model", "recipe", "data", "set", "stages", "seed"}
STAGE_KEYS = {"name", "init_from", "train", "model", "diffusion", "lm"}


@dataclass
class Stage:
    name: str
    config: dict          # legacy Config.to_dict()
    config_hash: str
    init_from: str | None  # None | "previous" | run id


@dataclass
class Experiment:
    name: str
    ref: str
    model: dict
    data: str
    stages: list


def _check_keys(d, allowed, what):
    bad = sorted(set(d) - allowed)
    if bad:
        raise ValueError(f"unknown {what} keys {bad}; allowed {sorted(allowed)}")


def load_experiment(ref, sets=(), allow_path=False):
    raw = load_yaml(resolve_ref("experiments", ref, allow_path))
    _check_keys(raw, TOP_KEYS, "experiment")
    raw = dict(raw)
    if "model" not in raw:
        raise ValueError("experiment needs `model`")
    model_cfg = dict(load_yaml(resolve_ref("models", raw["model"], allow_path)))
    recipe = load_yaml(resolve_ref("recipes", raw["recipe"], allow_path)) if raw.get("recipe") else {}
    _check_keys(recipe, set(SECTIONS), "recipe")
    data = raw.get("data") or ""
    if data and not _DATASET.match(data):
        raise ValueError(f"invalid dataset name {data!r}")

    base = legacy_config({**model_cfg, "set": deep_merge(model_cfg.get("set") or {}, deep_merge(recipe, raw.get("set") or {}))}).to_dict()
    if data:
        base["train"]["data"] = data
    if "seed" in raw:
        base["train"]["seed"] = int(raw["seed"])
    base = apply_sets(base, sets)  # CLI/API overrides win over the yaml
    from config import Config  # legacy; validates field names and types

    stages_raw = raw.get("stages") or [{"name": "main"}]
    stages, prev = [], None
    for i, sr in enumerate(stages_raw):
        _check_keys(sr, STAGE_KEYS, "stage")
        cfg = deep_merge(base if prev is None else prev, {k: sr[k] for k in SECTIONS if k in sr})
        try:
            Config.from_dict(cfg)
        except (TypeError, KeyError) as e:
            raise ValueError(f"stage {sr.get('name') or i}: invalid config ({type(e).__name__}: {e})") from None
        init = sr.get("init_from")
        if init == "previous" and i == 0:
            raise ValueError("stage 0 cannot init_from 'previous'")
        stages.append(Stage(sr.get("name") or f"stage{i}", cfg, stable_hash(cfg), init))
        prev = cfg
    return Experiment(raw.get("name") or ref, ref, model_cfg, data, stages)
