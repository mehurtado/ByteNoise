"""Run a protocol's modes against any ByteModel. A mode whose required capability is missing is skipped, not failed."""
import time

from ..models.base import ByteModel

_MODES = {}


def register_mode(name, needs=()):
    def deco(fn):
        _MODES[name] = (fn, frozenset(needs))
        return fn
    return deco


def list_modes():
    from . import modes  # noqa: F401
    return {n: sorted(needs) for n, (_, needs) in _MODES.items()}


def run_suite(model: ByteModel, protocol, device="cpu", only=None):
    """Returns a schema-versioned result dict (json-serialisable)."""
    from . import modes  # noqa: F401  (registers builtin modes)
    wanted = [m for m in protocol.modes if only is None or m in only]
    unknown = [m for m in wanted if m not in _MODES]
    if unknown:
        raise ValueError(f"unknown eval modes {unknown}; known: {sorted(_MODES)}")
    model.to(device)
    t0 = time.time()
    results = {}
    for name in wanted:
        fn, needs = _MODES[name]
        missing = sorted(needs - model.capabilities)
        if missing:
            results[name] = {"skipped": f"model lacks capability {', '.join(missing)}"}
            continue
        t = time.time()
        ctx = {"data": protocol.spec["data"], "split": protocol.spec["split"], "seed": protocol.spec["seed"], "device": device}
        results[name] = fn(model, dict(protocol.modes[name] or {}), ctx)
        results[name]["seconds"] = round(time.time() - t, 3)
    return {"schema": 1, "protocol": {"name": protocol.name, "hash": protocol.hash, "spec": protocol.spec},
            "model": model.describe(), "started": t0, "seconds": round(time.time() - t0, 3), "results": results}
