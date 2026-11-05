"""Run an experiment's stages, each as one native run directory.

The optimisation loop is the original train.run_training (wrapped, not rewritten); this layer adds the manifest, the
stage chaining (warm start from the previous stage) and failure bookkeeping. Trainer output lives in <run>/train/.
"""
import time
import traceback

from ..core import legacy
from ..runstore.reader import payload_dir, safe_run_path
from ..runstore.writer import RunWriter
from .experiment import load_experiment


def _ckpt_of(run_path):
    return payload_dir(run_path) / "checkpoint.pt"


def run_experiment(ref, sets=(), device="auto", allow_path=False, on_run=None):
    """Execute all stages in order. Returns the list of run ids. on_run(run_id) is called as each stage run is created."""
    exp = load_experiment(ref, sets, allow_path)
    legacy.ensure()
    import train as legacy_train
    from config import Config

    run_ids, prev_path = [], None
    for i, st in enumerate(exp.stages):
        init_path = None
        parent = None
        if st.init_from == "previous":
            parent, init_path = run_ids[-1], _ckpt_of(prev_path)
        elif st.init_from:
            p = safe_run_path(st.init_from)
            parent, init_path = p.name, _ckpt_of(p)
        if init_path is not None and not init_path.exists():
            raise FileNotFoundError(f"init_from checkpoint {init_path} does not exist")

        w = RunWriter.create(f"{exp.name}-{st.name}", {
            "experiment": exp.ref, "stage": st.name, "stage_index": i, "parent": parent,
            "model": {"family": exp.model["family"], "preset": exp.model.get("preset")},
            "config_hash": st.config_hash, "data": st.config["train"].get("data"), "status": "created"})
        w.write_json("config.resolved.json", st.config)
        run_ids.append(w.path.name)
        prev_path = w.path
        if on_run:
            on_run(w.path.name)
        w.update(status="running", started=time.time())
        w.event("start", device=device)
        try:
            cfg = Config.from_dict(st.config)
            legacy_train.run_training(cfg, str(w.path / "train"), mode="fresh", device=device,
                                      init_from=str(init_path) if init_path else None, quiet=True)
        except BaseException as e:  # KeyboardInterrupt/SystemExit too: the manifest must not stay "running"
            w.update(status="failed", finished=time.time(), error=f"{type(e).__name__}: {e}")
            w.event("failed", error=traceback.format_exc(limit=5))
            raise
        w.update(status="done", finished=time.time())
        w.event("done")
    return run_ids
