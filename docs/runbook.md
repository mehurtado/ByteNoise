# ByteNoise v2 runbook

Additive layer over the original modules: nothing was moved. `src/bytenoise/` wraps `main.py` / `train.py` / `models_lm.py`
/ `tasks.py` (see `core/legacy.py`, the only `sys.path` shim) and reads the existing `runs/` and `data/` untouched.

## Setup

```
set BYTENOISE_HOME=C:\Users\mehur\ByteNoise        # absolute; holds data/ and runs/ (and var/ for new state)
set PYTHONPATH=<repo>\src                          # or: pip install -e .  (avoid in the shared .venv from a temporary worktree)
python -m bytenoise <command>                      # same as `bn <command>` once installed
```

## Commands

| Need | Command |
|---|---|
| list / inspect runs (all 230 legacy run dirs + native ones) | `bn runs ls`, `bn runs show RUN`, `bn runs metrics RUN --keys loss` |
| model families, capabilities, sampler knobs | `bn models` |
| evaluate a run on a protocol | `bn eval run RUN --protocol quick [--data D] [--modes bpb generation]` |
| compare runs (refuses mismatched protocol hashes) | `bn eval compare RUN1 RUN2 --protocol quick` |
| sample from a run | `bn gen RUN --prompt "the " -n 100 --sampler temperature=0.7` |
| train an experiment (foreground) / queue it | `bn train debug_smoke` / `bn train debug_smoke --detach` |
| plot | `bn viz RUN --key loss,grad_norm --out a.png` (needs matplotlib) or without `--out` for the JSON spec |
| UI + agent | `bn serve` then open http://127.0.0.1:8765 |
| remote | `bn remote add desktop https://desktop-667n4qr.tailb3695b.ts.net --token T`, `bn remote train desktop my_exp --set train.max_updates=500 --device cuda`, `bn remote jobs desktop` |

## Add a model family

Subclass `ByteModel` (`models/base.py`), decorate with `@register_model("name")`, declare `capabilities`, implement
`from_config`, `load_weights`, `to`, `num_params` plus whichever of `loss` / `score` / `generate`. Add `configs/models/<x>.yaml`.
`tests/bn/test_core_and_models.py::test_adapter_conformance` shows the contract every family must satisfy.

## Add an eval mode / protocol

Mode: a function in `evalx/modes/` decorated with `@register_mode("name", needs={capability})`. Protocol: `configs/protocols/<x>.yaml`.
Results are stamped with the protocol hash; any edit to a protocol changes it.

## Add an experiment

`configs/experiments/<x>.yaml` = model ref + recipe ref + data + optional `stages` (see `train/experiment.py` docstring). No new script.

## Exposing it over Tailscale (not done by the code; run on the GPU box yourself)

```
bn serve --port 8765 --token <secret> --allow-host desktop-667n4qr.tailb3695b.ts.net
tailscale serve --bg 8765        # tailnet only; never `tailscale funnel`
```
`--tailnet-users` (header-only auth) works only when no token is set; with a token the token is always required, because any local process can forge the Tailscale header. The agent binds 127.0.0.1 only and refuses a non-loopback bind without a token. Jobs on `--device cuda` stay queued while another
python process holds the GPU (same preflight as the old `launch_*.ps1` scripts); `--device cpu` jobs start immediately.

## Known limits (honest list)

- `/api/generate` and `/api/eval` are CPU-only; GPU work goes through the job queue (which preflights for other GPU users).
- Jobs run with the agent's privileges. `train.data_dir` and absolute dataset paths are rejected for API/CLI-submitted jobs.
- `bn remote` follows HTTP redirects with the token attached and stores the token in plaintext in `remotes.json` (chmod is a no-op on Windows); use only trusted tailnet URLs.
- Checkpoints load with `torch.load(weights_only=False)`: only put trusted files under `runs/`.
- Native runs killed hard are shown `incomplete` after 30 min of metric silence, not instantly.

* Legacy run status is inferred: `done` = status.json says stopped by max_updates/max_seconds, `timed_out` = stopped by the wall-clock
  budget before reaching max_updates, `running` = metrics changed in the last 5 min, anything else is `incomplete` (really "unknown";
  about two thirds of old runs have no usable status.json).
* `bn remote add` stores the token in plaintext in `~/.bytenoise/remotes.json` (override: `BYTENOISE_REMOTES`).
* `bn eval compare` needs a prior `bn eval run` with the same protocol and `--which` for every run.

* Training still executes the original `train.run_training` loop; v2 adds manifests, stage chaining, queueing, failure bookkeeping.
  Checkpoints of native runs are `<run>/train/checkpoint.pt` (`runstore.reader.payload_dir`).
* The diffusion adapter is unconditional: prefix-conditioned continuation lives in uncommitted work and is not wrapped yet, so
  diffusion models skip `bpb`/`tasks` and declare no `continue`. Only `ar_lm` and `diffusion` (unet/birnn/hybrid/hier/rpw) are adapted;
  `latenthyp`, `latentloop`, `maskrev` are not.
* Legacy `state_N.pt` / `model_N.pt` run formats are indexed and plotted but cannot be loaded for generation/eval (only `checkpoint.pt`).
* `/api/generate` and `/api/eval` are synchronous and serialised (one model op at a time); a long diffusion sample blocks that request
  (hybrid, 1000 steps, CPU: ~2 min).
* Phase 5 was done in part: `scripts/` is now `experiments/` (except `scripts/status.py`), root docs moved under `docs/`. The legacy core modules stay at the root; `~/ByteNoise_dev` was left alone.
