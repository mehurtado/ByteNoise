# ByteNoise repo restructure (design only, nothing moved or edited)

## 1. Diagnosis (measured 2026-11-04)

| Symptom | Evidence |
|---|---|
| No stable library boundary | `scripts/` has 480 flat files; 249 do `sys.path.insert`; scripts import each other as modules (`import latenthyp_slotdiff` x87, `latentloop_eval` x55). Scripts *are* the library. |
| 5+ model families, no shared interface | UNet/Hybrid/Hier/RPW diffusion all inside `main.py` (1474 lines); AR `models_lm.py`; `latenthyp/`, `latentloop/`, `maskrev/` packages; `design/` prototypes. Each has its own train + eval entry. |
| Training is one bespoke script per run | 24 `run_*`, 14 `persistent_*_task.py`, `launch_*.ps1`, `mtp_*`, `train_*`; 81 `runner_snapshot.py` + 43 `model_snapshot.py` copies inside `runs/` (poor-man's reproducibility). |
| Eval is half-generic | `eval.py` works for `train.py` run dirs only; `scripts/eval_*` (9+), `eval_standard_harness.py` hard-codes arms A0-A4 and imports 4 latenthyp modules. Protocols live in docstrings. |
| `runs/` is an unindexed dump | 604 entries (230 run directories, 74 non-run dirs, 300 loose log files), 82 GB; logs/`.out` files beside run dirs; 3 checkpoint conventions (`checkpoint.pt`, `state_N.pt`, `model_N.pt`); 2-3 status/metrics schemas; metadata encoded in folder names. |
| `data/` mixes code, downloads, outputs | 37 GB; `build_*.py` next to `_fineweb_download/`, `_scratch_*_downloads/`, `.bin` files. |
| Root clutter | 8 handoff/progress `.md`, 922 MB `.pth`, discord bot + logs, plot script + PNG, `mtp_k2_..._20261028/` (593 MB run), `scratch/`, and a sibling `~/ByteNoise_dev`, which is a live git worktree on branch `rungs` (not a stale copy). |

Root cause: every experiment added a script instead of a config. Fix = make *model*, *data*, *recipe*, *eval protocol* the only extension points.

## 2. Target layout

```
ByteNoise/
  pyproject.toml            # pip install -e .  -> kills all sys.path hacks; exposes `bn` CLI
  README.md                 # 1 page: install, `bn` cheat-sheet, link to docs/
  configs/                  # THE extension surface (yaml, versioned)
    models/   hybrid_selfcond.yaml  rpw_base.yaml  ar_small.yaml ...
    data/     scratch_english_v2.yaml  reasoning_mix35.yaml ...   # sources + mix weights + manifest hash
    recipes/  diffusion_default.yaml  warmstart.yaml ...           # optimizer, schedule, ema, callbacks
    experiments/  <name>.yaml                                      # model + data + recipe + stages + overrides
    protocols/    standard256.yaml  gen_quality.yaml  prefix_paired.yaml   # eval suites, versioned
  src/bytenoise/
    core/       paths (BYTENOISE_HOME), seeds, device, config loading, registry, checkpoint io, hashing
    models/     one subpackage per family, each registers an adapter (see §3)
      unet/ hybrid/ hier/ rpw/      # split out of main.py; diffusion math -> models/_diffusion.py
      ar_lm/                        # models_lm.py
      latenthyp/ latentloop/ maskrev/
    datasets/   registry, window/loader, task generators (tasks.py), builders/ (all build_*.py), manifests
    train/      Trainer, stages, callbacks, objectives, launch (local detach)
    evalx/      harness, modes/, protocols loader, metrics (text_metrics etc.), report + compare
    runstore/   RunStore: layout, manifest, index (sqlite cache), legacy readers
    viz/        plot functions -> plot spec JSON; same code renders PNG (CLI) or UI
    serve/      agent (jobs, GPU, logs), api, ui/ (static)
    apps/       discord_bot
  experiments/  # FROZEN HISTORY, never imported by library
    latenthyp/  (81 scripts)  rpw/  mtp/  phase3-4/  ...   each with a README stating what it answered
  tests/        unit/ (per package)   integration/ (golden-run, adapter conformance)
  docs/         architecture.md  runbook.md  experiments.md (was EXPERIMENTS.md)  archive/handoffs/
  var/  (gitignored; or BYTENOISE_HOME elsewhere)
    data/  runs/  artifacts/  cache/  remote/
```

Rules that keep it permanent: dependency direction `core <- models, data <- train, evalx <- serve/ui`; library never imports `experiments/`; new model = adapter + yaml, never a new top-level script; CI check fails on new files in repo root or new `sys.path` edits; code in git, bulk bytes (`data/`, `runs/`) never.

## 3. Model contract (the "simple endpoints")

```python
class ByteModel(Protocol):
    name: str; seq_len: int
    capabilities: set[str]            # {"generate","score","denoise","loss"}
    @classmethod
    def from_config(cls, cfg) -> "ByteModel": ...
    def load(self, ckpt: Path, which="ema"): ...
    # training endpoint
    def loss(self, batch: Tensor) -> LossOut: ...           # total + named parts (ce, noise, ...)
    # inference endpoints (all bytes in / bytes out)
    def generate(self, prompt: bytes, n_bytes: int, *, seed: int, **sampler) -> bytes
    def score(self, data: bytes) -> ScoreOut                # per-byte nll; `exact: bool` (AR=True, diffusion bound=False)
    def denoise(self, x_t, t) -> ...                        # optional, diffusion only
```

Wiring a model = ~50-line adapter + `@register_model("rpw")` + a model yaml. `sampler` kwargs are declared by the adapter (`sampler_schema()`), so the UI and CLI render the right knobs automatically (steps, tau mode, temperature, top_k, passes...). Existing classes are wrapped, not rewritten, in phase 2 (strangler).

Adapter conformance test (`tests/integration/test_adapter_conformance.py`) runs every registered model on the `debug` preset: loss finite, generate returns bytes of requested length and is seed-deterministic, declared capabilities match implemented methods.

## 4. Eval harness

- **Protocol** = yaml: data split + count, seeds, noise buckets, sampler settings, mode list, metric list. Hash of the resolved protocol is stamped into every result; `compare` refuses (or flags) mismatched hashes. This replaces the docstring protocols in `eval_standard_harness.py`.
- **Modes** (each a small function `(model, protocol) -> dict`; skipped automatically when a capability is missing):
  `bpb` (score; labelled exact vs bound), `tasks` (copy/recall/rule/pattern + math-exact), `generation` (text_metrics, memorization, word validity), `prefix_paired` (matched vs shifted prompt, bootstrap CI), `noise_buckets` (diffusion), `longctx` (NIAH/stateful), `probe` (chat/curriculum probes).
- Output: `var/evals/<run_id>/<protocol>@<which>.json` (outside `runs/`, so evaluating never mutates a run), schema-versioned. `bn eval compare A B C` yields one table.
- Cross-family comparison works because everything goes through `generate`/`score`, not model internals.

## 5. Training structure (same shape)

- `experiments/<name>.yaml` = `model:` ref + overrides, `data:` ref + mix, `recipe:` ref, and `stages:` (list; each stage can `init_from: <run>@<ckpt>`, change lr/data/steps, reset optimizer). This replaces `*_task.py`, `launch_*.ps1`, `run_*.sh`, warm-start and resume scripts.
- One `Trainer` drives anything exposing `loss()`: grad-accum, amp, EMA, clipping, finite checks, wall-clock stop (`max_seconds`), callbacks (checkpoint, sample, eval-every-N via the §4 harness, status heartbeat). Current `train.run_training` logic is the seed.
- Run directory (one convention, written by Trainer only):
```
runs/<YYYYMMDD>-<slug>-<4char>/
  run.json            id, parent(run@step), model, experiment, config hash, git rev + dirty diff, data manifest hash, env, host, status
  config.resolved.yaml
  metrics.jsonl       {step, t, scalars:{...}}   (one schema)
  events.jsonl        launch / stage change / stop reason / errors
  ckpt/               step_N.pt, latest.pt, best.pt   (raw + EMA in one file)
  samples/ eval/ logs/
```
  Replaces the `runner_snapshot.py` copies: reproducibility comes from `run.json` + git rev + resolved config.
- **Legacy runs are not migrated.** `runs/legacy.py` reads the 598 old runs through adapters (metrics.jsonl/status.json/protocol.json variants) and records them in the sqlite index with `legacy=true`, so the UI and viz work on history without touching a byte of it. Index is a rebuildable cache; run files stay the source of truth.

## 6. Remote training over Tailscale

- `bn serve` runs one **agent** on the GPU box (single process, absolute `BYTENOISE_HOME`, never cwd-derived; launched by a `restart.bat`-style script per existing practice). Exposed with `tailscale serve` (tailnet only, not funnel); auth = allowlist on the `Tailscale-User-Login` header plus a bearer token for CLI.
- Job model: serial GPU queue (8 GB card). Start = spawn **detached** `python -m bytenoise.train run <resolved.yaml>` with pidfile + heartbeat in `run.json`/`events.jsonl`, so jobs survive agent restarts. Pre-flight checks lifted from `launch_*.ps1`: required files exist, run dir doesn't, no other python on the GPU.
- API (also what the UI and CLI call): `POST /jobs` (experiment + `--set k=v` overrides), `GET /jobs`, `POST /jobs/{id}/stop|resume`, `GET /jobs/{id}/logs` (SSE tail), `GET /gpu`, `GET /runs`, `GET /runs/{id}/metrics?since=`, `POST /eval`, `POST /generate`.
- CLI from any tailnet machine: `bn remote add desktop https://desktop-667n4qr.tailb3695b.ts.net`, then `bn --remote desktop train configs/experiments/foo.yaml --set recipe.lr=3e-4`.

## 7. UI (thin; every action is also an API call and a `bn` command)

Single web app served by the agent, no node build and no new dependencies (stdlib `http.server` + vanilla JS canvas charts).

| Tab | Content |
|---|---|
| Runs | indexed table (status, model, data, step, bpb, last eval), filters/tags, multi-select -> compare |
| Run | loss/grad/lr curves, noise-bucket curves, sample browser by step, eval results, resolved config + git rev, live log tail, stop/resume |
| Train | pick experiment yaml, edit overrides in a form, preview resolved config, launch, queue view, GPU gauge |
| Test | pick run+ckpt (or two for A/B), choose mode/protocol or free-prompt playground; sampler knobs from `sampler_schema()`; results saved as eval artifacts |
| Data | datasets, manifests, hash, sizes, build status |

`viz/` functions return plot specs (JSON); UI renders them, `bn viz` renders the same specs to PNG (replaces `plot_training_history_global_100k.py`).

CLI surface: `bn train | eval | gen | runs ls/show/compare | viz | data ls/build | serve | remote`.

## 8. Where existing things land (mapping)

| Today | Becomes |
|---|---|
| `main.py` | `models/{unet,hybrid,hier,rpw}` + `models/_diffusion.py` (GaussianDiffusion) |
| `models_lm.py` | `models/ar_lm` |
| `latenthyp/ latentloop/ maskrev/` | `models/*` (package code) |
| `train.py` | `train/` (+ `train/checkpoint.py`, `train/data_windows.py`) |
| `eval.py`, `scripts/eval_*`, standard harness | `evalx/` metrics + modes + `configs/protocols/*` |
| `config.py` presets | `configs/models/*.yaml` + dataclass schema in `core/config.py` |
| `data.py`, `tasks.py`, `data/build_*.py`, `scripts/build_*`, `prepare_*`, `mix_*` | `data/` + `data/builders/` |
| `scripts/latenthyp_*` (81), `persistent_*_task`, `run_*`, `launch_*`, `mtp_*`, `phase3/4` | `experiments/<family>/` frozen archive (+ yaml equivalents for any still-live recipe) |
| `scripts/test_*`, `tests/` | `tests/unit`, `tests/integration` |
| `discord_bot.py` | `apps/discord` (uses `ByteModel.generate`) |
| `plot_training_history*.py`, PNG | `viz/`, outputs to `var/artifacts` |
| Handoff/progress/plan `.md` | `docs/archive/handoffs/`; `EXPERIMENTS.md` -> `docs/experiments.md` |
| `byte_diffusion_model.pth`, `mtp_*_20261028/` | `var/artifacts/legacy/`, `var/runs/` |
| `scratch/` | `experiments/scratch/` or delete after review |
| `~/ByteNoise_dev` | live worktree on `rungs`; leave alone, merge or remove that branch deliberately |

## 9. Migration plan (strangler; old layout keeps working until the last step)

0. **Freeze**: git tag, pick `BYTENOISE_HOME`, build on a branch/worktree. Memory notes multiple concurrent Claude sessions touch this repo, so no file moves while sessions are active; confirm no live training before phase 5 (many scripts and `runner_snapshot.py` copies reference current paths).
1. **core + RunStore + legacy readers + viz + read-only UI (Runs/Run tabs).** Zero risk, immediate value: all 230 historical run directories browsable.
2. **ByteModel adapters + evalx**, wrapping existing classes. Port hybrid_selfcond (Discord model), RPW, AR first. Port `eval.py` metrics and the standard-256 protocol as yaml. Test tab goes live.
3. **Trainer + stages + new run layout.** Golden test: debug preset, 200 updates, loss curve matches the old `train.py` to tolerance. Then translate the live recipes (`persistent_*`, rpw, hier) into experiment yamls.
4. **Agent + remote jobs + Train tab** over Tailscale.
5. **Physical move** (`git mv`, history kept): `scripts/` -> `experiments/`, root clutter -> docs/var, Add the root-clutter CI check.

Each phase ships independently; stopping after phase 2 already gives the unified eval + browse/compare UI.

## 10. Choices made (change if you disagree)

- Web UI over TUI/notebook, because remote-over-Tailscale is a requirement.
- yaml configs over Python presets, so a new experiment is data, not code.
- Legacy runs indexed read-only, never rewritten.
- Tests/protocol hash enforce comparability, because current cross-run comparisons depend on remembered conventions.
- Single repo, installable package (not multiple repos); heavy artifacts outside git.

## 11. Implementation status (branch `restructure-v2`)

Done, additive (nothing moved): phases 1-4 in `src/bytenoise/`, `configs/`, `tests/bn/` (see `docs/runbook.md`). Deviations from the
plan above: stdlib server instead of FastAPI; eval results under `var/evals`; the trainer wraps the original loop rather than
replacing it; native runs keep trainer files in `<run>/train/`; package dirs named `runstore` and `datasets` (`runs/` is gitignored).
Not done: phase 5 (physical moves), adapters for `latenthyp`/`latentloop`/`maskrev`, prefix-conditioned diffusion `continue`,
a full tailscale deployment, PNG export verification (matplotlib not installed).
