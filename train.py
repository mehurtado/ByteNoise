import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

from config import Config, PRESETS, get_preset, override, validate
from main import ByteDiffusionLM, ByteTokenizer, EMA, load_text_data
from models_lm import ByteLM

torch.set_float32_matmul_precision("high")
CHECKPOINT_NAME = "checkpoint.pt"
CHECKPOINT_VERSION = 2


class CheckpointError(RuntimeError):
    pass


# --- Pure helpers (unit-tested) ---

def group_sizes(num_batches, accum_steps):
    """Microbatch count of each optimizer update in one epoch; the last group may be partial."""
    full, rem = divmod(num_batches, accum_steps)
    return [accum_steps] * full + ([rem] if rem else [])


def resolve_device(name="auto"):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def git_revision():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=here, capture_output=True, text=True, timeout=10).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def amp_settings(precision, device):
    """(autocast_device or None, autocast dtype, use GradScaler)"""
    if device != "cuda" or precision == "fp32":
        return None, None, False
    return "cuda", (torch.float16 if precision == "fp16" else torch.bfloat16), precision == "fp16"


def accumulate_group(loss_fn, batches, scaler=None, autocast_device=None, autocast_dtype=None):
    """Backprop one optimizer update's microbatches, normalising by the true group size.

    loss_fn(batch) -> (loss, stats). Returns (unscaled microbatch losses, stats list).
    """
    n = len(batches)
    losses, all_stats = [], []
    for batch in batches:
        if autocast_device:
            with torch.autocast(autocast_device, dtype=autocast_dtype):
                loss, stats = loss_fn(batch)
        else:
            loss, stats = loss_fn(batch)
        check_finite_loss(loss, stats)
        scaled = loss / n
        (scaler.scale(scaled) if scaler is not None else scaled).backward()
        losses.append(loss.item())
        all_stats.append(stats)
    return losses, all_stats


def check_finite_loss(loss, stats):
    if not torch.isfinite(loss):
        t = stats.get("t")
        per = stats.get("per_sample_noise_loss")
        detail = ""
        if t is not None and per is not None:
            bad = (~torch.isfinite(per)).nonzero().flatten().tolist() or list(range(len(t)))
            detail = f"; failing timesteps: {[int(t[i]) for i in bad]}"
        comps = {k: float(v) for k, v in stats.items() if k in ("noise_loss", "ce_loss", "nll")}
        raise FloatingPointError(f"non-finite loss {float(loss)} (components {comps}){detail}")


def bucket_losses(t, per_sample, num_timesteps, num_buckets):
    """Mean noise loss per noise-level bucket (bucket 0 = least noise). NaN where empty."""
    idx = (t.float() / num_timesteps * num_buckets).long().clamp(max=num_buckets - 1)
    sums = torch.zeros(num_buckets).index_add_(0, idx.cpu(), per_sample.float().cpu())
    counts = torch.zeros(num_buckets).index_add_(0, idx.cpu(), torch.ones_like(per_sample, dtype=torch.float).cpu())
    return sums, counts


def lr_lambda(warmup, total, lr, lr_min, elapsed=None, max_seconds=0.0):
    """Warmup then cosine. With a wall-clock budget the decay follows whichever of update or time progress is further along."""
    floor = lr_min / lr

    def f(step):  # step = number of completed scheduler steps
        if warmup and step < warmup:
            return (step + 1) / warmup
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        if max_seconds and elapsed is not None:
            prog = max(prog, min(1.0, elapsed() / max_seconds))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
    return f


# --- Models / data ---

def build_model(cfg):
    return ByteDiffusionLM(cfg) if cfg.arch == "diffusion" else ByteLM(cfg)


def window_len(cfg):
    return cfg.model.seq_len if cfg.arch == "diffusion" else cfg.lm.seq_len + 1


class Windows:
    """Non-overlapping fixed-length byte windows over a uint8 tensor; the trailing partial window is dropped."""

    def __init__(self, tokens, length):
        self.length = length
        n = tokens.numel() // length
        self.view = tokens[: n * length].view(n, length)

    def __len__(self):
        return self.view.size(0)

    def get(self, idx):
        return self.view[idx].long()


def _to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data.to(torch.uint8)
    return torch.tensor(list(data), dtype=torch.uint8)


def tensor_hash(tokens):
    return hashlib.sha256(tokens.numpy().tobytes()).hexdigest()


def make_data(cfg, data=None, val=None):
    """Returns (train Windows, val tensor or None, train hash). `data`/`val` override the configured dataset."""
    t = cfg.train
    if data is not None:
        train_tokens, val_tokens = _to_tensor(data), (_to_tensor(val) if val is not None else None)
    elif t.data:
        from data import load_split
        train_tokens, val_tokens = load_split(t.data, "train"), load_split(t.data, "val")
    else:
        train_tokens, val_tokens = torch.tensor(ByteTokenizer().encode(load_text_data(t.data_dir)), dtype=torch.uint8), None
    windows = Windows(train_tokens, window_len(cfg))
    if len(windows) < t.batch_size:
        raise ValueError(f"dataset has {len(windows)} windows of {windows.length} bytes, fewer than batch_size {t.batch_size}")
    return windows, val_tokens, tensor_hash(train_tokens)


def epoch_batches(windows, cfg, epoch):
    g = torch.Generator().manual_seed(cfg.train.seed + epoch)
    perm = torch.randperm(len(windows), generator=g)
    bs = cfg.train.batch_size
    return [perm[i * bs:(i + 1) * bs] for i in range(len(windows) // bs)]


@torch.no_grad()
def quick_eval(model, cfg, val_tokens, device, n_batches):
    """Deterministic validation: AR -> bits/byte; diffusion -> noise/ce loss with fixed timesteps and noise."""
    was_training = model.training
    model.eval()
    win = Windows(val_tokens, window_len(cfg))
    bs = cfg.train.batch_size
    n = min(len(win), n_batches * bs)
    out = {}
    if cfg.arch == "diffusion":
        g = torch.Generator().manual_seed(0)
        T = cfg.diffusion.num_timesteps
        nb = cfg.train.noise_buckets
        noise_sum = ce_sum = 0.0
        bsum = torch.zeros(nb)
        bcnt = torch.zeros(nb)
        for i in range(0, n, bs):
            x = win.get(torch.arange(i, min(i + bs, n))).to(device)
            t = torch.randint(0, T, (x.size(0),), generator=g).to(device)
            noise = torch.randn(x.size(0), x.size(1), cfg.model.byte_embed_dim, generator=g).to(device)
            _, s = model(x, t=t, noise=noise)
            noise_sum += float(s["noise_loss"]) * x.size(0)
            ce_sum += float(s.get("ce_loss", 0.0)) * x.size(0)
            a, b = bucket_losses(s["t"], s["per_sample_noise_loss"], T, nb)
            bsum += a
            bcnt += b
        out = {"val_noise_loss": noise_sum / n, "val_ce_loss": ce_sum / n,
               "val_bucket_noise_loss": [(bsum[i] / bcnt[i]).item() if bcnt[i] > 0 else None for i in range(nb)]}
    else:
        nll = 0.0
        count = 0
        for i in range(0, n, bs):
            x = win.get(torch.arange(i, min(i + bs, n))).to(device)
            loss, _ = model(x)
            nll += float(loss) * x.size(0)
            count += x.size(0)
        out = {"val_bpb": nll / count / math.log(2)}
    model.train(was_training)
    return out


# --- Checkpoints ---

def check_compat(model, state_dict, what="model"):
    expected = model.state_dict()
    problems = []
    for k in expected:
        if k not in state_dict:
            problems.append(f"missing key {k}")
        elif tuple(state_dict[k].shape) != tuple(expected[k].shape):
            problems.append(f"{k}: checkpoint {tuple(state_dict[k].shape)} vs config {tuple(expected[k].shape)}")
    for k in state_dict:
        if k not in expected:
            problems.append(f"unexpected key {k}")
    if problems:
        shown = "; ".join(problems[:5]) + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else "")
        raise CheckpointError(f"checkpoint {what} is incompatible with the configured architecture: {shown}")


def _ema_state_dict(sd):
    if "num_updates" not in sd:
        sd = dict(sd)
        sd["num_updates"] = torch.zeros((), dtype=torch.long)
    return sd


def save_checkpoint(path, payload):
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint_into(ckpt, model, ema=None):
    check_compat(model, ckpt["model_state_dict"], "model")
    model.load_state_dict(ckpt["model_state_dict"])
    if ema is not None and "ema_state_dict" in ckpt:
        ema_sd = _ema_state_dict(ckpt["ema_state_dict"])
        check_compat(ema, ema_sd, "EMA")
        ema.load_state_dict(ema_sd)


def prepare_run_dir(run_dir, mode):
    ckpt = os.path.join(run_dir, CHECKPOINT_NAME)
    if mode == "fresh":
        if os.path.exists(ckpt) or (os.path.isdir(run_dir) and os.listdir(run_dir)):
            raise CheckpointError(f"{run_dir} is not empty; fresh runs need a new directory (use --resume to continue it)")
        os.makedirs(run_dir, exist_ok=True)
    elif mode == "resume":
        if not os.path.exists(ckpt):
            raise CheckpointError(f"no checkpoint at {ckpt} to resume from")
    return ckpt


def make_optimizer(model, t):
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [{"params": decay, "weight_decay": t.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return optim.AdamW(groups, lr=t.learning_rate)


# --- Training ---

def sample_text(model, n, seed):
    torch.manual_seed(seed)
    tok = ByteTokenizer()
    if hasattr(model, "generate"):
        dev = next(model.parameters()).device
        prompt = torch.full((n, 1), 10, dtype=torch.long, device=dev)
        out = model.generate(prompt, 200, temperature=0.8, top_k=40)[:, 1:].cpu().tolist()
    else:
        out = model.sample(batch_size=n).cpu().tolist()
    return [tok.decode(row) for row in out]


def run_training(cfg, run_dir, mode="fresh", device="auto", data=None, val=None, stop_after=None, quiet=False):
    """Train toward cfg.train.max_updates optimizer updates (or cfg.train.max_seconds of wall clock).

    stop_after halts (with a checkpoint) once that many total updates are done.
    """
    device = resolve_device(device)
    ckpt_path = prepare_run_dir(run_dir, mode)

    ckpt = None
    if mode == "resume":
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(ckpt["config"])
        if ckpt["version"] != CHECKPOINT_VERSION:
            raise CheckpointError(f"checkpoint version {ckpt['version']} != {CHECKPOINT_VERSION}")
    validate(cfg)
    t = cfg.train

    windows, val_tokens, data_hash = make_data(cfg, data, val)
    if ckpt is not None and ckpt["meta"]["dataset_hash"] != data_hash:
        raise CheckpointError("dataset differs from the one this run was started with; data order cannot be resumed")

    if ckpt is None:
        torch.manual_seed(t.seed)
    model = build_model(cfg).to(device)
    ema = EMA(model, decay=t.ema_decay, warmup=t.ema_warmup).to(device)
    optimizer = make_optimizer(model, t)
    clock = {"start": time.time()}
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(t.warmup_updates, t.max_updates, t.learning_rate, t.lr_min,
                                                                 lambda: time.time() - clock["start"], t.max_seconds))
    ac_dev, ac_dtype, use_scaler = amp_settings(t.precision, device)
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    num_params = sum(p.numel() for p in model.parameters())
    cuda = device == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    progress = {"updates": 0, "epoch": 0, "batch_offset": 0, "epoch_loss_sum": 0.0, "epoch_micro": 0,
                "examples": 0, "elapsed": 0.0, "clipped": 0}
    if ckpt is not None:
        load_checkpoint_into(ckpt, model, ema)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        progress = dict(ckpt["progress"])
        torch.set_rng_state(ckpt["rng"]["cpu"])
        if cuda and ckpt["rng"].get("cuda"):
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
    else:
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)

    meta = {"seed": t.seed, "dataset": t.data or t.data_dir, "dataset_hash": data_hash, "git_revision": git_revision(),
            "device": device, "param_count": num_params, "architecture_hash": cfg.architecture_hash(),
            "torch": torch.__version__, "precision": t.precision}
    if ckpt is None:
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    def say(msg):
        if not quiet:
            print(msg, flush=True)

    say(f"run={run_dir} preset={cfg.name} arch={cfg.arch} params={num_params:,} device={device} precision={t.precision} "
        f"updates={progress['updates']}/{t.max_updates} rev={meta['git_revision']}")

    def loss_fn(batch_idx):
        return model(windows.get(batch_idx).to(device))

    def save():
        rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if cuda else None}
        save_checkpoint(ckpt_path, {
            "version": CHECKPOINT_VERSION, "config": cfg.to_dict(), "meta": meta,
            "model_state_dict": model.state_dict(), "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "progress": progress, "rng": rng,
        })

    def log_eval():
        if val_tokens is None:
            return None
        row = {"eval": True, "update": progress["updates"], "elapsed": progress["elapsed"]}
        for which, m in (("", model), ("ema_", ema.model)):
            with torch.autocast(ac_dev, dtype=ac_dtype, enabled=bool(ac_dev)) if ac_dev else torch.autocast("cpu", enabled=False):
                for k, v in quick_eval(m, cfg, val_tokens, device, t.eval_batches).items():
                    row[which + k] = v
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        key = "val_bpb" if cfg.arch != "diffusion" else "val_noise_loss"
        say(f"  eval @ {progress['updates']}: raw {row[key]:.4f}  ema {row['ema_' + key]:.4f}  ({key})")
        return row

    start_time = time.time() - progress["elapsed"]
    clock["start"] = start_time
    params = list(model.parameters())
    seq = cfg.model.seq_len if cfg.arch == "diffusion" else cfg.lm.seq_len
    last_eval = [None]

    def finish(reason):
        progress["elapsed"] = time.time() - start_time
        if val_tokens is not None and t.eval_interval and last_eval[0] != progress["updates"]:
            log_eval()
        save()
        with open(os.path.join(run_dir, "status.json"), "w") as f:
            json.dump({"stopped": reason, "updates": progress["updates"], "elapsed": progress["elapsed"]}, f)
        return {"progress": progress, "run_dir": run_dir, "model": model, "ema": ema, "config": cfg, "stopped": reason}

    while progress["updates"] < t.max_updates:
        epoch = progress["epoch"]
        batches = epoch_batches(windows, cfg, epoch)
        sizes = group_sizes(len(batches), t.grad_accum_steps)
        pos = progress["batch_offset"]
        first_group = pos // t.grad_accum_steps
        for gi in range(first_group, len(sizes)):
            group = batches[pos:pos + sizes[gi]]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses, stats = accumulate_group(loss_fn, group, scaler if use_scaler else None, ac_dev, ac_dtype)

            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, t.grad_clip))
            if not math.isfinite(grad_norm) and not use_scaler:
                tt = torch.cat([s["t"] for s in stats if "t" in s]).tolist() if any("t" in s for s in stats) else []
                raise FloatingPointError(f"non-finite gradient norm at update {progress['updates']}; sampled timesteps {tt}")
            clipped = math.isfinite(grad_norm) and grad_norm > t.grad_clip
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)

            pos += len(group)
            progress["batch_offset"] = pos
            progress["updates"] += 1
            progress["examples"] += len(group) * t.batch_size
            progress["epoch_loss_sum"] += sum(losses)
            progress["epoch_micro"] += len(losses)
            progress["clipped"] += int(clipped)
            progress["elapsed"] = time.time() - start_time

            row = {
                "update": progress["updates"], "epoch": epoch, "lr": scheduler.get_last_lr()[0],
                "loss": sum(losses) / len(losses),
                "noise_loss": sum(float(s["noise_loss"]) for s in stats) / len(stats),
                "grad_norm": grad_norm, "clipped": clipped, "clip_fraction": progress["clipped"] / progress["updates"],
                "examples": progress["examples"], "bytes_seen": progress["examples"] * seq, "microbatches": len(losses),
                "elapsed": progress["elapsed"], "bytes_per_s": progress["examples"] * seq / max(progress["elapsed"], 1e-9),
                "ema_decay": ema.current_decay(),
            }
            if cuda:
                row["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 2**20
            if cfg.arch == "diffusion":
                nb = t.noise_buckets
                bsum, bcnt = torch.zeros(nb), torch.zeros(nb)
                for s in stats:
                    a, b = bucket_losses(s["t"], s["per_sample_noise_loss"], cfg.diffusion.num_timesteps, nb)
                    bsum += a
                    bcnt += b
                row["bucket_noise_loss"] = [(bsum[i] / bcnt[i]).item() if bcnt[i] > 0 else None for i in range(nb)]
                row["ce_loss"] = sum(float(s.get("ce_loss", 0.0)) for s in stats) / len(stats)
                row["emb_norm"] = sum(float(s["x_start_norm"]) for s in stats) / len(stats)
                row["x0_pred_norm"] = sum(float(s.get("x0_pred_norm", 0.0)) for s in stats) / len(stats)
            else:
                row["train_bpb"] = row["loss"] / math.log(2)
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if progress["updates"] % t.log_interval == 0 or progress["updates"] == t.max_updates:
                extra = f"ce {row['ce_loss']:.4f} " if "ce_loss" in row else f"bpb {row['train_bpb']:.3f} "
                say(f"update {row['update']}/{t.max_updates} lr {row['lr']:.2e} loss {row['loss']:.4f} {extra}"
                    f"gnorm {grad_norm:.3f} bytes {row['bytes_seen']:,} {row['bytes_per_s'] / 1000:.1f}kB/s t={progress['elapsed']:.0f}s")

            if t.eval_interval and progress["updates"] % t.eval_interval == 0:
                log_eval()
                last_eval[0] = progress["updates"]
            if t.sample_interval and progress["updates"] % t.sample_interval == 0:
                write_samples(run_dir, model, ema, progress["updates"], seeds=[0], quiet=quiet)

            if progress["updates"] >= t.max_updates:
                return finish("max_updates")
            if stop_after is not None and progress["updates"] >= stop_after:
                return finish("stop_after")
            if t.max_seconds and progress["elapsed"] >= t.max_seconds:
                return finish("max_seconds")
            if progress["updates"] % t.save_interval == 0:
                save()

        say(f"--- epoch {epoch} done: avg loss {progress['epoch_loss_sum'] / progress['epoch_micro']:.4f} over {progress['epoch_micro']} microbatches")
        progress["epoch"] += 1
        progress["batch_offset"] = 0
        progress["epoch_loss_sum"] = 0.0
        progress["epoch_micro"] = 0
    return finish("max_updates")


# --- Generation / loading ---

def write_samples(run_dir, model, ema, tag, seeds=(0,), quiet=False):
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
    out = {}
    for which, m in (("raw", model), ("ema", ema.model)):
        was_training = m.training
        out[which] = {str(s): sample_text(m, 1, s)[0] for s in seeds}
        m.train(was_training)
    with open(os.path.join(run_dir, "samples", f"samples_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    if not quiet:
        for which, d in out.items():
            for s, txt in d.items():
                print(f"[{which} seed={s}] {ascii(txt[:100])}")
    return out


def load_model_from_run(run_dir, which="ema", device="cpu"):
    path = os.path.join(run_dir, CHECKPOINT_NAME)
    if not os.path.exists(path):
        raise CheckpointError(f"no checkpoint at {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = build_model(cfg).to(device)
    ema = EMA(model, decay=cfg.train.ema_decay).to(device)
    load_checkpoint_into(ckpt, model, ema)
    chosen = ema.model if which == "ema" else model
    return chosen.eval(), cfg


def generate(run_dir=None, checkpoint=None, preset=None, which="ema", seeds=(0,), device="auto"):
    device = resolve_device(device)
    if run_dir:
        models = {w: load_model_from_run(run_dir, w, device)[0] for w in (("raw", "ema") if which == "both" else (which,))}
    else:
        if not (checkpoint and preset):
            raise CheckpointError("generate needs --run, or --checkpoint together with --preset")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = get_preset(preset)
        model = build_model(cfg).to(device)
        ema = EMA(model, decay=cfg.train.ema_decay).to(device)
        load_checkpoint_into(ckpt, model, ema)
        models = {"raw": model, "ema": ema.model}
        models = {w: m for w, m in models.items() if which in ("both", w)}
    results = {name: {s: sample_text(m, 1, s)[0] for s in seeds} for name, m in models.items()}
    if run_dir:
        os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)
        with open(os.path.join(run_dir, "samples", f"generate_{datetime.now():%Y%m%d_%H%M%S}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    return results


# --- CLI ---

def main(argv=None):
    p = argparse.ArgumentParser(description="ByteNoise training, evaluation and generation")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="start a fresh run, or resume one")
    tr.add_argument("--preset", default="compact", help=f"one of {sorted(PRESETS)}")
    tr.add_argument("--run-dir", help="output directory (default runs/<preset>-<timestamp>)")
    tr.add_argument("--resume", metavar="RUN_DIR", help="continue an existing run using its stored config")
    tr.add_argument("--max-updates", type=int)
    tr.add_argument("--max-seconds", type=float, help="wall-clock budget")
    tr.add_argument("--seed", type=int)
    tr.add_argument("--data", help="prepared dataset name under data/ (e.g. natural-v1)")
    tr.add_argument("--precision", choices=["fp32", "bf16", "fp16"])
    tr.add_argument("--set", nargs="*", default=[], metavar="train.key=value", help="override other TrainConfig fields")
    tr.add_argument("--device", default="auto")

    ge = sub.add_parser("generate", help="sample from a checkpoint")
    ge.add_argument("--run")
    ge.add_argument("--checkpoint", help="legacy .pth file (requires --preset)")
    ge.add_argument("--preset")
    ge.add_argument("--which", choices=["raw", "ema", "both"], default="ema")
    ge.add_argument("--seeds", type=int, nargs="+", default=[0])
    ge.add_argument("--device", default="auto")

    ev = sub.add_parser("eval", help="evaluation report for a run (see eval.py)")
    ev.add_argument("rest", nargs=argparse.REMAINDER)

    args = p.parse_args(argv)
    try:
        if args.cmd == "train":
            if args.resume:
                if any(v is not None for v in (args.max_updates, args.max_seconds, args.seed, args.data, args.precision)) or args.set:
                    p.error("--resume uses the run's stored config; do not pass overrides")
                run_training(None, args.resume, "resume", args.device)
            else:
                cfg = get_preset(args.preset)
                ov = {k: v for k, v in (("max_updates", args.max_updates), ("max_seconds", args.max_seconds), ("seed", args.seed),
                                        ("data", args.data), ("precision", args.precision)) if v is not None}
                for item in args.set:
                    k, v = item.split("=", 1)
                    k = k.removeprefix("train.")
                    cur = getattr(cfg.train, k)
                    ov[k] = (v.lower() in ("1", "true")) if isinstance(cur, bool) else type(cur)(v)
                cfg = validate(override(cfg, **ov))
                run_dir = args.run_dir or os.path.join("runs", f"{cfg.name}-{datetime.now():%Y%m%d_%H%M%S}")
                run_training(cfg, run_dir, "fresh", args.device)
        elif args.cmd == "eval":
            import eval as E
            E.main(args.rest)
        else:
            res = generate(args.run, args.checkpoint, args.preset, args.which, args.seeds, args.device)
            for name, d in res.items():
                for s, txt in d.items():
                    print(f"[{name} seed={s}] {ascii(txt)}")
    except (CheckpointError, ValueError) as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
