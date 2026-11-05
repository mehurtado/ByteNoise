"""Ladder evaluation harness for diffusion runs (EMA weights).

Per run: 128 paired samples (noise seeds fixed per batch, so different runs see the same initial noise) at s=1.0 and
s=0.8 (1000-step DDPM; s scales the injected posterior noise, logits runs only) -> word_hit (+bootstrap CI), valid_utf8,
distinct_4gram, repeated_8gram, memorization, n-gram bits/byte (+bootstrap CI) under a Kneser-Ney byte scorer trained on
natural-v1 train (experiments/ngram_score.py), with real-val and shuffled controls; deterministic val CE overall / t in
[500,750) / t>=750 (logits runs); s/sample at batch 32, params, VRAM, ms/update from metrics.jsonl.
Writes ladder_eval.json + ladder_eval.md (+ ladder_samples.json) into --out-dir (default: the run dir).
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import torch
import data, train, eval as ev
from gate_text import wordlist, WORD
from sampler_sweep import uniform_steps
from sampler_sweep_logits import sample_logits
import ngram_score as ns

BATCH, SEED0 = 32, 1000


def per_sample_words(samples, words):
    hit, tot = [], []
    for s in samples:
        ws = WORD.findall(bytes(s))
        tot.append(len(ws)); hit.append(sum(w.lower() in words for w in ws))
    return np.array(hit), np.array(tot)


def boot_ci(num, den, n=2000, seed=0):
    """Bootstrap over samples of sum(num)/sum(den) (den=None -> mean of num). Returns (point, lo, hi)."""
    num = np.asarray(num, float); den = None if den is None else np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (n, len(num)))
    f = (lambda i: num[i].mean(1)) if den is None else (lambda i: num[i].sum(1) / np.maximum(den[i].sum(1), 1))
    pt = num.mean() if den is None else num.sum() / max(den.sum(), 1)
    lo, hi = np.percentile(f(idx), [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def make_samples(model, logits_mode, s, steps, n):
    """n samples in batches of BATCH with per-batch fixed seeds. Returns (bytes list, s/sample, peak VRAM MB)."""
    out, times = [], []
    torch.cuda.reset_peak_memory_stats()
    for b in range(n // BATCH):
        torch.manual_seed(SEED0 + b)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ids = sample_logits(model, BATCH, steps, s) if logits_mode else model.sample(BATCH).cpu().to(torch.uint8)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        out += [bytes(r.tolist()) for r in ids]
    sps = float(np.mean(times[1:] if len(times) > 1 else times)) / BATCH  # drop the first (warm-up) batch
    return out, sps, torch.cuda.max_memory_allocated() / 2**20


def text_block(samples, words, kn, mem_bytes):
    m = ev.text_metrics(samples, mem_bytes)
    hit, tot = per_sample_words(samples, words)
    bpb = ns.score_bpb(kn, samples)
    wh = boot_ci(hit, tot); nb = boot_ci(bpb, None)
    return dict(word_hit=wh[0], word_hit_ci=[wh[1], wh[2]], ngram_bpb=nb[0], ngram_bpb_ci=[nb[1], nb[2]],
                valid_utf8=m["valid_utf8_rate"], printable=m["printable_rate"], distinct_4gram=m["distinct_4gram_ratio"],
                repeated_8gram=m["repeated_8gram_fraction"], memorization=m.get("memorization_rate"),
                n_words=int(tot.sum()), per_sample=dict(word_hit=hit.tolist(), words=tot.tolist(), ngram_bpb=bpb.tolist()))


@torch.no_grad()
def val_ce(model, cfg, val_tokens, n=2048, bs=32, seed=0, data_name=None):
    """Deterministic CE (nats/byte): same windows, t and noise for every run. Regions: all t, [500,750), [750,1000)."""
    win = train.make_windows(val_tokens, cfg, "val") if cfg.train.doc_aligned and cfg.train.data and data_name in (None, cfg.train.data) else train.Windows(val_tokens, train.window_len(cfg))
    T = cfg.diffusion.num_timesteps
    idx_all = torch.linspace(0, len(win) - 1, n // 2).long().repeat(2)
    res = {}
    for name, lo, hi in (("overall", 0, T), ("t500_750", 500 * T // 1000, 750 * T // 1000), ("t750_1000", 750 * T // 1000, T)):
        g = torch.Generator().manual_seed(seed)
        ce = acc = 0.0
        for i in range(0, n, bs):
            x = win.get(idx_all[i:i + bs]).to("cuda")
            t = torch.randint(lo, hi, (x.size(0),), generator=g).to("cuda")
            noise = torch.randn(x.size(0), x.size(1), cfg.model.byte_embed_dim, generator=g).to("cuda")
            _, st = model(x, t=t, noise=noise)
            ce += float(st["ce_loss"]) * x.size(0); acc += float(st["x0_acc"]) * x.size(0)
        res[name] = dict(ce=ce / n, x0_acc=acc / n)
    return res


def train_stats(run):
    p = os.path.join(run, "metrics.jsonl")
    out = {}
    if not os.path.exists(p):
        return out
    rows = [json.loads(l) for l in open(p)]
    tr = [r for r in rows if not r.get("eval")]
    ev_rows = [r for r in rows if r.get("eval")]
    if len(tr) > 100:
        u = np.array([r["update"] for r in tr]); e = np.array([r["elapsed"] for r in tr])
        d = np.diff(e) / np.maximum(np.diff(u), 1)
        out["ms_per_update"] = float(np.median(d[d > 0]) * 1000) if (d > 0).any() else None
        out["updates"] = int(u[-1])
        v = [r["peak_vram_mb"] for r in tr if "peak_vram_mb" in r]
        out["train_peak_vram_mb"] = max(v) if v else None
    key = "ema_val_ce_loss"
    out["val_ce_traj"] = {int(r["update"]): r[key] for r in ev_rows if key in r}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--data", default="natural-v1", help="word list + n-gram scorer data (train split)")
    ap.add_argument("--which", default="ema")
    a = ap.parse_args()
    out_dir = a.out_dir or a.run
    os.makedirs(out_dir, exist_ok=True)
    model, cfg = train.load_model_from_run(a.run, a.which, "cuda"); model.eval()
    gd = model.gaussian_diffusion; T = gd.num_timesteps
    logits_mode = gd.param == "logits"
    tb = data.load_split(a.data, "train").numpy().tobytes()
    words = wordlist(tb)
    mem_name = cfg.train.data or a.data
    mem_bytes = tb if mem_name == a.data else data.load_split(mem_name, "train").numpy().tobytes()
    kn = ns.load_or_train(a.data)
    nparams = sum(p.numel() for p in model.parameters())
    R = dict(run=a.run, name=cfg.name, denoiser=cfg.model.denoiser, model_dim=cfg.model.model_dim, param=gd.param,
             precision=cfg.train.precision, seed=cfg.train.seed, params=nparams, n_samples=a.n, data=a.data,
             memorization_vs=mem_name, wordlist_size=len(words), train=train_stats(a.run), samplers={})

    # controls (independent of the model): real val windows at fixed starts, shuffled letters, shuffled bytes
    val = data.load_split(a.data, "val"); L = cfg.model.seq_len
    g = torch.Generator().manual_seed(0)
    if cfg.train.doc_aligned and os.path.exists(os.path.join(data.DATA_ROOT, a.data, "documents.jsonl")):  # real controls drawn from document-aligned windows, like the training windows
        ds = data.doc_window_starts(a.data, "val", L)[0]
        starts = torch.as_tensor(ds)[torch.randint(0, len(ds), (a.n,), generator=g)]
    else:
        starts = torch.randint(0, len(val) - L, (a.n,), generator=g)
    real = [bytes(val[s:s + L].numpy().tobytes()) for s in starts]
    rng = np.random.default_rng(0)

    def shuf_letters(b):
        arr = np.frombuffer(b, dtype=np.uint8).copy()
        m = ((arr | 32) >= 97) & ((arr | 32) <= 122)
        arr[m] = rng.permutation(arr[m])
        return bytes(arr)

    R["controls"] = {"real_val": text_block(real, words, kn, tb),
                     "shuffled_letters": text_block([shuf_letters(b) for b in real], words, kn, tb),
                     "shuffled_bytes": text_block([bytes(rng.permutation(np.frombuffer(b, dtype=np.uint8))) for b in real], words, kn, tb)}

    full = uniform_steps(T, T)
    all_samples = {}
    for s in (1.0, 0.8):
        if not logits_mode and s != 1.0:
            R["samplers"][f"s{s}"] = None  # eps runs: no s-scaled sampler implemented (skipped)
            continue
        samples, sps, vram = make_samples(model, logits_mode, s, full, a.n)
        blk = text_block(samples, words, kn, mem_bytes)
        blk.update(sec_per_sample_b32=sps, sample_peak_vram_mb=vram)
        R["samplers"][f"s{s}"] = blk
        all_samples[f"s{s}"] = [x.hex() for x in samples]
        print(f"[{a.run}] s={s} wh={blk['word_hit']:.3f} {blk['word_hit_ci']} bpb={blk['ngram_bpb']:.3f} utf8={blk['valid_utf8']:.3f} "
              f"d4={blk['distinct_4gram']:.3f} {sps:.3f}s/sample", flush=True)
    if logits_mode:
        R["val_ce"] = val_ce(model, cfg, data.load_split(a.data, "val"), data_name=a.data)
        print(f"[{a.run}] val_ce {R['val_ce']}", flush=True)
    else:
        R["val_ce"] = None

    json.dump(R, open(os.path.join(out_dir, "ladder_eval.json"), "w"), indent=1)
    json.dump(all_samples, open(os.path.join(out_dir, "ladder_samples.json"), "w"))
    open(os.path.join(out_dir, "ladder_eval.md"), "w", encoding="utf-8").write(to_md(R))
    print("wrote", out_dir, flush=True)


def to_md(R):
    t = R["train"]
    msu = t.get("ms_per_update")
    L = [f"# ladder_eval: {R['run']}", "",
         f"denoiser={R['denoiser']} model_dim={R['model_dim']} param={R['param']} precision={R['precision']} seed={R['seed']} "
         f"params={R['params']:,} updates={t.get('updates')} ms/update={msu and round(msu, 1)} "
         f"train peak VRAM={t.get('train_peak_vram_mb') and round(t['train_peak_vram_mb'])} MB; {R['n_samples']} paired samples, EMA, "
         f"1000 steps; memorization vs {R['memorization_vs']} train.", "",
         "| set | word_hit [95% CI] | n-gram bpb [95% CI] | valid_utf8 | distinct_4gram | repeated_8gram | memorization | s/sample (b32) |",
         "|---|---|---|---|---|---|---|---|"]

    def row(name, b):
        if b is None:
            return f"| {name} | n/a | | | | | | |"
        f = lambda x, d=3: "-" if x is None else f"{x:.{d}f}"
        return (f"| {name} | {b['word_hit']:.3f} [{b['word_hit_ci'][0]:.3f}, {b['word_hit_ci'][1]:.3f}] | {b['ngram_bpb']:.3f} "
                f"[{b['ngram_bpb_ci'][0]:.3f}, {b['ngram_bpb_ci'][1]:.3f}] | {b['valid_utf8']:.3f} | {b['distinct_4gram']:.3f} | "
                f"{b['repeated_8gram']:.4f} | {f(b.get('memorization'))} | {f(b.get('sec_per_sample_b32'), 3)} |")
    for k, b in R["samplers"].items():
        L.append(row(f"samples {k}", b))
    for k, b in R["controls"].items():
        L.append(row(f"control {k}", b))
    if R.get("val_ce"):
        L += ["", "| val CE (nats/byte, EMA, deterministic) | CE | x0 acc |", "|---|---|---|"]
        for k, v in R["val_ce"].items():
            L.append(f"| {k} | {v['ce']:.4f} | {v['x0_acc']:.4f} |")
    if t.get("val_ce_traj"):
        L += ["", "val CE (EMA) trajectory from metrics.jsonl: " + ", ".join(f"{u}: {c:.4f}" for u, c in sorted(t["val_ce_traj"].items()))]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
