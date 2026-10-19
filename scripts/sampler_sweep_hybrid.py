"""Sampler-only sweep on a finished logits checkpoint (EMA), same 128 paired samples / metrics as ladder_eval.

Options (defaults reproduce sampler_sweep_logits.sample_logits bit-for-bit; checked at runtime against the 1000-step baseline):
  steps     : respaced timestep list (uniform_steps, or hi_steps = fraction `hi_frac` of the budget in t>=hi_t, rest uniform below)
  s         : scale of injected posterior noise std
  tau       : logit temperature in x0_hat = softmax(logits / tau) @ E (also feeds self-conditioning); final decode stays argmax(logits)
  refine    : extra denoise passes after the last step: re-noise the current x0_hat to each t in `refine` (descending),
              denoise again (self-cond = previous x0_hat), x0_hat <- softmax(logits/tau) @ E; decode = argmax of the last logits.
Reads the run read-only; writes only under --out-dir.
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import torch
import data, train
from gate_text import wordlist, WORD
from sampler_sweep import uniform_steps
import ngram_score as ns
import ladder_eval as le


def hi_steps(T, k, hi_frac, hi_t=750):
    """k steps total: round(hi_frac*k) evenly spaced in [hi_t, T-1], the rest evenly spaced in [0, hi_t). Descending."""
    k_hi = int(round(hi_frac * k)); k_lo = k - k_hi
    s = set(np.round(np.linspace(hi_t, T - 1, k_hi)).astype(int).tolist())
    s |= set(np.round(np.linspace(0, hi_t, k_lo, endpoint=False)).astype(int).tolist())
    s |= {0, T - 1}
    return sorted(s, reverse=True)


@torch.no_grad()
def sample_opt(model, n, steps, s=1.0, tau=1.0, refine=()):
    gd = model.gaussian_diffusion
    dev = gd.betas.device
    table = model.embedding_table().detach()
    shape = (n, gd.seq_len, model.byte_embedder.embedding.embedding_dim)
    full = len(steps) == gd.num_timesteps
    x = torch.randn(shape, device=dev)
    x0_prev = None
    ac = gd.alphas_cumprod.double()
    x0 = None
    for j, i in enumerate(steps):
        t = torch.full((n,), i, device=dev, dtype=torch.long)
        logits = gd.denoise(x, t, x0_prev=x0_prev if gd.self_cond else None).float()
        x0 = gd.x0_from_logits(logits if tau == 1.0 else logits / tau, table)
        x0_prev = x0
        last = j + 1 == len(steps)
        if full:
            mean, _, log_var = gd.q_posterior(x0, x, t)
            sd = (0.5 * log_var).exp()
        else:
            ab_t = ac[i]
            ab_p = ac[steps[j + 1]] if not last else torch.tensor(1.0, dtype=torch.double, device=dev)
            a_eff = ab_t / ab_p; beta = 1 - a_eff
            mean = (ab_p.sqrt() * beta / (1 - ab_t)).float() * x0 + (a_eff.sqrt() * (1 - ab_p) / (1 - ab_t)).float() * x
            sd = ((1 - ab_p) / (1 - ab_t) * beta).sqrt().float()
        x = mean if last else mean + sd * s * torch.randn_like(x)
    for tr in refine:
        t = torch.full((n,), int(tr), device=dev, dtype=torch.long)
        xt = gd.q_sample(x0, t)
        logits = gd.denoise(xt, t, x0_prev=x0_prev if gd.self_cond else None).float()
        x0 = gd.x0_from_logits(logits if tau == 1.0 else logits / tau, table)
        x0_prev = x0
    return logits.argmax(-1).cpu().to(torch.uint8)


def make_samples(model, n, steps, s, tau, refine):
    out, times = [], []
    torch.cuda.reset_peak_memory_stats()
    for b in range(n // le.BATCH):
        torch.manual_seed(le.SEED0 + b)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ids = sample_opt(model, le.BATCH, steps, s, tau, refine)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
        out += [bytes(r.tolist()) for r in ids]
    sps = float(np.mean(times[1:] if len(times) > 1 else times)) / le.BATCH
    return out, sps, torch.cuda.max_memory_allocated() / 2**20


def paired_diff(a, b, den=None, n=2000, seed=0):
    """Paired bootstrap of metric(a)-metric(b) over samples. a,b arrays (num; with den -> ratio of sums)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(a), (n, len(a)))
    if den is None:
        d = a[idx].mean(1) - b[idx].mean(1); pt = a.mean() - b.mean()
    else:
        da, db = den
        d = a[idx].sum(1) / np.maximum(da[idx].sum(1), 1) - b[idx].sum(1) / np.maximum(db[idx].sum(1), 1)
        pt = a.sum() / max(da.sum(), 1) - b.sum() / max(db.sum(), 1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(pt), float(lo), float(hi)


def word_len(samples):
    L = [len(w) for s in samples for w in WORD.findall(bytes(s))]
    return float(np.mean(L)) if L else 0.0


def configs(T):
    C = []  # (name, group, steps, s, tau, refine)
    U = lambda k: uniform_steps(T, k)
    for s in (1.0, 0.8):
        for k in (1000, 500, 250, 125, 64):
            C.append((f"uni{k}_s{s}", "steps", U(k), s, 1.0, ()))
    for s in (1.0, 0.8):
        for k in (250, 125):
            for f in (0.5, 0.75):
                C.append((f"hi{int(f*100)}_{k}_s{s}", "spacing", hi_steps(T, k, f), s, 1.0, ()))
    for s in (1.0, 0.8):
        for tau in (0.7, 0.85):
            C.append((f"uni250_s{s}_tau{tau}", "tau", U(250), s, tau, ()))
    for tau in (0.7, 0.85):
        C.append((f"uni1000_s1.0_tau{tau}", "tau", U(1000), 1.0, tau, ()))
    for s in (1.0, 0.8):
        for r in ((15,), (25, 10)):
            C.append((f"uni250_s{s}_ref{'-'.join(map(str, r))}", "refine", U(250), s, 1.0, r))
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/diff_nat_v2_hybrid_bs128_s0")
    ap.add_argument("--out-dir", default="runs/_sampler_hybrid")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--data", default="natural-v1")
    ap.add_argument("--only", default=None, help="substring filter on config names")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    model, cfg = train.load_model_from_run(a.run, "ema", "cuda"); model.eval()
    gd = model.gaussian_diffusion; T = gd.num_timesteps
    assert gd.param == "logits"
    tb = data.load_split(a.data, "train").numpy().tobytes()
    words = wordlist(tb)
    mem_name = cfg.train.data or a.data
    mem_bytes = tb if mem_name == a.data else data.load_split(mem_name, "train").numpy().tobytes()
    kn = ns.load_or_train(a.data)
    jp = os.path.join(a.out_dir, "results.json")
    R = json.load(open(jp)) if os.path.exists(jp) else {}
    samples_all = {}
    base = {}
    ref_samples = os.path.join(a.run, "ladder_samples.json")
    ref = json.load(open(ref_samples)) if os.path.exists(ref_samples) else {}
    for name, group, steps, s, tau, refine in configs(T):
        if a.only and a.only not in name and not name.startswith("uni1000_s"):
            continue
        samples, sps, vram = make_samples(model, a.n, steps, s, tau, refine)
        blk = le.text_block(samples, words, kn, mem_bytes)
        blk.update(group=group, nsteps=len(steps), s=s, tau=tau, refine=list(refine), sec_per_sample_b32=sps,
                   peak_vram_mb=vram, mean_word_len=word_len(samples))
        if name == f"uni1000_s{s}":
            same = ref.get(f"s{s}") == [x.hex() for x in samples] if ref else None
            blk["matches_ladder_samples"] = same
            base[s] = blk
        R[name] = blk
        samples_all[name] = [x.hex() for x in samples]
        print(f"{name:<26} n={len(steps):4d} wh={blk['word_hit']:.3f} [{blk['word_hit_ci'][0]:.3f},{blk['word_hit_ci'][1]:.3f}] "
              f"bpb={blk['ngram_bpb']:.3f} utf8={blk['valid_utf8']:.3f} d4={blk['distinct_4gram']:.3f} wl={blk['mean_word_len']:.2f} "
              f"{sps:.3f}s/sample {vram:.0f}MB match={blk.get('matches_ladder_samples')}", flush=True)
        json.dump(R, open(jp, "w"), indent=1)
        sp = os.path.join(a.out_dir, "samples.json")
        old = json.load(open(sp)) if os.path.exists(sp) else {}
        old.update(samples_all); json.dump(old, open(sp, "w"))
    # paired differences vs the same-s 1000-step baseline
    for name, b in R.items():
        bb = R.get(f"uni1000_s{b['s']}")
        if bb is None or name == f"uni1000_s{b['s']}":
            continue
        ps, pb = b["per_sample"], bb["per_sample"]
        b["d_word_hit_vs_base"] = paired_diff(ps["word_hit"], pb["word_hit"], (np.array(ps["words"]), np.array(pb["words"])))
        b["d_bpb_vs_base"] = paired_diff(ps["ngram_bpb"], pb["ngram_bpb"])
    json.dump(R, open(jp, "w"), indent=1)
    rows = ["| config | steps | s/sample | VRAM MB | word_hit [CI] | d vs base (CI) | bpb | d bpb (CI) | utf8 | d4 | r8 | mem | word len |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, b in R.items():
        dw = b.get("d_word_hit_vs_base"); db = b.get("d_bpb_vs_base")
        fd = lambda d: "-" if d is None else f"{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]" + ("*" if d[1] > 0 or d[2] < 0 else "")
        rows.append(f"| {name} | {b['nsteps']} | {b['sec_per_sample_b32']:.3f} | {b['peak_vram_mb']:.0f} | {b['word_hit']:.3f} "
                    f"[{b['word_hit_ci'][0]:.3f},{b['word_hit_ci'][1]:.3f}] | {fd(dw)} | {b['ngram_bpb']:.3f} | {fd(db)} | "
                    f"{b['valid_utf8']:.3f} | {b['distinct_4gram']:.3f} | {b['repeated_8gram']:.4f} | {(b['memorization'] or 0):.3f} | {b['mean_word_len']:.2f} |")
    open(os.path.join(a.out_dir, "results.md"), "w").write("\n".join(rows) + "\n(* = paired bootstrap 95% CI excludes 0)\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
