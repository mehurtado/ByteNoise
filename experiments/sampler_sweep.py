"""Sampler sweep on existing natural-text diffusion checkpoints (no retraining).

(a) posterior noise scale s, (b) timestep respacing (generalized DDPM posterior between non-adjacent steps),
(c) best combination. Same 32 samples / seed protocol as gate_text (metrics via gate_text.report logic).
Keeps x0_rescale step and self-cond x0_prev carry (x0_prev = rescaled x0_hat, as in GaussianDiffusion.p_sample).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import data, eval as ev, train
from gate_text import wordlist, word_hit


def uniform_steps(T, k):
    return sorted(set(np.round(np.linspace(0, T - 1, k)).astype(int).tolist()), reverse=True)


def late_heavy_steps(T, k, lo=500, hi=900, frac=0.6):
    k_mid = int(k * frac); k_hi = (k - k_mid) // 2; k_lo = k - k_mid - k_hi
    s = set(np.round(np.linspace(lo, hi, k_mid, endpoint=False)).astype(int).tolist())
    s |= set(np.round(np.linspace(hi, T - 1, k_hi)).astype(int).tolist())
    s |= set(np.round(np.linspace(0, lo, k_lo, endpoint=False)).astype(int).tolist())
    s |= {0, T - 1}
    return sorted(s, reverse=True)


@torch.no_grad()
def sample(model, n, steps, s=1.0, seed=0):
    gd = model.gaussian_diffusion
    dev = gd.betas.device
    R = model.mean_embed_norm()
    g = torch.Generator(device=dev).manual_seed(seed)
    shape = (n, gd.seq_len, model.byte_embedder.embedding.embedding_dim)
    x = torch.randn(shape, device=dev, generator=g)
    x0_prev = None
    ac = gd.alphas_cumprod.double()
    for j, i in enumerate(steps):
        t = torch.full((n,), i, device=dev, dtype=torch.long)
        eps = gd.denoise(x, t, x0_prev=x0_prev)
        x0 = gd.predict_start_from_noise(x, t, eps)
        x0 = x0 * (R / x0.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        if gd.self_cond:
            x0_prev = x0
        ab_t = ac[i]
        ab_p = ac[steps[j + 1]] if j + 1 < len(steps) else torch.tensor(1.0, dtype=torch.double, device=dev)
        a_eff = ab_t / ab_p
        beta_t = 1 - a_eff
        c1 = (ab_p.sqrt() * beta_t / (1 - ab_t)).float()
        c2 = (a_eff.sqrt() * (1 - ab_p) / (1 - ab_t)).float()
        mean = c1 * x0 + c2 * x
        if j + 1 == len(steps):
            x = mean
        else:
            sd = ((1 - ab_p) / (1 - ab_t) * beta_t).sqrt().float() * s
            x = mean + sd * torch.randn(shape, device=dev, generator=g)
    return model.get_logits(x).argmax(-1).cpu().to(torch.uint8)


def metrics(ids, words, tb):
    samples = [bytes(r.tolist()) for r in ids]
    m = ev.text_metrics(samples, tb)
    m["word_hit"], m["n_words"] = word_hit(samples, words)
    return m, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/diff_nat_gate_s0", "runs/diff_nat_hybrid_selfcond_s0"])
    ap.add_argument("--data", default="natural-v1")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/sampler_sweep_results.json")
    a = ap.parse_args()
    dev = "cuda"
    tb = data.load_split(a.data, "train").numpy().tobytes()
    words = wordlist(tb)
    allres = {}
    for run in a.runs:
        model, cfg = train.load_model_from_run(run, "ema", dev)
        model.eval()
        T = model.gaussian_diffusion.num_timesteps
        res = {}
        def go(name, steps, s):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ids = sample(model, a.n, steps, s, a.seed)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            m, smp = metrics(ids, words, tb)
            m["sec_per_sample"] = dt / a.n; m["nsteps"] = len(steps); m["s"] = s
            m["snips"] = [ascii(x[:160]) for x in smp[:2]]
            res[name] = m
            print(f"[{run}] {name:<26} wh={m['word_hit']:.3f} utf8={m.get('valid_utf8',float('nan')):.3f} "
                  f"print={m.get('printable',float('nan')):.3f} d4={m.get('distinct_4gram',float('nan')):.3f} "
                  f"r8={m.get('repeated_8gram',float('nan')):.3f} {dt/a.n:.2f}s/sample", flush=True)
        # gate-protocol control with the model's own sampler
        torch.manual_seed(a.seed); model.sample(1)
        ids = model.sample(a.n).cpu().to(torch.uint8)
        m, smp = metrics(ids, words, tb); m["snips"] = [ascii(x[:160]) for x in smp[:2]]
        res["control_model.sample"] = m
        print(f"[{run}] control_model.sample wh={m['word_hit']:.3f}", flush=True)
        full = uniform_steps(T, T)
        for s in (1.0, 0.8, 0.5, 0.25, 0.0):
            go(f"full1000_s{s}", full, s)
        sched = {"uni250": uniform_steps(T, 250), "uni500": uniform_steps(T, 500),
                 "late250": late_heavy_steps(T, 250), "late500": late_heavy_steps(T, 500)}
        for nm, st in sched.items():
            go(f"{nm}_s1.0", st, 1.0)
        # (c) combos: best schedule x best s among (a)
        bs = max((1.0, 0.8, 0.5, 0.25, 0.0), key=lambda s: res[f"full1000_s{s}"]["word_hit"])
        bn = max(sched, key=lambda k: res[f"{k}_s1.0"]["word_hit"])
        for nm in dict.fromkeys([bn, "uni250"]):
            for s in dict.fromkeys([bs, 0.5]):
                if s != 1.0:
                    go(f"{nm}_s{s}", sched[nm], s)
        allres[run] = res
        json.dump(allres, open(a.out, "w"), indent=1)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
