"""Sampler sweep for diffusion.param=logits checkpoints (no retraining).

Own loop mirrors GaussianDiffusion.p_sample (logits path): x0_hat = softmax(logits) @ E, DDPM posterior on x0_hat,
self-cond carries x0_hat, final decode = argmax of the LAST step's logits, x0_rescale ignored. Uses the global torch RNG
in the same call order as model.sample, so s=1.0 with all T steps is bit-identical to the control (checked at runtime).
s scales the posterior std; respaced steps use the generalized posterior between non-adjacent timesteps.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import data, train
from gate_text import wordlist
from sampler_sweep import uniform_steps, late_heavy_steps, metrics


@torch.no_grad()
def sample_logits(model, n, steps, s=1.0):
    gd = model.gaussian_diffusion
    dev = gd.betas.device
    table = model.embedding_table().detach()
    shape = (n, gd.seq_len, model.byte_embedder.embedding.embedding_dim)
    full = len(steps) == gd.num_timesteps
    x = torch.randn(shape, device=dev)
    x0_prev = None
    ac = gd.alphas_cumprod.double()
    for j, i in enumerate(steps):
        t = torch.full((n,), i, device=dev, dtype=torch.long)
        logits = gd.denoise(x, t, x0_prev=x0_prev if gd.self_cond else None).float()
        x0 = gd.x0_from_logits(logits, table)
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
    jh = getattr(gd.denoise_model, 'joint', None)
    if jh is not None and jh.mode == 'joint':  # joint head: bounded candidate selection under the full mixture (state of the last step's denoise call)
        return jh.joint_decode().cpu().to(torch.uint8)
    return logits.argmax(-1).cpu().to(torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/diff_nat_logits_warp_s0", "runs/diff_nat_logits_s0"])
    ap.add_argument("--data", default="natural-v1")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/sampler_sweep_logits_results.json")
    a = ap.parse_args()
    dev = "cuda"
    tb = data.load_split(a.data, "train").numpy().tobytes()
    words = wordlist(tb)
    allres = {}
    for run in a.runs:
        model, cfg = train.load_model_from_run(run, "ema", dev)
        model.eval()
        assert model.gaussian_diffusion.param == "logits"
        T = model.gaussian_diffusion.num_timesteps
        res = {}

        def go(name, steps, s):
            torch.manual_seed(a.seed); model.sample(1)  # warmup, same RNG protocol as gate_text
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ids = sample_logits(model, a.n, steps, s)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            m, smp = metrics(ids, words, None)
            m.update(sec_per_sample=dt / a.n, nsteps=len(steps), s=s, snips=[ascii(x[:160]) for x in smp[:2]])
            res[name] = m
            print(f"[{run}] {name:<22} wh={m['word_hit']:.4f} utf8={m['valid_utf8_rate']:.3f} "
                  f"print={m['printable_rate']:.3f} d4={m['distinct_4gram_ratio']:.3f} "
                  f"r8={m['repeated_8gram_fraction']:.4f} {dt/a.n:.2f}s/sample", flush=True)
            return ids

        torch.manual_seed(a.seed); model.sample(1)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        cid = model.sample(a.n).cpu().to(torch.uint8)
        dt = time.perf_counter() - t0
        m, smp = metrics(cid, words, None)
        m.update(sec_per_sample=dt / a.n, nsteps=T, s=1.0, snips=[ascii(x[:160]) for x in smp[:2]])
        res["control_model.sample"] = m
        print(f"[{run}] control_model.sample wh={m['word_hit']:.4f} utf8={m['valid_utf8_rate']:.3f} "
              f"d4={m['distinct_4gram_ratio']:.3f} {dt/a.n:.2f}s/sample", flush=True)
        full = uniform_steps(T, T)
        ids = go("full1000_s1.0", full, 1.0)
        same = bool((ids == cid).all())
        res["own_loop_matches_control"] = same
        print(f"[{run}] own loop == control bytes: {same}", flush=True)
        for s in (0.8, 0.6, 0.5, 0.25):
            go(f"full1000_s{s}", full, s)
        u250 = uniform_steps(T, 250)
        for s in (1.0, 0.8, 0.5):
            go(f"uni250_s{s}", u250, s)
        bs = max((1.0, 0.8, 0.6, 0.5, 0.25), key=lambda s: res[f"full1000_s{s}"]["word_hit"])
        go(f"late250_s{bs}", late_heavy_steps(T, 250), bs)
        res["best_s_by_word_hit"] = bs
        allres[run] = res
        json.dump(allres, open(a.out, "w"), indent=1)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
