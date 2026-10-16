"""Alternative samplers for ByteDiffusionLM: clamp x0_hat each step, then DDPM posterior (or DDIM eta=0).

Variants: none (control), abs (elementwise clamp to [-A,A], A=embedding absmax), norm (rescale each position
vector to mean real embedding norm), both (abs then norm). Does not touch main.py's default sampler.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import train
from main import extract
from sample_validity import stats, ALPHA


@torch.no_grad()
def sample_variant(model, n, clamp="none", ddim=False, seed=0, A=None, R=None, final_clip=False):
    gd = model.gaussian_diffusion
    W = model.byte_embedder.embedding.weight
    A = W.abs().max().item() if A is None else A
    R = W.norm(dim=-1).mean().item() if R is None else R
    dev = W.device
    g = torch.Generator(device=dev).manual_seed(seed)
    shape = (n, gd.seq_len, W.shape[1])
    x = torch.randn(shape, device=dev, generator=g)
    for i in reversed(range(gd.num_timesteps)):
        t = torch.full((n,), i, device=dev, dtype=torch.long)
        eps = gd.denoise_model(x, t)
        x0 = gd.predict_start_from_noise(x, t, eps)
        if clamp in ("abs", "both"):
            x0 = x0.clamp(-A, A)
        if clamp in ("norm", "both"):
            x0 = x0 * (R / x0.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        if ddim:
            ab = extract(gd.alphas_cumprod, t, x.shape)
            abp = extract(gd.alphas_cumprod_prev, t, x.shape)
            e2 = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
            x = abp.sqrt() * x0 + (1 - abp).sqrt() * e2
        else:
            mean, _, logvar = gd.q_posterior(x0, x, t)
            x = mean if i == 0 else mean + (0.5 * logvar).exp() * torch.randn(shape, device=dev, generator=g)
    if final_clip:
        x = x.clamp(-1., 1.)
    return model.get_logits(x).argmax(-1).cpu().to(torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/diff_base_s0", "runs/diff_ce0x_s0"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ddim_of", default=None, help="clamp for the DDIM variant (default: best by correct/alpha)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for run in a.runs:
        model, _ = train.load_model_from_run(run, "ema", dev)
        W = model.byte_embedder.embedding.weight
        print(f"== {run}: A={W.abs().max().item():.2f} R={W.norm(dim=-1).mean().item():.2f} clip_denoised={model.gaussian_diffusion.clip_denoised}")
        res = {}
        cfgs = [("ancestral/none/finalclip[-1,1]", "none", False, True)]
        cfgs += [(f"ancestral/{c}", c, False, False) for c in ("none", "abs", "norm", "both")]
        for name, c, d, fc in cfgs:
            ids = sample_variant(model, a.n, c, d, a.seed, final_clip=fc)
            res[name] = ids
        best = a.ddim_of or max(["abs", "norm", "both"], key=lambda c: (stats([bytes(r.tolist()) for r in res[f"ancestral/{c}"]])["correct"], stats([bytes(r.tolist()) for r in res[f"ancestral/{c}"]])["alpha"]))
        for c in ("none", best):
            res[f"ddim/{c}"] = sample_variant(model, a.n, c, True, a.seed)
        print(f"{'variant':<34}{'alpha':>7}{'prefix':>8}{'correct':>9}{'lines':>6}")
        for name, ids in res.items():
            s = stats([bytes(r.tolist()) for r in ids])
            print(f"{name:<34}{s['alpha']:>7.3f}{s['prefix']:>8.3f}{s['correct']:>9.3f}{s['lines']:>6}")
        for name, ids in res.items():
            for i in range(2):
                print(f"  {name} s{i}: {bytes(ids[i].tolist())[:90]!r}")


if __name__ == "__main__":
    main()
