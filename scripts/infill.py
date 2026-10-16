"""RePaint-style prompt-conditioned generation with an unconditional byte-diffusion model on tasks-v1.

Window of seq_len bytes: the example's prompt (bytes before the first answer byte) sits at the window start and is
clamped; answer, newline and the rest of the window are unknown. After each reverse step the known positions are
overwritten with q_sample(clean_embed, t-1) (exact clean embedding at t=0). --resample r repeats denoise at each
step RePaint-style (jump-back of length 1: re-noise x_{t-1} -> x_t, denoise again, r passes total).
Scores per task/regime: answer-byte accuracy (mask==1 positions), exact match (all answer bytes right), and strict
exact match (additionally the next byte is a newline).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import data
import train


@torch.no_grad()
def infill(model, prompts, resample=1, seed=0):
    gd = model.gaussian_diffusion
    W = model.byte_embedder.embedding.weight
    dev, L, E = W.device, gd.seq_len, W.shape[1]
    R = model.mean_embed_norm()
    n = len(prompts)
    g = torch.Generator(device=dev).manual_seed(seed)
    ids = torch.zeros(n, L, dtype=torch.long, device=dev)
    known = torch.zeros(n, L, dtype=torch.bool, device=dev)
    for i, p in enumerate(prompts):
        ids[i, :len(p)] = torch.tensor(list(p), device=dev)
        known[i, :len(p)] = True
    clean = model.byte_embedder(ids)
    k = known.unsqueeze(-1)

    def rn():
        return torch.randn(n, L, E, device=dev, generator=g)

    def step(x, i):
        t = torch.full((n,), i, device=dev, dtype=torch.long)
        x0 = gd.predict_start_from_noise(x, t, gd.denoise_model(x, t))
        x0 = x0 * (R / x0.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        mean, _, logvar = gd.q_posterior(x0, x, t)
        xp = mean if i == 0 else mean + (0.5 * logvar).exp() * rn()
        if i == 0:
            return torch.where(k, clean, xp)
        tp = torch.full((n,), i - 1, device=dev, dtype=torch.long)
        return torch.where(k, gd.q_sample(clean, tp, rn()), xp)

    x = rn()
    for i in reversed(range(gd.num_timesteps)):
        for u in range(resample):
            xp = step(x, i)
            if u < resample - 1 and i > 0:  # jump back x_{t-1} -> x_t
                b = gd.betas[i]
                x = (1 - b).sqrt() * xp + b.sqrt() * rn()
            else:
                x = xp
    return model.get_logits(x).argmax(-1).cpu().to(torch.uint8)


def score(out, examples):
    nb = nc = ne = ns = 0
    for row, (b, m) in zip(out, examples):
        g = bytes(row.tolist())
        pos = [i for i, v in enumerate(m) if v == "1" or v == 1]
        ok = [g[i] == b[i] for i in pos]
        nb += sum(ok); nc += len(ok); ne += all(ok)
        ns += bool(all(ok) and g[len(b) - 1] == 10)  # example line ends with the newline right after the answer
    n = len(examples)
    return nb / nc, ne / n, ns / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["runs/diff_ce0_24k_s0", "runs/diff_ce0_long_s0"])
    ap.add_argument("--data", default="tasks-v1")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--which", default="ema")
    ap.add_argument("--resample", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=2, help="print this many decoded examples per cell")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ev = data.load_eval_tasks(a.data)
    for run in a.runs:
        model, _ = train.load_model_from_run(run, a.which, dev)
        model.eval()
        for r in a.resample:
            print(f"== {run} resample={r}")
            print(f"{'task/regime':<24}{'n':>4}{'byte_acc':>10}{'exact':>8}{'strict':>8}")
            for task, regs in ev.items():
                for reg, items in regs.items():
                    ex = items[:a.n]
                    prompts = [b[:m.index("1") if isinstance(m, str) else list(m).index(1)] for b, m in ex]
                    out = infill(model, prompts, r, a.seed)
                    ba, em, st = score(out, ex)
                    print(f"{task + '/' + reg:<24}{len(ex):>4}{ba:>10.3f}{em:>8.3f}{st:>8.3f}", flush=True)
                    for i in range(min(a.samples, len(ex))):
                        print(f"    want {ex[i][0]!r}\n    got  {bytes(out[i].tolist())[:len(ex[i][0]) + 4]!r}")


if __name__ == "__main__":
    main()
