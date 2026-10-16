"""Answer accuracy for cond_prompt diffusion runs on tasks-v1: prompt bytes (before the first mask==1 byte) are exact
clean positions with the mask channel set; the rest of the window is sampled (x0-rescale sampler). Scores as scripts/infill.py."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import data
import train
from scripts.infill import score


@torch.no_grad()
def cond_sample(model, prompts, seed=0):
    gd = model.gaussian_diffusion
    dev, L = model.byte_embedder.embedding.weight.device, gd.seq_len
    E = model.byte_embedder.embedding.weight.shape[1]
    R, n = model.mean_embed_norm(), len(prompts)
    g = torch.Generator(device=dev).manual_seed(seed)
    ids = torch.zeros(n, L, dtype=torch.long, device=dev)
    known = torch.zeros(n, L, dtype=torch.bool, device=dev)
    for i, p in enumerate(prompts):
        ids[i, :len(p)] = torch.tensor(list(p), device=dev)
        known[i, :len(p)] = True
    clean, k = model.byte_embedder(ids), known.unsqueeze(-1)
    x = torch.where(k, clean, torch.randn(n, L, E, device=dev, generator=g))
    for i in reversed(range(gd.num_timesteps)):
        t = torch.full((n,), i, device=dev, dtype=torch.long)
        x0 = gd.predict_start_from_noise(x, t, gd.denoise(x, t, known))
        x0 = x0 * (R / x0.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        mean, _, logvar = gd.q_posterior(x0, x, t)
        x = mean if i == 0 else mean + (0.5 * logvar).exp() * torch.randn(n, L, E, device=dev, generator=g)
        x = torch.where(k, clean, x)
    return model.get_logits(x).argmax(-1).cpu().to(torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/diff_cond_s0")
    ap.add_argument("--data", default="tasks-v1")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--which", default="ema")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=2)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = train.load_model_from_run(a.run, a.which, dev)
    model.eval()
    print(f"== {a.run} ({a.which})")
    print(f"{'task/regime':<24}{'n':>4}{'byte_acc':>10}{'exact':>8}{'strict':>8}")
    for task, regs in data.load_eval_tasks(a.data).items():
        for reg, items in regs.items():
            ex = items[:a.n]
            prompts = [b[:m.index("1") if isinstance(m, str) else list(m).index(1)] for b, m in ex]
            out = cond_sample(model, prompts, a.seed)
            ba, em, st = score(out, ex)
            print(f"{task + '/' + reg:<24}{len(ex):>4}{ba:>10.3f}{em:>8.3f}{st:>8.3f}", flush=True)
            for i in range(min(a.samples, len(ex))):
                print(f"    want {ex[i][0]!r}\n    got  {bytes(out[i].tolist())[:len(ex[i][0]) + 4]!r}")


if __name__ == "__main__":
    main()
