"""Matched train-vs-val CE for a logits run: same procedure as train.quick_eval (uniform t, fixed noise, EMA weights),
evaluated on N evenly spaced windows of the train split and on the val split."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import data, train


@torch.no_grad()
def ce(model, cfg, tokens, n=1024, bs=32, seed=0):
    win = train.Windows(tokens, train.window_len(cfg))
    idx_all = torch.linspace(0, len(win) - 1, n).long()
    g = torch.Generator().manual_seed(seed); T = cfg.diffusion.num_timesteps; s = 0.0
    for i in range(0, n, bs):
        x = win.get(idx_all[i:i + bs]).to("cuda")
        t = torch.randint(0, T, (x.size(0),), generator=g).to("cuda")
        noise = torch.randn(x.size(0), x.size(1), cfg.model.byte_embed_dim, generator=g).to("cuda")
        _, st = model(x, t=t, noise=noise); s += float(st["ce_loss"]) * x.size(0)
    return s / n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--data", default="natural-v2")
    a = ap.parse_args()
    model, cfg = train.load_model_from_run(a.run, "ema", "cuda"); model.eval()
    for r in (0, 1):
        print(f"seed{r} train CE {ce(model, cfg, data.load_split(a.data,'train'), seed=r):.4f}  val CE {ce(model, cfg, data.load_split(a.data,'val'), seed=r):.4f}", flush=True)
