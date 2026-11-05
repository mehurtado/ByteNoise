"""Reverse-chain byte recovery: noise real val windows to t, reverse-sample with p_sample, compare argmax bytes."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import data
import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="tasks-v1")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ts", type=int, nargs="+", default=[200, 400, 600, 800, 999])
    ap.add_argument("--which", default="ema")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = train.load_model_from_run(a.run, a.which, dev)
    gd = model.gaussian_diffusion
    gd.clip_denoised = False
    tn = model.mean_embed_norm()
    table = model.embedding_table().detach()
    val = data.load_split(a.data, "val")
    g = torch.Generator().manual_seed(a.seed)
    starts = torch.randint(0, len(val) - a.len, (a.n,), generator=g)
    x = torch.stack([val[s:s + a.len] for s in starts]).long().to(dev)
    torch.manual_seed(a.seed)
    with torch.no_grad():
        x0 = model.byte_embedder(x)
        for t0 in a.ts:
            xt = gd.q_sample(x0, torch.full((a.n,), t0, device=dev, dtype=torch.long))
            x0p = None
            for i in reversed(range(t0 + 1)):
                tt = torch.full((a.n,), i, device=dev, dtype=torch.long)
                if gd.param == 'logits':
                    xt, x0p, last_lg = gd.p_sample(xt, tt, i, x0_prev=x0p if gd.self_cond else None, table=table, return_logits=True)
                elif gd.self_cond:
                    xt, x0p = gd.p_sample(xt, tt, i, tn, x0_prev=x0p, return_x0=True)
                else:
                    xt = gd.p_sample(xt, tt, i, tn)
            acc = ((last_lg if gd.param == 'logits' else model.get_logits(xt)).argmax(-1) == x).float().mean().item()
            print(f"t={t0}: recovery={acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
