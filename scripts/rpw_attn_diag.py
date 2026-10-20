"""Read/write bridge routing diagnostic for an rpw run (EMA weights, no training, no gradients).
Feeds fixed windows (real val windows of mix-clean-v1 and the run's own generated samples) through the training forward at fixed noise levels
and records the attention probabilities of the read bridge (64 summaries query 256 bytes) and write bridge (bytes query the summaries).
Per head: normalised entropy (1 = uniform over keys), effective number of keys, locality mass. Write also: how evenly the summaries are used.
Usage: rpw_attn_diag.py --run runs/rpw_s0 [--samples runs/rpw_s0/final_eval/mix_samples.json] [--out DIR]"""
import argparse, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import torch
import torch.nn.functional as F
import data, train
from main import _RPWAttn
from models_lm import apply_rotary

REC = {}


def recording_forward(self, xq, xkv, rq, rk):
    b, nq, _ = xq.shape; nk = xkv.size(1)
    q = self.q(xq).view(b, nq, self.h, 32).transpose(1, 2)
    k = self.k(xkv).view(b, nk, self.h, 32).transpose(1, 2)
    v = self.v(xkv).view(b, nk, self.h, 32).transpose(1, 2)
    q, k = apply_rotary(q, *rq), apply_rotary(k, *rk)
    if getattr(self, "name", None):
        REC.setdefault(self.name, []).append(torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(32), dim=-1).float().cpu())
    y = F.scaled_dot_product_attention(q, k, v)
    return self.o(y.transpose(1, 2).reshape(b, nq, -1))


def stats(P, kind):
    """P: (B, heads, nq, nk) probabilities averaged over the examples seen."""
    B, H, nq, nk = P.shape
    ent = -(P * P.clamp_min(1e-12).log()).sum(-1)               # (B,H,nq)
    out = {"entropy_norm": (ent.mean((0, 2)) / math.log(nk)).tolist(), "eff_keys": ent.exp().mean((0, 2)).tolist()}
    if kind == "write":   # query byte i, key summary j covering bytes 4j..4j+3
        own = (torch.arange(nq) // 4)[:, None] == torch.arange(nk)[None, :]
        near = ((torch.arange(nq) // 4)[:, None] - torch.arange(nk)[None, :]).abs() <= 2
        use = P.mean(2)                                         # (B,H,nk): share of queries' mass landing on each summary
        uent = -(use * use.clamp_min(1e-12).log()).sum(-1) / math.log(nk)
        out.update(own_summary_mass=(P * own).sum(-1).mean((0, 2)).tolist(), within2_summaries_mass=(P * near).sum(-1).mean((0, 2)).tolist(),
                   summary_usage_entropy_norm=uent.mean(0).tolist(), max_summary_share=use.max(-1).values.mean(0).tolist())
    else:                 # query summary j, key byte i
        c = 4 * torch.arange(nq)[:, None] + 1.5
        d = (torch.arange(nk)[None, :] - c).abs()
        out.update(own_span_mass=(P * (d <= 1.5)).sum(-1).mean((0, 2)).tolist(), within16_bytes_mass=(P * (d <= 16)).sum(-1).mean((0, 2)).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--samples", default=None)
    ap.add_argument("--mix", default="mix-clean-v1")
    ap.add_argument("--ts", default="200,600,900")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model, cfg = train.load_model_from_run(a.run, "ema", "cuda"); model.eval()
    d = model.denoise_unet
    d.read.name, d.write.name = "read", "write"
    _RPWAttn.forward = recording_forward
    L = train.window_len(cfg)
    z = np.load(os.path.join(data.DATA_ROOT, a.mix, "windows.npz")); val = data.load_split(a.mix, "val")
    mix = json.load(open(os.path.join(data.DATA_ROOT, a.mix, "mix.json")))
    sets = {}
    real = []
    for s in mix["sources"]:
        pool = z[f"val_{s['name']}"]; pick = pool[np.linspace(0, len(pool) - 1, 8).astype(int)]
        real += [val[p:p + L].long() for p in pick]
    sets["real_val_windows"] = torch.stack(real)
    if a.samples:
        S = json.load(open(a.samples))["s1.0"]
        sets["generated_samples"] = torch.stack([torch.frombuffer(bytearray(bytes.fromhex(h)[:L]), dtype=torch.uint8).long() for h in S])
    res = {}
    for name, X in sets.items():
        for t in [int(x) for x in a.ts.split(",")]:
            REC.clear(); g = torch.Generator().manual_seed(0)
            with torch.no_grad():
                for i in range(0, len(X), 16):
                    x = X[i:i + 16].cuda()
                    tt = torch.full((x.size(0),), t, dtype=torch.long, device="cuda")
                    noise = torch.randn(x.size(0), x.size(1), cfg.model.byte_embed_dim, generator=g).cuda()
                    model(x, t=tt, noise=noise)
            for kind in ("read", "write"):
                P = torch.cat(REC[kind][0::2] if False else REC[kind])   # (N, heads, nq, nk); self-conditioning runs two passes per call
                res[f"{name}/t{t}/{kind}"] = stats(P, kind)
    out = a.out or os.path.join(a.run, "attn_diag")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "attn_diag.json"), "w"), indent=1)
    f = lambda v: "/".join(f"{x:.2f}" for x in v)
    for k, r in res.items():
        print(k, {m: f(v) for m, v in r.items()})


if __name__ == "__main__":
    main()
