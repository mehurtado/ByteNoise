"""RPW encoder-local-attention swap: (A) loss at update 0 of the warm-started model vs the original, same batches/timesteps/noise (EMA weights, eval mode);
(B) fp32 training-step speed and peak VRAM of both variants, run back to back (old, new, old, new) so GPU contention hits both equally."""
import sys, os, time, math, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import torch
from dataclasses import replace
import data, train as T
from config import get_preset
from main import ByteDiffusionLM, EMA

RUN = "runs/rpw_s0"
B, A, WARM, N = 32, 4, 4, 12


def rpw_cfg(attn):
    c = get_preset("rpw")
    dif = replace(c.diffusion, param="logits", self_cond=True, ce_weight=0.0, x0_rescale=True, t_sampling="warp")
    return replace(c, diffusion=dif, model=replace(c.model, rpw_enc_attn=attn), train=replace(c.train, precision="fp32", batch_size=B, grad_accum_steps=A))


def update0_loss():
    old, _ = T.load_model_from_run(RUN, "ema", "cuda")
    new = ByteDiffusionLM(rpw_cfg(True)).cuda(); ema = EMA(new, decay=0.999, warmup=True).cuda()
    T.warm_start(os.path.join(RUN, "checkpoint.pt"), new, ema)
    new = ema.model
    z = np.load(os.path.join(data.DATA_ROOT, "mix-clean-v1", "windows.npz")); val = data.load_split("mix-clean-v1", "val")
    mix = json.load(open(os.path.join(data.DATA_ROOT, "mix-clean-v1", "mix.json")))
    res = {}
    for s in mix["sources"]:
        pool = z[f"val_{s['name']}"]; pick = pool[np.linspace(0, len(pool) - 1, 64).astype(int)]
        X = torch.stack([val[p:p + 256].long() for p in pick]).cuda()
        row = {}
        for tag, m in (("original", old), ("warm_started_untrained", new)):
            m.eval(); ls = []
            with torch.no_grad():
                for i in range(0, 64, 32):
                    torch.manual_seed(1234 + i); l, _ = m(X[i:i + 32]); ls.append(float(l))
            row[tag] = sum(ls) / len(ls)
        res[s["name"]] = row
    return res


def bench(attn):
    c = rpw_cfg(attn)
    torch.manual_seed(0)
    model = ByteDiffusionLM(c).cuda(); ema = EMA(model, decay=0.999, warmup=True).cuda()
    opt = T.make_optimizer(model, c.train)
    ac_dev, ac_dtype, use_scaler = T.amp_settings("fp32", "cuda")
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    g = torch.Generator().manual_seed(1)
    batches = [torch.randint(0, 256, (B, 256), generator=g).cuda() for _ in range(A)]
    params = list(model.parameters()); fin = [True]

    def step():
        model.train(); opt.zero_grad(set_to_none=True)
        ls, _ = T.accumulate_group(lambda b: model(b), batches, scaler if use_scaler else None, ac_dev, ac_dtype)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        fin[0] &= math.isfinite(gn) and all(math.isfinite(l) for l in ls)
        opt.step(); ema.update(model)
    for _ in range(WARM): step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    for _ in range(N): step()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / N
    out = dict(ms=dt * 1000, peak_gb=torch.cuda.max_memory_allocated() / 2**30, finite=fin[0], params=sum(p.numel() for p in model.parameters()))
    del model, ema, opt; torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    print("nvidia-smi compute apps:", subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() or "none")
    print("update-0 loss (EMA, eval mode, same batches/t/noise, 64 val windows per source):")
    print(json.dumps(update0_loss(), indent=1))
    runs = {False: [], True: []}
    for attn in (False, True, False, True):
        runs[attn].append(bench(attn))
    for attn, r in runs.items():
        print("enc_attn" if attn else "original", dict(ms=[round(x["ms"], 1) for x in r], mean_ms=round(sum(x["ms"] for x in r) / len(r), 1),
                                                       peak_gb=round(max(x["peak_gb"] for x in r), 2), finite=all(x["finite"] for x in r), params=r[0]["params"]))
