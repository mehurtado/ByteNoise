"""Precision benchmark of the actual rpw training step (self-conditioning, backward, clip, optimizer, EMA) on a disposable model.
Usage: bench_rpw.py [batch] [accum] [warmup] [timed]"""
import sys, os, time, math, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from dataclasses import replace
import train as T
from config import get_preset
from main import ByteDiffusionLM, EMA

B, A, W, N = [int(x) for x in (sys.argv[1:5] + [32, 4, 4, 12][len(sys.argv[1:5]):])]
res = {}
for prec in ("fp32", "bf16"):
    c = get_preset("rpw")
    dif = replace(c.diffusion, param="logits", self_cond=True, ce_weight=0.0, x0_rescale=True, t_sampling="warp")
    c = replace(c, diffusion=dif, train=replace(c.train, precision=prec, batch_size=B, grad_accum_steps=A))
    torch.manual_seed(0)
    model = ByteDiffusionLM(c).cuda(); ema = EMA(model, decay=0.999, warmup=True).cuda()
    opt = T.make_optimizer(model, c.train)
    ac_dev, ac_dtype, use_scaler = T.amp_settings(prec, "cuda")
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    g = torch.Generator().manual_seed(1)
    data = [torch.randint(0, 256, (B, 256), generator=g).cuda() for _ in range(A)]
    params = list(model.parameters()); finite = True; losses = []
    def step():
        global finite
        model.train(); opt.zero_grad(set_to_none=True)
        ls, _ = T.accumulate_group(lambda b: model(b), data, scaler if use_scaler else None, ac_dev, ac_dtype)
        scaler.unscale_(opt); gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        finite &= math.isfinite(gn) and all(math.isfinite(l) for l in ls)
        scaler.step(opt); scaler.update(); ema.update(model); losses.append(sum(ls) / len(ls))
    for _ in range(W): step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
    for _ in range(N): step()
    torch.cuda.synchronize(); dt = (time.perf_counter() - t0) / N
    res[prec] = dict(ms_per_update=round(dt * 1000, 1), peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), finite=bool(finite),
                     kB_per_s=round(B * A * 256 / dt / 1000, 1), first_last_loss=[round(losses[0], 3), round(losses[-1], 3)])
    del model, ema, opt; torch.cuda.empty_cache()
print(json.dumps(dict(batch=B, accum=A, results=res), indent=1))
