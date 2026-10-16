import statistics
import time

import psutil
import torch

import sys

from config import get_preset
from train import build_model, window_len

PRESET = sys.argv[1] if len(sys.argv) > 1 else "compact"
CFG = get_preset(PRESET)
TRAIN_BATCH_SIZE = CFG.train.batch_size
TRAIN_SEQ_LEN = window_len(CFG)


def run(dtype_name, repeats=5, steps=10):
    dev = "cuda"
    torch.manual_seed(0)
    model = build_model(CFG).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    use_amp = dtype_name != "fp32"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype_name)
    scaler = torch.amp.GradScaler(enabled=(dtype_name == "fp16"))
    x = torch.randint(0, 256, (TRAIN_BATCH_SIZE, TRAIN_SEQ_LEN), device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            loss, _ = model(x)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        return loss.detach(), gn.detach()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times, finite = [], True
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            loss, gn = step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / steps)
        finite &= bool(torch.isfinite(loss) and torch.isfinite(gn))
    ms = [t * 1000 for t in times]
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"{dtype_name:5s} step {statistics.median(ms):7.1f} ms (min {min(ms):.1f}, max {max(ms):.1f}) "
          f"{TRAIN_BATCH_SIZE / statistics.median(times):6.1f} seq/s  peak VRAM {peak:7.0f} MiB  finite={finite}")
    del model, opt
    torch.cuda.empty_cache()
    return finite


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available"
    free, total = torch.cuda.mem_get_info()
    print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}  VRAM free {free / 2**20:.0f}/{total / 2**20:.0f} MiB  "
          f"RAM used {psutil.virtual_memory().used / 2**30:.1f}/{psutil.virtual_memory().total / 2**30:.1f} GiB")
    ok = all(run(d) for d in ("fp32", "bf16", "fp16"))
    print("process RSS %.2f GiB" % (psutil.Process().memory_info().rss / 2**30))
    raise SystemExit(0 if ok else 1)
