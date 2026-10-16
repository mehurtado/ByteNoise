import argparse
import time

import torch

from config import get_preset
from main import ByteDiffusionLM


def benchmark(preset="compact", iters=10, device="auto"):
    device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    cfg = get_preset(preset)
    model = ByteDiffusionLM(cfg).to(device)
    unet = model.denoise_unet
    params = sum(p.numel() for p in model.parameters())
    bs, n, c = cfg.train.batch_size, cfg.model.seq_len, cfg.model.byte_embed_dim
    print(f"preset={preset} device={device} params={params:,} (unet {sum(p.numel() for p in unet.parameters()):,}) "
          f"heads={cfg.model.attn_heads} model_dim={cfg.model.model_dim}")

    x = torch.randn(bs, n, c, device=device)
    t = torch.randint(0, cfg.diffusion.num_timesteps, (bs,), device=device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def step():
        unet.zero_grad(set_to_none=True)
        unet(x, t).mean().backward()

    step()  # warm-up, excluded from timing
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"{iters} forward/backward passes (UNet only, after warm-up): {elapsed:.4f}s ({elapsed / iters * 1000:.1f} ms/pass)")
    if device == "cuda":
        print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="compact")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    benchmark(a.preset, a.iters, a.device)
