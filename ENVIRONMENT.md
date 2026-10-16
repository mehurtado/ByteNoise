# GPU environment

Hardware: RTX 4060 Ti (8 GB; ~7 GB free with desktop running), i9-12900K, 32 GB RAM. Windows 11, driver 591.86, Python 3.13.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt     # exact versions: requirements-lock.txt
.venv\Scripts\python gpu_smoke.py
```

## Smoke test results (2026-10-15, torch 2.11.0+cu128, compact ByteNoise, batch 16 x 256 bytes)

| Precision | ms/step (median of 5x10) | seq/s | Peak VRAM | Finite loss/grad |
|---|---|---|---|---|
| fp32 | 87.2 | 183 | 414 MiB | yes |
| bf16 | 110.5 | 145 | 404 MiB | yes |
| fp16 (GradScaler) | 112.4 | 142 | 404 MiB | yes |

Process RSS 2.05 GiB. Timings use warm-up and CUDA synchronization. For this small model mixed precision is slower than fp32 (overhead-bound, not compute-bound); revisit with larger models. System RAM was already 22.6 of 31.8 GiB used by other programs, so budget dataloader memory accordingly.
