# ByteNoise v0.1  
*A Byte-Level Diffusion Model for Byte Sequence Generation*

**ByteNoise** is an ongoing research project exploring the viability of denoising diffusion probabilistic models (DDPMs) applied directly to byte sequences. The model operates without tokenization, treating text as raw byte data and attempting to reconstruct coherent sequences from random noise.

This repository contains:
- A custom U-Net architecture for 1D byte diffusion
- Training loop and configuration for autoregressive sampling
- Roadmap for improvements and failure analysis

> **Warning:** This model is still unstable. Current outputs are mostly noise. Training is active and this repository will be updated as results evolve.


## Running (Batch 1: presets, run directories, corrected training)

Architecture and hyperparameters live in `config.py` as explicit presets — never edit globals to switch models.

| Preset | Params | Notes |
|---|---|---|
| `original` | 57,603,328 | embed 128, dim 192, 6 heads (matches `byte_diffusion_model.pth`) |
| `compact` | 14,430,848 | embed 64, dim 96, 4 heads (width **and** head count differ from `original`) |
| `debug` | 386,128 | tiny CPU model for correctness tests |

```
python train.py train --preset compact --max-updates 200          # fresh run -> runs/compact-<timestamp>/
python train.py train --resume runs/<run>                         # continue with the run's stored config
python train.py generate --run runs/<run> --which both --seeds 0 1
python train.py generate --checkpoint byte_diffusion_model.pth --preset original   # legacy .pth, read-only
python benchmark.py --preset compact
python -m pytest tests -q
```

Each run directory holds `config.json`, `meta.json` (seed, dataset hash, git revision, device, param count), `checkpoint.pt`, `metrics.jsonl` (per-update loss components, per-noise-bucket loss, grad norm, clipping, embedding norms) and `samples/`. Fresh runs refuse non-empty directories; resume rejects changed data or shape-incompatible checkpoints before training. Training length is `max_updates` optimizer updates.

Known open issue (Batch 2): the byte cross-entropy term is computed from the x0 estimate at every noise level and can dominate early (observed CE ~4000, grad norm ~500 on the debug preset).
