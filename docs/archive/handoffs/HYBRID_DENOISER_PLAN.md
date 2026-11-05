# Hybrid diffusion denoiser: parallel branches + self-conditioning (plan)

Status: proposal, not implemented. Written 2026-10-17 after the natural-text diffusion runs (see EXPERIMENTS.md).

## Motivation (evidence)

- Compact U-Net on natural-v1: 45-min gate run word-hit 0.351; 104k-update run word-hit 0.465 (real val text 0.933). Plateau, while validation noise loss kept falling (0.0122 -> 0.0071) and recovery got worse. Noise loss is not a good model-selection signal; use word-hit and sample quality.
- Failure shape: letter statistics, spacing and short words are learned; longer words are garbled and nothing is coherent. x0 accuracy is only ~0.32 at t >= 750, i.e. generation starts in the regime where the model is weakest.
- Toy tasks (copy/recall/rules): no diffusion variant (U-Net, bidirectional recurrent, conditional) computed answers; cross-position consistency is missing.
- Bidirectional recurrent denoiser alone (birnn): 3x lower noise loss than the U-Net but worse samples, 5x VRAM (3.35 GB vs 0.7 GB, fp32 log-space scans).

## Design

### Fusion block

Each block normalises the hidden states h (positions x channels), runs k branches in parallel on the same input, concatenates, and mixes with a gated FFN:

    y_i = Branch_i(norm(h)),  i = 1..k
    h  <- h + W_out * GatedFFN(concat(y_1 .. y_k)),   GatedFFN(z) = (silu(W_a z) * (W_b z))

Timestep conditioning (FiLM or additive) as in the existing denoisers. Predict eps; keep the DDPM process, `ce_weight=0`, `x0_rescale=true`.

### Branches

1. **Multi-scale convolution (U-Net path).** Local detail at strides 1/2/4/8 (existing code). 
2. **Bidirectional recurrent branch.** Forward and backward minGRU-style scans, `h_t = (1-z_t) h_{t-1} + z_t c_t`, using `models_lm.scan_parallel` (existing `BiMixer`). Gives a compressed global summary. Watch VRAM (fp32 scans); try bf16 inputs with fp32 scan accumulation, or smaller width.
3. **Global attention branch.** Full-resolution byte-level self-attention with rotary positions, 2-4 layers, exact lookup of earlier/later positions. NOTE: the existing U-Net already contains attention at some resolutions; check main.py first and make this branch additive (full resolution), not a duplicate.
4. **Local n-gram branch.** Direct word-piece evidence. Hashed n-gram embeddings need clean bytes, but training inputs are noisy embeddings, so use either (a) a depthwise conv with kernel 3-5 over the embeddings (cheap approximation), or (b) hashes of the self-conditioning estimate (below). Start with (a).

### Self-conditioning (separate ablation, also part of the combined model)

Feed the previous clean-embedding estimate as an extra input:

    eps_theta = f(x_t, t, x0_prev)

Training: with probability 0.5 set x0_prev = 0; otherwise compute x0_prev from a first forward pass with gradients stopped (`x0_prev = (x_t - sqrt(1-abar_t) eps_theta0)/sqrt(abar_t)`, then the usual norm rescale). Sampling: feed each step the previous step's estimate. Cost: ~1.5x training time, no extra passes at sampling. Targets the t >= 750 regime where the input carries almost no signal. References: Analog Bits (Chen et al.), CDCD (Dieleman et al.).

## Implementation notes

- New `ModelConfig.denoiser = "hybrid"` (default stays "unet"; architecture_hash must keep old hashes stable, following the `birnn` / `x0_rescale` / `cond_prompt` precedent).
- New `DiffusionConfig.self_cond: bool = False`, excluded from the hash when False; extra input channels only when True.
- Reuse: `models_lm.RMSNorm`, `GatedFFN`, `scan_parallel`, `main.BiRNNDenoiser/BiMixer`, `main.GaussianDiffusion.p_sample` (x0_rescale step).
- Tests: shapes; branch outputs depend on t; bidirectionality; self_cond zero-input equals plain forward when the channel is zeroed; old configs/hashes unchanged; full-model loss/backward/sampler smoke test.

## Experiment plan (single seed, 45 min each, natural-v1, fp32, bs32, lr 3e-4)

Baseline reference (done): `runs/diff_nat_gate_s0` - word-hit 0.351, valid_utf8 0.81, recovery t=200/400/600/800 = 0.920/0.919/0.892/0.478, 24.7k updates in 2700 s.

| run | change | question |
|---|---|---|
| A | compact U-Net + self-conditioning only | does self-conditioning alone fix the t >= 750 weakness? |
| B | hybrid (conv + recurrent + attention + n-gram conv), no self-cond | does parallel branch fusion help? |
| C | hybrid + self-conditioning | combined |

Match wall-clock (2700 s) and report updates reached, params, ms/update, peak VRAM. Metrics: word-hit (experiments/gate_text.py, same 32 samples/seed), valid_utf8, x0 accuracy by noise bucket (eval.py), recovery (experiments/recovery.py --data natural-v1). Do not select on noise loss.

Decision rule: a run is promising if word-hit exceeds the gate baseline by a clear margin (>= 0.10 absolute, e.g. >= 0.45 at 45 min) at matched wall-clock. Only then commit to a multi-hour run (and then repeat with a second seed).

## Known open problems (independent of this plan)

- Stray control/high bytes in samples (valid UTF-8 0.81 -> 0.03 with longer training): suspected under-trained embeddings for rare bytes; options: tie decoding to printable/frequent bytes, add a small byte-prior regulariser, or sample-time byte masking. Not yet investigated.
- Single-seed everywhere; word-hit differences below ~0.05 should not be treated as real.
- Autoregressive recurrent baseline on the same data: 1.626 bpb in 15 min (runs/nat_rec_L8); it remains the strongest model we have.
