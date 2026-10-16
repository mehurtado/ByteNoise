# Experiment log

All runs: RTX 4060 Ti, bf16, seed-controlled, `python train.py train --preset <p> --seed <s>`; evaluation by `python eval.py --run runs/<r> --split val`. Negative results are kept.

## E1. Controlled tasks (tasks-v1, 2000 updates, ~65-90 s, 3 seeds)
Transformer `tf_task` (4L d256, RoPE) vs recurrent `rec_task` (4L d256, parallel-scan minGRU mixer, 3.47M params). Teacher-forced token accuracy on answer bytes, EMA weights, mean±sd over seeds 0,1,2:

| task/regime | Transformer | Recurrent |
|---|---|---|
| pattern/id | 0.990±0.006 | 0.990±0.001 |
| pattern/unseen_period | 0.181±0.115 | 0.733±0.012 |
| pattern/long | 0.848±0.106 | 0.420±0.003 |
| copy/id | 0.999±0.001 | 0.983±0.005 |
| copy/long | 0.058±0.041 | 0.339±0.027 |
| recall/id | 0.643±0.290 | 0.115±0.015 |
| recall/long | 0.372±0.168 | 0.090±0.004 |
| rule/id | 0.998±0.001 | 0.990±0.001 |
| rule/unseen_combo | 0.406±0.019 | 0.838±0.009 |
| rule/long | 0.294±0.029 | 0.490±0.037 |

Findings: both learn the in-distribution rules. Attention wins associative recall (chance 0.10; the Transformer's high seed variance suggests a phase transition not reached in 2000 updates by every seed) and long periodic patterns; the recurrent model generalises better to unseen periods, unseen rule combinations and longer copies. Single small scale; do not generalise to language quality.

## E2. Natural text (natural-v1, document-level split), equal 15-minute wall-clock budget, seed 0, val bits/byte (EMA)
| run | params | updates | bytes seen | peak VRAM | val bpb |
|---|---|---|---|---|---|
| Transformer d512 L8 (`tf_nat_b`) | 25.3M | 1419 | 46.5M | 4.2 GB | 1.637 |
| Transformer d384 L12 (`tf_nat_a`) | 21.3M | 1573 | 51.5M | 4.5 GB | 1.650 |
| Transformer d640 L10 (`tf_nat_c`) | 49.3M | 356 | 11.7M | 6.7 GB | 1.883 |
| Recurrent d512 L8 (`rec_nat_L8`) | 27.5M | 1318 | 43.2M | 3.6 GB | 1.626 |

Single seed; the 0.011 bpb gap between the best Transformer and the recurrent model is not established as real. The 49M model loses on this budget because it processes about 4x fewer bytes. These are autoregressive models, so bpb is not comparable to diffusion losses.

## Diffusion track (2026-10-16) - tasks-v1, compact U-Net vs bidirectional recurrent denoiser

All single-seed, short runs (bs32, lr 3e-4, fp32). Validity = scripts/sample_validity.py (32 EMA samples, task-grammar checkers; 100% on real val lines).

Diagnosis (runs/diff_base_s0, 1500 updates, ce_weight 0.1):
- Training CE (x0 reconstruction at all t) is spiky and dominated by t>=750 (CE ~58); val noise loss 0.726.
- Sampler is correct (reverse chain from real data noised to t<=400 recovers ~100%); free-running samples were garbage (0% valid).
- Cause of garbage: x0_hat = (x_t - sqrt(1-ab) eps)/sqrt(ab) blows up at high t (norm 20-37 vs real ~7.8); the final clamp to [-1,1] is also wrong (embedding absmax ~4.2). One x0 step at t=999 decodes plausibly (96% in-alphabet); the 1000-step chain collapsed.
- ce_weight=0 helped a little (recovery t=600/800: 0.614/0.028 -> 0.756/0.298). In-loop per-position rescale of x0_hat to the mean embedding norm (config `x0_rescale`, default off) was the best sampler fix; DDIM eta=0 variant no better (scripts/sample_variants.py).

| run | updates | alphabet | valid-prefix | fully-correct lines | recovery t=200/400/600/800/999 |
|---|---|---|---|---|---|
| ce0x (U-Net 14.4M, rescale in eval script) | 1500 | 0.99 | 0.48 | 0 | 1.0/0.986/0.756/0.298/0.02 |
| diff_ce0_long_s0 (U-Net) | 8000 | 0.999 | 0.757 | 2/449 | 0.937/0.927/0.844/0.463/0.032 |
| birnn_ce0_long_s0 (bidir recurrent 10.7M) | 8000 | 1.000 | 0.526 | 0/605 | 1.0/1.0/0.992/0.567/0.037 |

Cost: U-Net 931 s, 698 MB peak; birnn 1044 s, 3350 MB peak (fp32 log-space scans).

Verdict: both learn line format, neither computes answers (copy/recall/rules) in 8000 updates. The bidirectional recurrent denoiser is not better at free-running generation despite a 3x lower noise loss (mostly low-t buckets), and costs 5x VRAM, so shared refinement loops on it are not justified yet. Limitations: one seed, no tuning, birnn not param-matched, no positional signal. Not run: natural-v1 diffusion, correct reduced-step DDIM for the speed goal (eta=0 variant exists in scripts/sample_variants.py but gave no quality gain).
