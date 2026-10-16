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

### 24k-update U-Net (runs/diff_ce0_24k_s0, same recipe, 2755 s, val noise loss 0.0154)
alphabet 0.998, valid-prefix 0.924, fully-correct lines 0.009 (4/458; pattern 1/74, copy 1/89, recall 0/114, rule 2/175). 3x more training raised format validity (0.757 -> 0.924) but not answer correctness (0.004 -> 0.009). Lines look like the tasks (`c:rcrlr|dcrrd`, `v:sort:vai=tvn`) but payloads are inconsistent with their own prefixes, i.e. the model learns local statistics, not the cross-position dependencies.

## Final summary (diffusion direction)
- Fixed: training/eval reporting, sampler collapse (x0 blow-up + wrong final clip; `x0_rescale`), CE term demoted (ce_weight 0 was better at high noise). Reproducible CLI, data (tasks-v1, natural-v1), validity/recovery scripts, 64+ tests.
- Learned: bytes-in-alphabet and line format; NOT answers to copy/recall/rules, with U-Net (1.5k/8k/24k updates) or bidirectional recurrent denoiser (8k). Autoregressive models on the same tasks do reach ~0.98+ in-distribution (E1), so the gap is specific to this diffusion formulation at this scale.
- Not done / recommended next: conditional (prompt-infilling) generation rather than unconditional line generation; a cross-position mechanism with global receptive field (attention in the denoiser) before more training; natural-v1 diffusion runs; DDIM only after quality exists. Single seed throughout.

### Prompt-conditioned (inpainting) test, no retraining (scripts/infill.py, commit 0951b76)
RePaint-style clamping of the prompt bytes during reverse sampling on diff_ce0_24k_s0 and diff_ce0_long_s0 (n=100/regime, r=1; n=30, r=3 jump-back): answer-byte accuracy ~0.02-0.06 and exact match ~0 for every task/regime, i.e. chance (letters ~0.038, recall digits 0.10). Samples keep the prompt then write line-shaped noise. Conclusion: models trained with the unconditional objective do not learn prompt->answer dependencies; clamping alone cannot supply them. AR models reach ~0.98+ in-distribution (teacher-forced metric, different from generated accuracy).
Untested: training WITH conditioning (noise only the answer span, clean prompt as input); natural-v1 diffusion.

### Conditional training (diffusion.cond_prompt, commit 3ff0e5e; runs/diff_cond_s0, 8000 updates)
Random clean prefix (0-128 bytes) per window, loss only on noised positions, 1-channel clean-mask input. Answer accuracy at chance on every task/regime (byte acc 0.03-0.05, exact ~0; recall 0.03-0.04 < 0.10 digit chance), same as unconditional models. Samples keep format, wrong answers even for copy/recall. Caveats: one seed, random-prefix (not answer-aligned) masking, small U-Net. Untried: line-aware answer-span masking, longer training, attention-heavy denoiser, natural-v1.
Overall conclusion: on these algorithmic tasks the byte-diffusion denoisers (U-Net and bidirectional recurrent, uncond. and cond.) learn format/local statistics but not answers within ~1 h of training on this GPU, whereas small AR models solve them in ~1.5 min. Recommendation for the research direction: do not judge diffusion on these tasks; evaluate on natural-v1 where local statistics matter, or switch to line-aware masked/discrete diffusion if reasoning-like tasks are required.

### Natural-text gate: compact U-Net diffusion on natural-v1 (runs/diff_nat_gate_s0, commit 7dd7004)
ce_weight 0, x0_rescale, bs32, lr 3e-4, fp32, 2700 s = 24,675 updates (~4.6 epochs), 9.1 upd/s, 698 MB VRAM, 14.4M params. Final EMA val noise loss 0.0122 and still falling.
Samples (32 x 256 B): word-hit 0.351 vs 0.933 on real val text; valid UTF-8 0.81 vs 0.97; English-like letter frequencies, spacing and short words, garbled longer words, no grammar. x0 accuracy by noise bucket 0.90/0.90/0.84/0.32 (t>=750 is the bottleneck); reverse-chain recovery t=200/400/600/800: 0.920/0.919/0.892/0.478. Latency 28 s per 1000-step batch (0.87 s/sample at batch 32).
Verdict: gate FAIL at 45 min, but under-trained (loss still falling), single seed. Next test would be a longer run and/or high-t-weighted loss, not a verdict on the method.
