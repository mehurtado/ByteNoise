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

### Natural-text long run: compact U-Net diffusion, 104,489 updates (runs/diff_nat_long_s0, 12,000 s, ~4.4x the gate run)
Same recipe as the gate run (ce_weight 0, x0_rescale, bs32, lr 3e-4, fp32). The PC crashed at ~82k updates; resumed from the 80k checkpoint (elapsed time restored), so metrics.jsonl has duplicate rows for 80k-82k (pre-crash log: runs/diff_nat_long_s0.precrash.log). Final val noise loss 0.0071 (gate run 0.0122).
| | gate (24.7k upd) | 80k upd | final (104k upd) | real val |
|---|---|---|---|---|
| word-hit | 0.351 | 0.478 | 0.465 | 0.933 |
| valid UTF-8 | 0.81 | 0.03 | 0.03 | 0.97 |
| printable | 0.995 | 0.983 | 0.984 | 0.975 |
| recovery t=200/400/600/800 | 0.920/0.919/0.892/0.478 | - | 0.853/0.853/0.826/0.432 (t=999: 0.073) | - |
Latency 0.92 s/sample at batch 32. Samples: recognisable fragments (that, and, her, had, the), garbled longer words, no grammar, stray control/high bytes (\x07 \x08 \xe8 \xc7). Verdict: 4x more training raised word-hit 0.35 -> ~0.47 then plateaued; noise loss kept falling (0.0122 -> 0.0071) while recovery and valid-UTF-8 got WORSE, so noise-prediction loss is not tracking sample quality. Single seed, no loss reweighting tried.

### Self-conditioning, run A: compact U-Net + self_cond (runs/diff_nat_selfcond_s0, commit 67702c9)
Same recipe as the gate run plus diffusion.self_cond=true (x0_prev channels, p=0.5 zeroed, detached first pass; sampler feeds previous step's rescaled x0 estimate). 2700 s = 31,021 updates (87 ms/update, 11.5 upd/s; gate 24,675 updates, 109 ms/update in its logged run, so no measurable slowdown was observed in wall-clock; timing may be affected by concurrent load), 708 MB VRAM, 14.47M params. Final EMA val noise loss 0.0104 (gate 0.0122; at matched 24k updates 0.0111 vs 0.0122 - eval uses two-pass inference).
| | gate | self-cond | real val |
|---|---|---|---|
| word-hit (32 samples) | 0.351 | 0.383 | 0.933 |
| valid UTF-8 | 0.81 | 0.375 | 0.97 |
| x0 acc buckets (eval.py, two-pass) | 0.90/0.90/0.84/0.32 | 0.902/0.900/0.846/0.323 | - |
| recovery t=200/400/600/800 | 0.920/0.919/0.892/0.478 | 0.921/0.920/0.890/0.501 (t=999: 0.075) | - |
Latency 0.42 s/sample at batch 32. Verdict: NOT promising. Word-hit 0.383 is +0.03 over the gate (below the 0.45 / +0.10 bar and within single-seed noise); the t>=750 x0 accuracy is unchanged at 0.32 and valid UTF-8 dropped; only recovery at t=800 improved slightly (0.478 -> 0.501). Self-conditioning alone does not fix the high-noise weakness.

### Hybrid denoiser, runs B and C (commits 958c2ff, a6ab3e7; single seed, 2700 s each, natural-v1)
Hybrid denoiser (config `hybrid` preset): d=256, 7 blocks, each = RMSNorm+FiLM -> parallel branches (dilated depthwise conv 1/2/4/8, bidirectional minGRU scan width 128, full-resolution non-causal rotary attention 8 heads, depthwise k=5 n-gram conv) -> concat -> linear -> GatedFFN -> residual. 14.00M params (14.02M with self-cond), U-Net compact 14.4M. Recipe as the gate run (ce_weight 0 for B, x0_rescale, bs32, lr 3e-4) EXCEPT precision bf16 (autocast; measured 252 vs 425 ms/update fp32 under GPU contention, 2.4 vs 3.6 GB), so the matched resource is wall-clock, not updates. B: no self-cond. C: self_cond + signal-weighted CE (ce_weight 0.05, ce_schedule=alpha_bar: per-example weight 0.05*alpha_bar_t applied to each example's mean CE before the batch mean; alpha_bar at t=0/250/500/750/999 = 0.9999/0.846/0.492/0.143/0.0). Note C changes two things at once vs B (self-cond and weighted CE) and the weight is only ~0 for t >~ 900.
| | gate (U-Net) | A (U-Net+selfcond) | B hybrid | C hybrid+selfcond+wCE | real val |
|---|---|---|---|---|---|
| updates in 2700 s | 24,675 | 31,021 | 19,873 | 17,593 | |
| peak VRAM (train) | 0.70 GB | - | 2.34 GB | 2.34 GB | |
| word-hit (32, seed 0, ema) | 0.351 | 0.383 | 0.356 | 0.332 | 0.933 |
| valid UTF-8 (gate_text) | 0.81 | 0.375 | 0.9375 | 0.9688 | 0.969 |
| distinct_4gram / repeated_8gram | - | - | 0.955 / 0 | 0.969 / 0 | 0.922 / 0.006 |
| eval.py valid UTF-8 (val) | - | - | 0.969 | 1.0 | |
| val noise loss | 0.0122 | - | 0.0063 | 0.0072 | |
| x0 acc buckets (diagnostic) | 0.90/0.90/0.84/0.32 | 0.90/0.90/0.85/0.32 | 1.00/1.00/0.98/0.44 | 1.00/1.00/0.98/0.44 | |
| recovery t=200/400/600/800 (diagnostic) | 0.920/0.919/0.892/0.478 | 0.921/0.920/0.890/0.501 | 1.000/1.000/0.996/0.667 | 1.000/1.000/0.996/0.644 | |
Latency 1.2-1.3 s/sample at batch 32 (U-Net 0.4-0.9). Samples (both): lowercase English-like letter and spacing statistics, many short real words ("the", "and", "he"), garbled longer words, no grammar, no stray control bytes; B and C produce near-identical text from the same noise seed. Verdict: NOT promising by the plan's rule (word-hit 0.356 / 0.332 vs bar 0.45; +0.005 / -0.02 over the gate, inside single-seed noise). The hybrid clearly improves denoising (noise loss halved, recovery up) and valid UTF-8 with C (0.97) is better than the gate/A, but readable-word rate did not move; word-hit is flat across all four 45-min runs (0.33-0.38). Single seed, bf16 vs fp32 and fewer updates are confounds; no attribution to individual components is possible (C bundles self-cond + weighted CE).

## Sampler sweep on existing natural-text checkpoints (scripts/sampler_sweep.py, no retraining)
32 samples, seed 0, EMA, batch 32. Own loop with generalized DDPM posterior (ab_t, ab_prev), x0_rescale and self-cond carry kept; s scales posterior noise. Control = model.sample (gate protocol; reproduces 0.351 / 0.332). Raw: runs/sampler_sweep_results.json.

| variant | gate U-Net word_hit | hybrid+selfcond word_hit (distinct_4gram) | s/sample (U-Net / hybrid) |
|---|---|---|---|
| control (model.sample) | 0.351 | 0.332 (0.969) | 0.41 / 1.23 |
| 1000 steps s=1.0 (own loop) | 0.330 | 0.325 | |
| s=0.8 | 0.365 | 0.406 (0.942) | |
| s=0.5 | 0.301 | 0.473 (0.889) | |
| s=0.25 | 0.021 (d4 0.53, r8 0.02) | 0.487 (0.730, r8 0.03) | |
| s=0.0 | 0.000 (collapsed) | 0.000 (collapsed) | |
| uniform 250 / 500, s=1 | 0.337 / 0.332 | 0.314 / 0.312 | 0.11,0.23 / 0.30,0.62 |
| late-heavy 250 / 500, s=1 | 0.326 / 0.333 | 0.316 / 0.324 | |
| uni250 s=0.8 | 0.385 | - | 0.11 |
| uni250 s=0.5 / 0.25 | 0.294 / - | 0.453 (0.896) / 0.483 (0.704, r8 0.075) | 0.31 |
| late500 s=0.5 / 0.25 | - | 0.463 / 0.481 | 0.61 |

Verdict: respacing alone does nothing to word_hit (but 250 steps is 4x faster at equal quality). Lowering noise helps only the hybrid+selfcond model (+0.07 at s=0.8, +0.14 at s=0.5, 0.45-0.49 for s<=0.5, valid UTF-8 -> 1.0), at the cost of diversity (distinct_4gram 0.97 -> 0.89 at s=0.5, repeats appear at 0.25); sample text is still non-words/garbled ("ene her ane fovesee as an is of alich"), so part of the gain is more common short words, not real language. U-Net gate: nothing meaningful (best 0.385 at uni250 s=0.8, +0.03, noise). Single seed.

## natural-v2 dataset (larger Gutenberg train set)

`python data.py prepare gutenberg-v2` builds `data/natural-v2` (use `train.py --data natural-v2`). Train = natural-v1 train (57 docs' train share) + 278 extra English Gutenberg books (prose; catalogue-filtered, hash-ordered candidates, 1 s politeness delay) forced into train only; val/test are byte copies of natural-v1's (sha256 match recorded in the manifest as `val_test_identical_to_base`, and checked by tests/test_data.py). Extra paragraphs duplicating any v1 train/val/test paragraph (or each other) were dropped (836 total). Sizes: train 160,286,597 B (v1: 43.7 MB), val 2,281,541 B, test 5,175,949 B. 65 candidates skipped (64 under 100k chars, 1 download error). Scores on val/test remain directly comparable with natural-v1 runs.

### CDCD-style logit head + fixed embedding norm + time warping (runs L1-L3, commit d0fd30f; single seed, natural-v1, 2700 s each)
New default-off options (hash-stable, old configs/checkpoints load unchanged): `diffusion.param=logits` (denoiser outputs 256 byte logits; x0_hat = softmax(logits) @ E, a convex combination of embeddings, E detached inside x0_hat; training loss = CE(logits, byte) at every t, no 1/sqrt(ab) factor; sampling = DDPM posterior with x0_hat, `x0_rescale` ignored, final decode = argmax of the last step's logits; self-cond feeds x0_hat), `model.embed_norm=R` (every embedding row rescaled to norm R, differentiable normalisation; R=8=sqrt(embed_dim) gives unit per-element RMS), `diffusion.t_sampling=warp` (EMA of per-sample training loss in 50 t-bins, rate 0.05, p(bin) = 0.8*normalised EMA + 0.2 uniform, uniform inside the bin, no importance correction; state in buffers `t_loss_ema/t_loss_seen`, checkpoint/resume tested), optional `diffusion.x0_mse_weight` (not used). In logits mode "noise_loss" is the MSE of the implied eps, bucket/val losses are CE in nats/byte, `val_ce_loss` is the training metric. Gate recipe otherwise (compact U-Net, ce_weight 0, bs32, lr 3e-4, fp32); `train.py --set model.embed_norm=8 diffusion.param=logits [diffusion.t_sampling=warp]`.
| | gate (eps) | L1 logits R=8 | L2 L1+warp | L3 logits R=16 | real val |
|---|---|---|---|---|---|
| updates in 2700 s (ms/update) | 24,675 (109) | 42,104 (64) | 45,533 (59) | 48,614 (56) | |
| peak VRAM | 0.70 GB | 0.71 GB | 0.71 GB | 0.71 GB | |
| word-hit (32, seed 0, ema) | 0.351 | 0.525 | 0.558 | 0.513 | 0.933 |
| valid UTF-8 (gate_text) | 0.81 | 0.969 | 1.000 | 0.969 | 0.969 |
| distinct_4gram / repeated_8gram | - | 0.941 / 0.0020 | 0.932 / 0.0025 | 0.938 / 0.0010 | 0.922 / 0.006 |
| val CE nats/byte (EMA, uniform t) | - | 0.396 | 0.396 | 0.201 (easier: 2x signal scale) | |
| val CE by t quartile | - | 0/0/0.018/1.566 | 0/0.001/0.031/1.553 | 0/0/0/0.805 | |
| x0 acc by t quartile | 0.90/0.90/0.84/0.32 | 1.00/1.00/0.995/0.565 | 1.00/1.00/0.991/0.567 | 1.00/1.00/1.00/0.776 | |
| recovery t=200/400/600/800/999 | 0.920/0.919/0.892/0.478/- | 1.00/1.00/1.00/0.844/0.062 | 1.00/1.00/0.998/0.842/0.067 | 1.00/1.00/1.00/0.999/0.064 | |
Latency 0.35-0.41 s/sample at batch 32. Samples (seed 0, first of 32): L1 `st Phe, that if nor beale, and of tho\nthey I wis no so that the Wit tive thathout\nof brost so his sidion wepart.` ; L2 `s, and is at in nor suave, and io tho\ndhet nou someas, that thed it live pentiourly, but that his aiviour` ; L3 `s, the it at tare a gporyigs vithersand pyom. Borthere seen them it tive to reqlime, alo\nthat has eiveoued on the ellme et he nay`. Text is still not English (invented words, no grammar) but has far more real short words and punctuation/quotes/UTF-8 punctuation are well-formed; L1 and L2 produce near-identical text from the same noise seed.
Verdict: the logit head is the first change to move word-hit past the plan bar (0.35 -> 0.51-0.56 at the same 45 min; the other variants stayed 0.33-0.38 and the 4x-longer eps run plateaued at ~0.47). Valid UTF-8 is also back to real-text level (0.97-1.0). Time warping (+0.03 word-hit, 1.0 valid UTF-8) and R=16 (-0.01) are inside single-seed noise (about +-0.03 at 32 samples); the L3 CE is not comparable because doubling the embedding radius is a noise-schedule shift. Training CE at t<750 is ~0 for all three, i.e. all remaining error is in the t>=750 quartile (x0 acc 0.57; 0.78 with R=16) and the free-running sample is the pure-noise end of the chain. Caveats: single seed, 32 samples, word-hit rewards frequent short words; also ~1.7x more updates than the gate in equal wall-clock (logit head is faster per update), so part of the gain may be update count (the eps 4x-longer run reached only 0.47 at 104k updates).

### Sampler sweep for param=logits checkpoints (scripts/sampler_sweep_logits.py, no retraining)
32 samples, seed 0, EMA, batch 32, natural-v1 word list, global-RNG protocol of gate_text (reseed, warmup `sample(1)`, then sample). Own loop mirrors the logits `p_sample` (x0_hat = softmax(logits)@E, self-cond carries x0_hat, final decode = argmax of last-step logits, x0_rescale ignored); s scales the posterior std; respacing uses the generalized posterior between non-adjacent steps. Control check: the own loop at s=1.0 / 1000 steps is byte-identical to `model.sample` for both runs (warp: word_hit 0.5579, valid_utf8 1.0, as in gate_text). Raw: runs/sampler_sweep_logits_results.json (+ _extra.json for late250 at s=0.8). Run while the GPU was shared with the long training run. Note: scripts/sampler_sweep.py reads metric keys that do not exist (`valid_utf8`, `printable`, ...) so its utf8/print/d4/r8 columns printed nan; this script uses the real keys (valid_utf8_rate etc.).

| variant (steps, s) | warp wh / utf8 / print / d4 / r8 | logits wh / utf8 / print / d4 / r8 | s/sample |
|---|---|---|---|
| control 1000, s=1.0 | 0.558 / 1.000 / 0.980 / 0.932 / 0.0025 | 0.525 / 0.969 / 0.978 / 0.941 / 0.0020 | 0.54 |
| 1000, s=0.8 | 0.615 / 1.000 / 0.993 / 0.910 / 0.0040 | 0.585 / 1.000 / 0.987 / 0.906 / 0.0043 | 0.54 |
| 1000, s=0.6 | 0.650 / 1.000 / 0.996 / 0.878 / 0.0075 | 0.629 / 1.000 / 0.999 / 0.867 / 0.0073 | 0.54 |
| 1000, s=0.5 | 0.637 / 1.000 / 1.000 / 0.862 / 0.0132 | 0.644 / 1.000 / 1.000 / 0.845 / 0.0118 | 0.54 |
| 1000, s=0.25 | 0.456 / 0.875 / 0.997 / 0.910 / 0.0043 | 0.571 / 0.656 / 0.999 / 0.724 / 0.0838 | 0.54 |
| uniform 250, s=1.0 | 0.542 / 0.969 / 0.985 / 0.942 / 0.0006 | 0.517 / 0.906 / 0.986 / 0.943 / 0.0000 | 0.14 |
| uniform 250, s=0.8 | 0.596 / 1.000 / 0.991 / 0.914 / 0.0015 | 0.589 / 1.000 / 0.994 / 0.916 / 0.0006 | 0.14 |
| uniform 250, s=0.5 | 0.629 / 1.000 / 1.000 / 0.870 / 0.0083 | 0.679 / 1.000 / 1.000 / 0.861 / 0.0077 | 0.14 |
| late-heavy 250, best-s (0.6 warp / 0.5 logits) | 0.653 / 1.000 / 0.999 / 0.883 / 0.0041 | 0.658 / 1.000 / 1.000 / 0.847 / 0.0083 | 0.14 |
| late-heavy 250, s=0.8 (extra) | 0.615 / 0.969 / 0.993 / 0.911 / 0.0018 | 0.586 / 1.000 / 0.993 / 0.910 / 0.0023 | 0.14 |

Snippets (warp, first sample, same noise seed). control: `s, and is at in nor suave, and io tho\ndhet nou someas, that thed it live pentiourly, but that his aiviour` ; s=0.8: `at the iskat in nor suave, and he shood by the somies, that them\nit live pentioutly, but them his astrourd on the mulp in the ward ded` ; s=0.6: `at the ittat in nor shave, and he shind of the somies, that the \nit live pentiontly, but they has acleoped on the molm of the ward her`. Same underlying text: lowering s mostly swaps garbled tokens for frequent short words ("io tho"->"he", "thed"->"them", "that his"->"them his").

Verdict (rule: word_hit +>=0.05 with d4 >= ~0.90, r8 <= ~0.01, samples no worse): only s=0.8 at 1000 steps passes for the warp run (+0.057, d4 0.910, r8 0.004, utf8 1.0); s=0.6/0.5 raise word_hit more (+0.09/+0.08) but d4 falls to 0.88/0.86 and the gain is largely frequent-short-word substitution, so not "better" by the rule. s=0.25 is bad (utf8 0.88 / 0.66, repeats). For plain logits only s=0.8 is borderline (+0.06, d4 0.906). Respacing: uniform 250 at s=1.0 is ~equal (warp -0.016, logits -0.008, within +-0.03 noise) and 3.9x faster (0.14 vs 0.54 s/sample), but its valid_utf8 at s=1.0 dips (0.969 / 0.906); with s=0.8 it recovers to 1.0 and is +0.039 / +0.064 over the control (warp misses the +0.05 bar by 0.011; d4 0.914, r8 0.0015, the cleanest diversity of any variant). Late-heavy 250 at s=0.8 equals uniform 250 / 1000-step s=0.8 (0.615 / 0.586), i.e. no benefit from the non-uniform grid. Single seed, 32 samples (+-0.03). The text is still not English, so these are sampler-level cosmetic gains, not a capability change. Recommended for the long run's final evaluation: report s=1.0/1000 steps as the primary (control) number and s=0.8 as the sampler-tuned number; 1000 steps s=0.8 is the single setting that satisfies the rule, uniform 250 steps s=0.8 is the ~4x cheaper near-equivalent.
