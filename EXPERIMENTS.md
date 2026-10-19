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

### Final long run: runs/diff_nat_v2_long_s0 (L2 recipe, natural-v2 train, 4 h)
logits + warp, embed_norm 8, compact U-Net 14.4M, fp32, bs32, lr 3e-4, seed 0; 248,383 updates (58 ms/update, peak VRAM 708 MB), 7.95M examples = 12.7 epochs over natural-v2 train (160 MB). EMA, 32 samples, seed 0, natural-v1 word list (scripts/final_sweep.py; own loop byte-identical to model.sample).
| sampler | word-hit | utf8 | d4 | r8 | memorization | s/sample |
|---|---|---|---|---|---|---|
| 1000, s=1.0 | 0.587 | 0.969 | 0.933 | 0.0015 | 0.000 | 0.34 |
| 1000, s=0.8 | 0.645 | 1.000 | 0.901 | 0.0044 | 0.000 | 0.34 |
| 1000, s=0.6 | 0.678 | 1.000 | 0.860 | 0.0122 | 0.000 | 0.34 |
| uniform 250, s=1.0 | 0.561 | 1.000 | 0.944 | 0.0015 | 0.000 | 0.09 |
| uniform 250, s=0.8 | 0.623 | 1.000 | 0.919 | 0.0028 | 0.000 | 0.09 |
Val CE (EMA, uniform t): 0.4225 @5k, 0.3925 @65k, 0.3837 @248k (flat the last ~30k). Matched train-vs-val CE (scripts/overfit_check.py, 1024 windows): train 0.3865 / val 0.3772 (seed 0), 0.4225 / 0.4121 (seed 1): no overfitting after 12.7 epochs. eval.py val: ce 0.3878, x0 acc by t quartile 1.00/1.00/0.994/0.574, valid UTF-8 0.969, memorization 0.008 (its own sample draw). Recovery t=200..999: 1.00/1.00/0.998/0.866/0.063. Monitor word-hit (16 samples): 0.48 @10k -> ~0.58-0.60 from 110k, flat after. Versus the 45-min L2 run: word-hit 0.558 -> 0.587 (within noise), val CE 0.396 -> 0.384 - more compute gave a small, saturating gain. Samples are still not coherent English; s<1 gains are largely frequent-short-word substitution with lower diversity. Single seed, +-0.03-0.04. See FINAL_REPORT.md.

### Ladder: evaluation harness + equal-update width/hybrid ladder (scripts/ladder_eval.py, ngram_score.py, ladder_report.py, run_ladder.sh)
**Harness.** EMA weights, 128 samples (4 batches of 32, fixed per-batch seeds so initial noise is paired across runs), 1000-step DDPM at s=1.0 and s=0.8 (s only for logits runs; eps runs have s=1.0 only), natural-v1 word list. Metrics: word_hit with 95% bootstrap CI (resampling samples; this is sampling noise only, not training-seed noise), valid_utf8, distinct_4gram, repeated_8gram, memorization (vs the run's own train set), bits/byte under a scoring-only byte-level order-5 interpolated Kneser-Ney model trained on natural-v1 train (cached data/ngram_natural-v1.pkl, gitignored; normalisation checked, max |sum P - 1| = 5e-8; scored at positions >= 4 so all have a full history). Controls on the same 128 real val windows: real val 2.05 bpb / word_hit 0.94; letters shuffled within window 7.61 / 0.13; bytes shuffled 8.12 / 0.31; (train-window bpb 1.96). Deterministic val CE (2048 evenly spaced val windows, fixed t and noise, same for all runs) overall, t in [500,750), t>=750, logits runs only. s/sample at batch 32 (first batch dropped as warm-up), params, train peak VRAM and ms/update (median of consecutive-log deltas) from metrics.jsonl. Outputs ladder_eval.json/.md/ladder_samples.json (hex) per run; pre-existing runs were re-scored read-only into runs/_ladder_rescore/<run>/ (their own dirs untouched).

**Part 1 re-score** (s=1.0; val CE = this harness's deterministic CE, not comparable to the old uniform-t eval numbers to 3 digits):
| run | updates | word_hit [CI] | n-gram bpb | utf8 | val CE | CE t>=750 | s=0.8 wh / bpb |
|---|---|---|---|---|---|---|---|
| L1 | 42.1k | 0.540 [0.527,0.551] | 4.25 | 0.961 | 0.3810 | 1.548 | 0.604 / 3.70 |
| L2 | 45.5k | 0.562 [0.550,0.573] | 4.04 | 0.969 | 0.3829 | 1.537 | 0.608 / 3.59 |
| L3 (R=16; CE not comparable) | 48.6k | 0.509 [0.497,0.521] | 4.47 | 0.961 | 0.2082 | 0.806 | 0.580 / 3.90 |
| wide192 45 min | 25.6k | 0.559 [0.547,0.570] | 4.05 | 0.984 | 0.3827 | 1.540 | 0.620 / 3.54 |
| v2 long 4 h | 248k | 0.581 [0.567,0.594] | 3.89 | 0.992 | 0.3733 | 1.502 | 0.630 / 3.46 |
| eps gate | 24.7k | 0.338 [0.328,0.347] | 6.05 | 0.859 | - | - | n/a |
| eps hybrid (bf16) | 19.9k | 0.364 [0.352,0.376] | 5.65 | 0.953 | - | - | n/a |
| eps hybrid+selfcond | 17.6k | 0.314 [0.302,0.326] | 5.87 | 0.938 | - | - | n/a |
All 8 scored (eps runs via model.sample, no CE/s=0.8). **Metric validation:** Spearman(word_hit, -bpb) = +0.98 over 8 runs at s=1.0 and +1.00 over the 5 logits runs at s=0.8; the only rank swap is among the worst two eps runs (gate vs hybrid+selfcond). Both put long run > L2 ~ wide192 > L1 > L3 > eps hybrid > eps gate. Caveat: the n-gram scorer and word_hit are both local-statistics metrics and share blind spots (neither sees meaning); their agreement shows the scorer is a sane, more continuous second metric, not that either measures coherence. Note also that long run vs L2/wide192 at s=1.0 is only ~0.02 word_hit (CIs overlap) but 0.15-0.17 bpb.

**Part 2 ladder.** L2 recipe (logits, warp, embed_norm 8, ce_weight 0, x0_rescale, bs32, lr 3e-4, fp32, natural-v1), 25,000 updates, max_seconds unset (0), 2 seeds each. LR schedule verified tied to max_updates (cosine, 3e-4 -> 1e-6 exactly at 25000/25000 in the smoke and real logs; no warmup). Hybrid = `--preset hybrid` (dim 256, 7 layers) with diffusion.param=logits, fp32: trained fine without bf16 (3.5 GB peak VRAM). No crashes or resumes. Seed-1 checkpoints deleted after eval; seed-0 checkpoints kept.
Val CE (EMA, uniform t, from metrics.jsonl) at 6k/12k/18k/25k: w48 0.4324/0.4198/0.4149/0.4136 (s1 0.4320/0.4200/0.4153/0.4138); w96 0.4144/0.4045/0.3994/0.3980 (s1 0.4136/0.4037/0.3989/0.3973); w192 0.4057/0.4000/0.3955/0.3942 (s1 0.4052/0.3990/0.3949/0.3927); hybrid 0.3960/0.3858/0.3814/0.3799 (s1 0.3937/0.3831/0.3786/0.3771).
Per configuration, mean +- half the seed difference (n=2, so "spread" is |s0-s1|/2 and itself uncertain):
| metric | w48 | w96 | w192 | hybrid+logits |
|---|---|---|---|---|
| params | 3.65M | 14.45M | 57.5M | 14.05M |
| ms/update (median) | 51.5 | 55.2 | 119 (108.7 / 129.5) | 163.3 |
| train peak VRAM | 328 MB | 708 MB | 1799 MB | 3499 MB |
| val CE overall | 0.4051 +- 0.0005 | 0.3901 +- 0.0004 | 0.3855 +- 0.0005 | 0.3708 +- 0.0002 |
| val CE t>=750 | 1.616 +- 0.003 | 1.563 +- 0.002 | 1.551 +- 0.002 | 1.499 +- 0.000 |
| word_hit s=1.0 | 0.437 +- 0.005 | 0.521 +- 0.001 | 0.549 +- 0.002 | 0.613 +- 0.006 |
| n-gram bpb s=1.0 | 4.96 +- 0.03 | 4.31 +- 0.01 | 4.15 +- 0.04 | 3.73 +- 0.02 |
| valid_utf8 | 0.949 | 0.922 | 0.957 | 0.891 (all +-0.02, noisy at 128 samples) |
| distinct_4gram | 0.949 | 0.939 | 0.942 | 0.939 |
| word_hit / bpb at s=0.8 | 0.492 / 4.48 | 0.595 / 3.74 | 0.617 / 3.62 | 0.720 / 2.98 |
| s/sample (b32) | 0.35 | 0.35 | 0.75 | 1.21 |
| sampling peak VRAM | 60 MB | 121 MB | 325 MB | 261 MB |
Paired bootstrap on samples (sampling noise only): w96-w48 d_bpb -0.65 [-0.70,-0.61], d_wh +0.084; w192-w96 -0.16 [-0.21,-0.12], +0.028 [+0.019,+0.037]; hyb-w96 -0.58 [-0.63,-0.54], +0.092 [+0.082,+0.101]. Seed-to-seed differences (s0-s1): bpb 0.03-0.09, word_hit <=0.012, val CE <=0.0011.
Honesty notes: (1) ladder_hyb_s0 wall time was 11,023 s vs 4,113 s for s1 though ms/update medians agree (163.8 vs 162.8); the stretch is not explained by the logs (machine sleep or contention with other work) and does not affect the update-matched results. (2) ladder_w192 ms/update differs between seeds (108.7 vs 129.5), so w192 cost is +-10%. (3) Only 2 seeds; spread estimates are crude. (4) The CE numbers differ slightly between metrics.jsonl (32 batches) and the harness (2048 windows) as they use different t/noise draws. (5) Hybrid valid_utf8 is the lowest (0.89), so its text is not uniformly cleaner; the gain is in word/n-gram statistics and CE. (6) Memorization 0.000 everywhere.


### Rungs: batch size, embedding radius, tied output head, per-word (span) noise (scripts/run_rungs.sh, rung_report.py; tests/test_rungs.py)
Setup: width 96 (14.45M), natural-v1, L2 recipe (logits, warp, embed_norm 8, ce_weight 0, x0_rescale, fp32, bs32, lr 3e-4), 25,000 updates unless stated, seed 0, evaluated with ladder_eval.py (128 paired samples, same noise seeds as the ladder, EMA). Code (all default-off, excluded from architecture_hash, old checkpoints load): `model.tied_head` (logits = exp(log_scale) * z @ E^T, z = byte_embed_dim projection of the U-Net output, learnable scalar temperature initialised to 1/row-norm so initial logits are O(1); E is NOT detached in the head, x0_hat keeps its own detach) and `diffusion.span_noise_prob` (per example, with prob p the byte window is split into words - ASCII letters plus bytes >=128 form words, each word also takes the non-letter bytes preceding it - and each word gets an independent t from the same uniform/warp sampler; an extra input channel carries sqrt(alpha_bar) per position, constant sqrt(ab_t) for ordinary examples, validation and sampling; time embedding / CE buckets / warp stats use the example's mean t; training mode only). Tied-head stability check before the runs (CPU, debug-size model, 600 updates): val CE untied 1.012, tied 1.053, tied with E detached 1.633, so the non-detached head was used.

**Baseline seed spread (4 seeds of w96: ladder s0,s1 + new s2,s3).** Mean / SD / minimal real difference of two single runs, 2*sqrt(2)*SD: val CE 0.3895 / 0.0012 / 0.0035; CE t>=750 1.562 / 0.0025 / 0.0069; word_hit s1.0 0.518 / 0.0041 / 0.012; word_hit s0.8 0.587 / 0.0099 / 0.028; bpb s1.0 4.33 / 0.028 / 0.08; bpb s0.8 3.79 / 0.065 / 0.18. (The 2-seed half-ranges used earlier understated the spread of the sample metrics.) Confirmation rule used: seed-0 result better than the baseline mean by more than these minimal differences on CE, word_hit s1.0 or bpb s1.0.

| config | params | ms/upd | val CE all | CE t>=750 | word_hit s1.0 [CI] | word_hit s0.8 | bpb s1.0 | bpb s0.8 | utf8 | d4 | s/sample |
|---|---|---|---|---|---|---|---|---|---|---|---|
| w96 baseline, mean of 4 seeds | 14.45M | 55 (s0,s1; s2,s3 ran at 104) | 0.3895 | 1.562 | 0.518 [0.505,0.530] | 0.587 | 4.33 | 3.79 | 0.936 | 0.939 | 0.35 (0.8 when slow) |
| bs128, lr 6e-4, 6250 upd (equal examples) | 14.45M | 145 | 0.3921 | 1.572 | 0.509 [0.498,0.518] | 0.568 | 4.39 | 3.92 | 0.969 | 0.938 | 0.80 |
| bs128, lr 6e-4, 25000 upd (4x examples) | 14.45M | 147 | 0.3797 | 1.526 | 0.579 [0.568,0.590] | 0.631 | 3.90 | 3.48 | 0.961 | 0.936 | 0.81 |
| R4 (embed_norm 4) | 14.45M | 104 | 0.7433 (not comparable) | 2.499 (nc) | 0.519 [0.508,0.530] | 0.576 | 4.32 | 3.86 | 0.945 | 0.945 | 0.80 |
| R6 (embed_norm 6) | 14.45M | 105 | 0.5089 (nc) | 1.966 (nc) | 0.517 [0.507,0.528] | 0.586 | 4.32 | 3.77 | 0.945 | 0.939 | 0.84 |
| tied head, R8 | 14.43M | 106 | 0.3910 | 1.575 | 0.519 [0.508,0.530] | 0.579 | 4.36 | 3.82 | 0.930 | 0.938 | 0.82 |
| tied head, R6 | 14.43M | 106 | 0.5114 (nc) | 1.988 (nc) | 0.511 [0.499,0.522] | 0.576 | 4.38 | 3.86 | 0.930 | 0.941 | 0.81 |
| span noise p=0.3 | 14.45M | 107 | 0.3871 | 1.562 | 0.513 [0.501,0.524] | 0.574 | 4.38 | 3.94 | 0.930 | 0.940 | 0.80 |
| span noise p=0.6 | 14.45M | 107 | 0.3902 | 1.576 | 0.509 [0.498,0.520] | 0.569 | 4.45 | 4.06 | 0.945 | 0.943 | 0.81 |
Notes: (1) The machine slowed down by about 2x after the ladder (identical code: the baseline went 55 -> 104-105 ms/update; a clean re-measurement of 400 updates now gives base 105.3 / 105.1, tied 108.0, span 107.0), so only ratios are meaningful: the tied head costs about +2.5% and span noise about +1.6% per update, bs128 costs 1.38x per update for 4x the examples (about 2.9x the throughput per example). The ladder's own 55 ms and 0.35 s/sample are the uncontended values. (2) Val CE depends on the embedding radius at fixed t (R4 0.74, R6 0.51, R8 0.39, R16 0.21) because the signal-to-noise ratio of x_t changes, so CE is NOT comparable across radii; only the sample metrics are. This also means the driver's choice of the radius for the tied-head run (lower CE of R4/R6) simply picked the larger radius, R6. (3) bs128 25000-update run is a companion (4x the data seen), not a screening rung, and was not seed-confirmed.

**Batch size.** At equal examples (800k) bs128/lr 6e-4 is a hair worse than bs32/lr 3e-4: CE +0.0026 (below the 0.0035 real-difference threshold), word_hit -0.009, bpb +0.06 (both inside the spread), CE t>=750 +0.010 (marginal). It is, however, 4x fewer updates and about 2.9x faster per example, so it is a throughput win at no clear quality cost. At equal updates (25000) bs128 sees 4x the examples and is clearly better (CE -0.0099, CE t>=750 1.526 vs 1.562, word_hit +0.061, bpb -0.43, s0.8 0.631 / 3.48): the improvement is examples seen (a step-count-limited run is under-trained), not a batch-size effect per se. lr 3e-4 vs 6e-4 was not separated. Trajectory caveat: quick_eval draws t/noise per batch, so its CE depends on the batch size; the same final checkpoints read 0.4002 (bs128 run) / 0.3980 (baseline s0) under the bs32 protocol and 0.3798 / 0.3769 under the bs128 protocol. The rung_summary.md trajectory table therefore adds a +0.0208 shift to the logged bs128 values; with it bs128 trails bs32 by ~0.02 at 64k examples and ~0.003 at the end (shift assumed constant; deterministic ladder_eval CE at the end is the authoritative comparison: +0.0026).
**Radius.** R4 and R6 equal the baseline on every sample metric (word_hit 0.519 / 0.517 vs 0.518; bpb 4.32 / 4.32 vs 4.33; s0.8 within spread). Radius 4-8 is flat; the earlier R=16 result (single seed, worse) stands as the only radius effect.
**Tied head.** No effect: R8 word_hit 0.519, bpb 4.36 (+0.03, inside 0.08), CE +0.0015, but CE t>=750 +0.013 (about 2x the real-difference threshold, i.e. mildly worse on the bottleneck region); R6 word_hit -0.007, bpb +0.05, inside spread. It saves 17k parameters. Not a gain.
**Span noise.** p=0.3 is neutral (CE -0.0024 inside the threshold, CE t>=750 identical 1.562, word_hit -0.005, bpb +0.05; s0.8 bpb 3.94 vs 3.79 is at the edge of the 0.18 threshold). p=0.6 is worse: bpb +0.12 and s0.8 bpb +0.27 (both beyond the thresholds), CE t>=750 +0.014, word_hit -0.009. The idea of teaching the net to use clean neighbouring words did not reduce the t>=750 error (the high-noise bottleneck stays at about 1.56 nats).
**Confirmation stage.** With the 4-seed rule only the bs128 25000-update companion exceeds the thresholds (a 4x-examples effect); no screening rung (bs128 equal examples, R4, R6, tied R8, tied R6, span 0.3, span 0.6) qualified, so no seed-1 runs were made. Single-seed screens cannot exclude small effects below the thresholds (CE 0.0035, word_hit 0.012, bpb 0.08).
**Recommendation (diffusion only).** Keep R8, untied head, no span noise. Use batch size 128 / lr 6e-4 for long runs (about 2.9x the examples per second at equal quality per example; the 25000-update bs128 run reached word_hit 0.579 / bpb 3.90 in about an hour at the slowed speed, comparable to the 4 h bs32 natural-v2 run's 0.581 / 3.89). The remaining lever is model capacity/architecture: hybrid+logits (word_hit 0.613, bpb 3.73, CE t>=750 1.499 at 25000 bs32 updates) beat every rung here, so the next run should be hybrid+logits with bs128 / lr 6e-4 (untested combination, so it needs its own check of lr stability) trained on natural-v2.


### Hybrid long run: runs/diff_nat_v2_hybrid_bs128_s0 (hybrid + logits, effective batch 128, lr 6e-4, natural-v2, 5 h)
Setup: hybrid denoiser (14.05M, 7 layers, width 256, conv/rnn/attn/ngram), L2 recipe (param=logits, t_sampling=warp, embed_norm=8, ce_weight=0, x0_rescale; ce_schedule=alpha_bar set as requested but inert with ce_weight=0, the ladder config had const), fp32, microbatch 32 x grad_accum 4 = 128 per update, lr 6e-4 with no warmup, cosine to 1e-6 tied to wall clock (max_seconds=18000), eval/checkpoint every 1000 updates, seed 0. Script: scripts/run_hybrid_long.sh (resumes on crash; no crash occurred). sample_monitor.py ran beside it (interval 900, natural-v2).
lr-stability check (300 updates, own short cosine, runs/_stab_hyb_bs128): loss 2.78 -> 1.08 by update 30, gnorm 1.8 -> 0.15-0.3, no NaN or spikes, so 6e-4 was kept (no fallback to 3e-4). Peak train VRAM 3552 MB. Speed 586.7 ms/update, 29,634 updates = 3.79M examples in 18,000 s.
Eval (scripts/ladder_eval.py, defaults: 128 paired samples, EMA, 1000 steps, word list/n-gram from natural-v1 train, same as the other rows):
| run | updates x batch | word_hit s1.0 [CI] | bpb s1.0 | word_hit s0.8 | bpb s0.8 | utf8 s1.0/0.8 | val CE | CE t>=750 |
|---|---|---|---|---|---|---|---|---|
| hybrid bs128 long (this) | 29,634 x 128 | 0.683 [0.669,0.697] | 3.31 | 0.774 [0.761,0.786] | 2.68 | 0.969 / 0.992 | 0.3558 | 1.438 |
| w96 U-Net long (v2, bs32) | 248,383 x 32 | 0.581 [0.567,0.594] | 3.89 | 0.630 [0.614,0.644] | 3.47 | 0.992 / 0.984 | 0.3733 | 1.502 |
| hybrid ladder s0 (25k x 32) | 25,000 x 32 | 0.607 [0.594,0.618] | 3.75 | 0.709 [0.697,0.720] | 3.02 | 0.906 / 0.969 | 0.3710 | 1.499 |
| real val text | - | 0.940 | 2.05 | - | - | 0.953 | - | - |
Other: distinct_4gram 0.929 (s1.0), 0.893 (s0.8); memorization 0.001 / 0.007; sampling 1.23 s/sample (b32). EMA val CE trajectory (logged quick_eval, bs128 protocol): 0.338 (1k), 0.311 (5k), 0.3055 (10k), 0.3003 (18k), 0.2985 (25k), 0.2982 (29.6k); raw and EMA converge, gains per 1000 updates < 0.0003 at the end (partly the cosine decay).
Note on the logged CE: quick_eval draws t/noise per batch, so its absolute value is not comparable with the deterministic ladder CE (0.3558); use the table.


### Sampler-only sweep on the hybrid+logits checkpoint (scripts/sampler_sweep_hybrid.py, output runs/_sampler_hybrid/)
Checkpoint runs/diff_nat_v2_hybrid_bs128_s0 (read-only, EMA), same 128 paired samples (seeds 1000+batch, batch 32) and metrics as ladder_eval; baselines re-measured and byte-identical to the stored ladder_samples.json (0.683/3.31 at s=1.0, 0.774/2.68 at s=0.8). Differences are paired bootstrap over samples vs the same-s 1000-step baseline; * = 95% CI excludes 0. s/sample at batch 32 (the test suite overlapped briefly with the sweep; timings are linear in steps and agree with the earlier 1.23 s/sample). Sampling VRAM peak 261 MB for every setting. Pure diffusion; only the sampler changed (defaults unchanged, main.py untouched).
(1) Steps by uniform respacing (word_hit / bpb / d4 / mean word length; s/sample):
| steps | s=1.0 | s=0.8 | s/sample |
|---|---|---|---|
| 1000 | 0.683 / 3.31 / 0.929 / 4.18 | 0.774 / 2.68 / 0.893 / 4.08 | 1.226 |
| 500 | 0.677 / 3.39 / 0.935 / 4.21 (n.s.) | 0.767 / 2.77 / 0.899 / 4.12 (n.s.) | 0.614 |
| 250 | 0.672 / 3.42* / 0.942 / 4.20 (wh n.s., bpb +0.11*) | 0.757 / 2.82* / 0.907 / 4.12 | 0.307 |
| 125 | 0.656* / 3.50* / 0.935 / 4.26 | 0.742* / 2.82* / 0.900 / 4.16 | 0.154 |
| 64 | 0.633* / 3.66* / 0.949 / 4.25 | 0.723* / 2.96* / 0.917 / 4.14 | 0.079 |
Quality degrades smoothly; 500 steps is indistinguishable from 1000; word length and diversity do not shift, so these losses are real, not a short-word artifact.
(2) Non-uniform spacing, fraction f of the budget in t>=750 (rest uniform below; uniform has 25%): at s=1.0, 250 steps f=0.75: 0.674 / 3.33 (bpb diff +0.02, n.s., vs uniform-250 +0.11*), f=0.5: 0.675 / 3.38; 125 steps f=0.75: 0.677 / 3.35 (bpb +0.04 n.s.; uniform-125 is 0.656* / +0.20*), f=0.5: 0.668 / 3.48*. At s=0.8: 250/f0.75 0.761 / 2.73 (n.s.), 125/f0.75 0.763 / 2.78 (bpb +0.11*, vs +0.145* uniform), f=0.5 no better than uniform. So placing ~75% of steps at t>=750 recovers most of the respacing loss in bpb and word_hit (word_hit differences between spacings are within the CI); the effect is clearest at 125 steps, where hi75 gets 8x speedup at -0.006 word_hit (n.s.) / +0.04 bpb (n.s.) at s=1.0. Word length and d4 are unchanged (4.25 / 0.935), so no short-word artifact.
(3) Logit temperature tau (x0_hat = softmax(logits/tau) E, also fed to self-cond): large word_hit/bpb gains that are mostly a diversity trade: 1000 steps s=1.0 tau 0.85: 0.724* / 2.99*, d4 0.893, word len 4.07; tau 0.7: 0.753* / 2.71*, d4 0.808, r8 0.089, memorization 0.035 (vs 0.001), word len 3.97. At 250 steps, s=0.8, tau 0.7: 0.832 / 2.19 but d4 0.742, r8 0.145, memorization 0.090 (collapse toward repeated/memorized text). Matched-diversity comparison: tau 0.85 at s=1.0 (d4 0.893, 0.724) is clearly worse than simply s=0.8 (d4 0.893, 0.774). Verdict: tau is not a free gain; it is another way to trade diversity for the metrics and is dominated by s at equal d4. Not recommended.
(4) Extra final denoises (re-noise x0_hat to t=15, or t=25 then 10, denoise again): the decoded bytes are identical to the base sampler in all 128 samples (0 differing bytes) at s=1.0 and 0.8; the final argmax is already stable at low noise, so there is nothing to gain (CE at t<750 is ~0.02).
Recommended fast-eval setting: 125 steps, f=0.75 hi-weighted spacing (hi_steps(T, 125, 0.75)), s=1.0 (0.153 s/sample, 8x faster than 1000); use 250 steps hi75 (0.307 s, 4x) when the comparison needs bpb resolution; compare only at the same s and always on the same paired seeds. Keep 1000 steps for reported final numbers. Caveats: one checkpoint, one 128-sample paired set; step/spacing differences of ~0.01 word_hit and ~0.08 bpb are inside the sampling CI; the bpb scorer and word list are natural-v1 train surface statistics.
