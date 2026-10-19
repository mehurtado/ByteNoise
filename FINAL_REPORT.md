# ByteNoise final report

## (a) What the project is
A research prototype: a byte-level (256-symbol, no tokenizer) text **diffusion** language model trained from scratch on one RTX 4060 Ti 8 GB. Bytes are embedded, Gaussian noise is added to the embeddings, a compact 1-D U-Net (14.4M params, 256-byte context) denoises, and text is produced by iterative refinement (1000-step DDPM sampling) from pure noise. Data: Project Gutenberg prose (natural-v1: 57 books; natural-v2: +278 books in train only, 160 MB; val/test byte-identical to v1). The project must remain diffusion-based.

## (b) What was tried, and the evidence
Metric: word-hit = fraction of generated alphabetic words found in the natural-v1 train word list (8,661 words); 32 samples, seed 0, EMA weights, 1000 steps. Single seed throughout; word-hit noise is about +-0.03-0.04.

| Run | word-hit | valid UTF-8 | Note |
|---|---|---|---|
| Real val text (reference) | 0.933 | 0.97 | upper bound for the metric |
| Baseline gate (eps-MSE, U-Net, 45 min, 24.7k upd) | 0.351 | 0.81 | |
| Long eps run (104k upd) | 0.465 | 0.03 | plateau; byte validity collapsed |
| A: self-conditioning | 0.383 | 0.375 | |
| B: hybrid denoiser | 0.356 | 0.94 | bf16 |
| C: hybrid + self-cond + weighted CE | 0.332 | 0.97 | bf16 |
| Sampler sweeps | hybrid C at s=0.5: 0.473 (diversity loss); U-Net: no gain | | |
| L1: logit head (CE on bytes), embed_norm 8 | 0.525 | 0.97 | 42k upd in 45 min |
| L2: L1 + time warping | 0.558 | 1.00 | 45.5k upd |
| L3: logits, embed_norm 16 | 0.513 | 0.97 | 48.6k upd |
| **FINAL: L2 recipe, 4 h, natural-v2** | **0.587** | **0.97** | 248k upd; see (d) |

## (c) Why the logit head
Every eps-prediction variant stayed at word-hit 0.33-0.47 and the 4x-longer run plateaued, so the bottleneck looked like the training target, not capacity or training time. The change adopted is CDCD-style: the network outputs 256 byte logits, trained with cross-entropy at every noise level; x0_hat = softmax(logits) @ embedding table is used inside the DDPM posterior. It is still diffusion (noise on byte embeddings, iterative refinement, 1000-step DDPM sampling, self-conditioning input); only the training target changed from eps-MSE to byte CE. It was the only change that moved word-hit well past the other variants (0.35 -> 0.51-0.56 at equal wall-clock) and restored valid UTF-8, and it is also faster per update.

## (d) Final long run (runs/diff_nat_v2_long_s0)
Setup: natural-v2 train, logits + warp, embed_norm 8, compact U-Net 14.4M, fp32, bs32, lr 3e-4, seed 0, 4 h = 248,383 updates (58 ms/update), peak VRAM 708 MB, 7.95M examples = **12.7 epochs** over natural-v2 train. EMA weights.

| sampler | word-hit | valid UTF-8 | distinct-4gram | repeated-8gram | memorization | s/sample |
|---|---|---|---|---|---|---|
| 1000 steps, s=1.0 (control) | 0.587 | 0.969 | 0.933 | 0.0015 | 0.000 | 0.34 |
| 1000 steps, s=0.8 | 0.645 | 1.000 | 0.901 | 0.0044 | 0.000 | 0.34 |
| 1000 steps, s=0.6 | 0.678 | 1.000 | 0.860 | 0.0122 | 0.000 | 0.34 |
| uniform 250, s=1.0 | 0.561 | 1.000 | 0.944 | 0.0015 | 0.000 | 0.09 |
| uniform 250, s=0.8 | 0.623 | 1.000 | 0.919 | 0.0028 | 0.000 | 0.09 |
| real val text | 0.933 | 0.969 | 0.922 | 0.0060 | - | - |

(s scales the sampler's injected noise std; memorization = share of 32-byte sample windows found verbatim in the natural-v2 train set. eval.py's own generation set, a different sample draw, shows 0.008, plausibly from whitespace runs.)

Training diagnostics: val CE (EMA, nats/byte, uniform t) fell 0.4225 (5k) -> 0.3925 (65k) -> 0.3837 (248k), essentially flat over the last 30k updates (LR had decayed to 1e-6). **No overfitting:** a matched check (scripts/overfit_check.py, same t/noise procedure on 1024 evenly spaced windows) gives train CE 0.3865 vs val CE 0.3772 (noise seed 0) and 0.4225 vs 0.4121 (seed 1); train is not lower than val, and val never rose. (The logged training loss, ~1.9, is not comparable: warp sampling oversamples hard high-noise steps.) The remaining error is the highest-noise quartile: val CE by t quartile 0/0/0.023/1.53, x0 accuracy 1.00/1.00/0.994/0.574; one-step recovery 1.00/1.00/0.998/0.866/0.063 at t=200/400/600/800/999.

Compared with the 45-min L2 run (same seed), 5.5x more updates and 3.7x more data gave word-hit 0.558 -> 0.587 (inside noise) and val CE 0.396 -> 0.384. The monitor (16 samples, so +-0.04) rose from 0.48 at 10k updates to ~0.58-0.60 by 110k and then stayed flat. In short: more compute at this size gave a small, saturating improvement.

Samples (same noise seed, first 250 chars; s=1.0 then s=0.8):
- `s, and is at in nor seely, and he would by\npiies me so that thet it live that out of buost he his endernation in\nnow whan the cord herself at remost of the bow. And those fleose\nhim douages with a Ren' as aurence what...`
- `s, and is at in for seely, and he would by\nshe some so that them it live that out of brost he has enterned on in\nthe wafe the cord herself at remost of the bow, and those clease\nhis dointes with a Real as aurence what...`
- `e\nbonly, she would serong it, and lelle rood, as\nthen I have not a minate) by the loweref cartiel with the plasers itt on\nI him crouts was nod hered I har and cresent over the impertainty gliqued...`

Honest reading: the output has English-like letter statistics, correct spacing, punctuation, quotes, paragraph breaks and many real short words, but it is **not coherent English**: most longer words are invented and there is no grammar or meaning. The s=0.8/0.6 gains come largely from swapping garbled tokens for frequent short words ("thet"->"them", "io tho"->"he"), with diversity dropping (distinct-4gram 0.93 -> 0.86). Real text scores 0.933, so the word-hit gap remains large. No evidence of memorization.

## (e) Caveats
- Single seed everywhere; 32 samples (+-0.03-0.04 word-hit). Differences of ~0.03 are not resolved.
- Update counts are confounded: logit runs got ~1.7-2x more updates than the eps gate in 45 min, so part of the L1-L3 gain may be update count (though the 104k-update eps run reached only 0.465).
- The eps gate runs mix precision: gate/A fp32, hybrid B/C bf16. The final run is fp32.
- word-hit rewards frequent short words and ignores meaning; it is a proxy, not a quality measure. Valid UTF-8 on 32 samples moves in steps of 0.03.
- natural-v2 train differs from v1 (more books); val/test are identical, so val CE is comparable, but earlier runs' word lists come from v1 train (kept for comparability).
- Sampler settings (s=0.8) were chosen on the same 32 seed-0 samples they are reported on.
- Infill/validity scripts are not yet logits-aware and were not rerun.

## (f) Next steps
1. Second and third seeds of the L2 recipe and the eps gate, with more samples, to size the real effect.
2. Hybrid (conv/rnn/attn/ngram) denoiser with the logit head; hybrids were only tried with the eps target.
3. Longer-context tests (256 bytes now) and scaling width/depth, since the 4 h run saturated at this size.
4. Make infill/validity scripts logits-aware.
5. Block diffusion (denoise a block, condition on previous blocks) for longer text at modest cost.
6. A semantic metric beyond word-hit (e.g. byte-level held-out likelihood bounds or bigram validity).

Artifacts: runs/diff_nat_v2_long_s0/{gate_text.txt, eval_out.txt, recovery.txt, final_sweep.json, overfit_check.txt, samples_log.md}; scripts/final_sweep.py, scripts/overfit_check.py.

## Ladder results (addendum)
New harness (scripts/ladder_eval.py: 128 paired samples, bootstrap CIs, byte-level Kneser-Ney n-gram bits/byte scorer, deterministic CE by noise level) re-scored the earlier runs; n-gram score and word-hit rank them nearly identically (Spearman +0.98), so the scorer is usable as a second metric. Equal-update ladder (25k updates, natural-v1, L2 recipe, 2 seeds each, mean val CE / word-hit / n-gram bpb at s=1.0): U-Net w48 (3.7M) 0.405 / 0.44 / 4.96; w96 (14.4M) 0.390 / 0.52 / 4.31; w192 (57.5M) 0.386 / 0.55 / 4.15; hybrid denoiser with logit head (14.0M, fp32) 0.371 / 0.61 / 3.73. Seed spread is tiny (CE <=0.001, word-hit <=0.01, bpb <=0.09), so all of these gaps are real: width helps with diminishing returns (w96->w192 costs 2.2x time for CE -0.005), and hybrid+logits beats w96 and even w192 at equal updates at ~3x w96 time per update (163 vs 55 ms) and 5x VRAM (3.5 GB). Samples remain non-English; see EXPERIMENTS.md for the full tables and caveats.

## Rungs (addendum)
Single-seed screens at width 96 / 25,000 updates / natural-v1 (L2 recipe), judged against a 4-seed baseline (SD: CE 0.0012, word_hit 0.004, bpb 0.028; a two-run difference counts as real only above CE 0.0035, word_hit 0.012, bpb 0.08). Full table and caveats in EXPERIMENTS.md ("Rungs").
- Embedding radius 4 and 6: no change in sample quality (word_hit 0.519 / 0.517 vs baseline 0.518; bpb 4.32 vs 4.33). Val CE is not comparable across radii.
- Tied output head (logits = z @ E^T with learned temperature): no gain (word_hit 0.519, bpb 4.36), CE t>=750 marginally worse; not worth keeping.
- Per-word span noise: p=0.3 neutral, p=0.6 worse (bpb +0.12); the t>=750 CE bottleneck (~1.56 nats) is unchanged.
- Batch 128 / lr 6e-4: at equal examples about equal quality (CE +0.0026, inside noise) at ~2.9x the throughput; with 4x the examples (25,000 updates) it is clearly better (word_hit 0.579, bpb 3.90, CE t>=750 1.526), matching the 4 h bs32 run in about an hour. The gain is examples seen, not a head/noise trick.
- Recommendation (diffusion only): R8, untied head, no span noise, bs128/lr 6e-4; next long run = hybrid+logits (best at equal updates: word_hit 0.613, bpb 3.73) with bs128, on natural-v2. That combination is untested.
- Caveats: seed 0 only for all rungs (no rung met the confirmation threshold); the machine ran ~2x slower after the ladder, so only relative ms/update (tied +2.5%, span +1.6%) is meaningful; bs128 lr 6e-4 vs 3e-4 not separated.

## Hybrid long run (addendum)
Run: runs/diff_nat_v2_hybrid_bs128_s0. Hybrid denoiser (14.05M params) + logit head, L2 recipe, natural-v2, effective batch 128 (32 x 4 accumulation, fp32, 3.55 GB peak), lr 6e-4 (300-update check was stable, no fallback), cosine to the 5 h wall-clock budget: 29,634 updates (3.79M examples), 587 ms/update. Still a pure diffusion model (1000-step ancestral sampling, no autoregression).
Result (ladder_eval, 128 paired samples, EMA): word_hit 0.683 [0.669, 0.697] / bpb 3.31 at s=1.0 and 0.774 / 2.68 at s=0.8 (real text 0.940 / 2.05); val CE 0.3558, CE at t>=750 1.438. Against the 4 h w96 U-Net long run (0.581 / 3.89; 0.630 at s=0.8; CE 0.3733) and the 25k-update bs32 hybrid (0.607 / 3.75; 0.709 at s=0.8; CE 0.3710): +0.10 word_hit and -0.58 bpb over the U-Net, +0.08 word_hit and -0.44 bpb over the short hybrid at s=1.0. valid_utf8 0.969 (s=1.0), 0.992 (s=0.8); distinct 4-grams 0.93; memorization 0.001.
Samples (s=1.0): "...The senasence of Venalitas in the rade memberine of expressessable in sprangerier, and large nothing..." and "height with the own times to girl in collage, they raid for-sumber the firit, from which they was intience the pine werey heevon". Mostly real English words and punctuation with plausible local syntax, but the sentences are still not coherent and many words are fused or misspelled; at s=0.8 spellings are cleaner ("fountaine centice", "light with the own times to curchen") but meaning does not improve.
Caveats: single seed (the earlier baseline seed SDs were word_hit about 0.004 and bpb about 0.03 for w96, so the gaps are large relative to them, but the hybrid long run itself was not repeated); the comparison mixes changes (architecture, 4x more examples per update, lr, 1.0-1.2x more updates than the short hybrid, 5 h vs 4 h of compute) and was not designed to separate them; the compute-matched U-Net comparison is the 4 h bs32 run, which had about 8M examples vs 3.8M here; s=0.8 was chosen on earlier runs, not tuned for this model; the n-gram scorer and word list come from natural-v1 train, and word_hit/bpb are surface-statistics proxies, not a measure of meaning; the validation CE improvement had nearly plateaued (logged EMA CE 0.3055 at 10k -> 0.2982, flat in the last 5k updates, partly due to the cosine tail), so more of the same would give small gains; the sample monitor's own word_hit (about 0.70 at s=1.0 on 16 samples) uses a different sample set/word list and is not comparable to the table.
