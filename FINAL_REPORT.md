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
