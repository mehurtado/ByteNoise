# Progress / handoff (read this first when resuming)

## Personal writing for the NEXT long run (2026-10-19)

User supplied `afarewell.txt` (A Farewell, 262 bytes) and `Speculative Fiction.txt` (A Stained Mind, 9,191 bytes). Original copies and provenance are staged under ignored `data/raw/personal-writing/`; read TRAINING_NOTES.md there before use. Keep current mixture pilot unchanged. User wants only a small influence. Earlier 1% suggestion predates measuring how tiny these works are; recommend initially <=0.1% aggregate share, taken from Gutenberg, with bounded/reported repetitions, not equal weighting between documents. Preserve document boundaries and monitor verbatim copying. Do not commit personal source text or change Discord automatically.


## Latest user direction - 2026-10-16 (takes precedence over the older plan below)

Keep ByteNoise a byte-level DIFFUSION model. The user explicitly corrected the earlier drift into an autoregressive-model project. Existing Transformer/causal recurrent experiments may be useful background, but they are not the requested final architecture.

Do not interrupt the currently running `rec_nat_L8` experiment or alter its code/configuration underneath it. Let it finish and use the results as evidence. The user wants Claude to make an informed implementation decision, not mechanically implement every item of a large plan.

Hardware: RTX 4060 Ti 8 GB VRAM, i9-12900K, 32 GB RAM. Prioritize useful learning per training hour and faster generation. Keep comfortable VRAM headroom; measure rather than promise speedups.

After the active run finishes, inspect what is already implemented and choose the smallest useful next step:
- Correct the diffusion training issues that actually remain: loss reporting, partial accumulation, short-run EMA lag, high-noise reconstruction-loss spikes, and embedding/clipping consistency. Diagnose before changing objectives.
- Establish a short controlled learning result with the existing compact diffusion denoiser.
- If justified, reuse existing recurrent components for a bidirectional diffusion denoiser: noisy byte embeddings plus diffusion timestep -> forward/backward recurrence and optional local convolution -> small per-position FFN with residuals -> predicted noise. Keep the diffusion process. Do not accidentally retain a causal next-byte objective.
- Test fixed shared refinement loops only if the simple recurrent denoiser learns and the extra compute is justified.
- Consider a correct reduced-step DDIM sampler as a separate generation-speed improvement; do not simply skip DDPM iterations.

This is a research prototype. Avoid a framework rewrite, broad test matrices, production hardening, extra configuration layers, or long multi-seed runs by default. Reuse current code. A short smoke test and one meaningful learning comparison are enough for an initial change. Extend testing only when a concrete failure or uncertainty warrants it. Preserve checkpoints, unrelated edits, and current experiment outputs. Report actual evidence and limitations.


Goal: most capable model on RTX 4060 Ti 8GB (see user's 10-step plan: GPU env -> reproducible runs -> trustworthy training/eval -> data+eval suite -> Transformer baseline -> parallel recurrent model -> shared loops -> bounded diffusion -> staged selection -> deliver report). Python: `.venv\Scripts\python`. Use bf16 (fp32 spills VRAM and is ~23x slower). Subagents: sonnet/haiku only, never opus. No Co-Authored-By lines in commits.

## Diffusion status (latest, supersedes 'Next' below)
Done: loss diagnosis, sampler fix (x0_rescale), U-Net vs birnn comparison - see EXPERIMENTS.md 'Diffusion track'. Open: models learn format not answers; try longer training / positional signal / conditioning only if the user wants to continue; DDIM speed item and shared loops deferred (not justified by evidence).

## Final state (2026-10-18)
Diffusion track closed out: logit head + warp + 4 h run reached word-hit 0.587 (0.645 at sampler s=0.8; real text 0.933), not coherent English, no overfitting. See docs/archive/handoffs/FINAL_REPORT.md and EXPERIMENTS.md. Still a research prototype; project stays diffusion-based.

## Done
- Step 1 GPU env (ENVIRONMENT.md, gpu_smoke.py - gpu_smoke.py still uses old diffusion globals, needs presets update).
- Step 2/3 reproducible runs, corrected training loop, tests (tests/, 58 passing): config.py presets, train.py (train/resume/generate/eval CLI), checkpoints validated.
- Step 4 data+eval: data.py (dummy-v1, tasks-v1, natural-v1 = 57 Gutenberg docs, doc-level split + paragraph dedup), tasks.py, eval.py.
- Step 5/6 models_lm.py: Transformer (RoPE, SwiGLU) and recurrent (minGRU-style parallel scan, verified vs sequential, causal) + `loops`/`iter_embedding` for shared refinement (step 7).
- Controlled-task pilots (2000 updates, ~65s each): runs/task_{tf,rec}_s{0,1,2}.

## Next
1. Aggregate task_* seeds (eval.py --run runs/X --split val), record in EXPERIMENTS.md.
2. Natural-text Transformer sweep presets tf_nat_a/b/c (15 min each, wall-clock budget): `train.py train --preset tf_nat_b --run-dir runs/nat_tf_b`; eval val bpb.
3. Recurrent model on natural text at matching params/time; then loop variants (loops=1,2,4; separate blocks at matching depth; wider shared).
4. Bounded diffusion check (compact ByteNoise; known issue: CE term spikes at high noise, clip/embedding scale mismatch).
5. >=3-seed confirmation for finalists, equal-time + equal-data tables, final report recommending a model (EXPERIMENTS.md), README update, commit.

## Hierarchy / depth / RPW experiments (2026-10-20, latest - supersedes older "next" lists)
- Hierarchy six-hour clean-mixture run finished (37,172 updates): stable, improving, but NOT better than the flat warm-start / `mix_clean_s0` (word-hit 0.65 vs 0.71-0.74, bpb 3.5 vs 2.9). Not promoted to Discord. Report: `runs/hier_mix_6h_s0/REPORT.md` (ignored dir; local only).
- Short probes, all failed the cost/benefit bar: coarse expansion to 4 blocks (+31% time, no gain); +2 full-resolution blocks (`hier_fine_blocks`, +28% time, tie); flat + parallel coarse branch (`hybrid_par_coarse`, +3% time, 1 h on mix-clean-v1: CE gain from mixture adaptation, word-hit slightly lower, inconclusive for the branch).
- RPW-v1 (`denoiser="rpw"`, 2,968,576 params, 348 ms/update, 0.71 GB, fp32 faster than bf16): 1 h from scratch on mix-clean-v1 (`runs/rpw_s0`, step-based schedule, paused/resumed at 10 min via `--stop-after`). Stable; word-hit 0.48-0.54, valid UTF-8 0.94, still non-word babble. Workspace-off ablation: CE +0.17 on FineWiki, ~0 elsewhere; samples not better with the workspace. Bridge routing diagnostic (`experiments/rpw_attn_diag.py`): mix of local and global heads, no collapse. Open: one bounded local-path capacity probe (3 local blocks/side or width 192) - needs user go-ahead.
- Pending: finish/commit clean-retry write-up (`mix-clean-v1` files, `mix_eval.py` flags), push of main to origin (not requested), stale `HYBRID_DENOISER_PLAN.md`.

## RPW encoder local-attention experiment (2026-10-20)
`model.rpw_enc_attn=true`: encoder block 2 = conv (d=1,2) -> windowed local attention (32-byte windows, stride 16) -> FFN; 3,066,496 params (+97,920), +11-16% time. Warm start from rpw_s0 (not identity), `runs/rpw_encattn_s0`, 8,800 updates (~56 min, lr 1e-4 -> 1e-5, seed 2). Final (32 samples, 64 windows): word-hit 0.525/0.545/0.503/0.575 vs 0.478/0.516/0.476/0.544, bpb 4.65/4.03/4.12/4.17 vs 5.04/4.32/4.40/4.49, UTF-8 1.000 vs 0.938, CE 0.3115/0.3629/0.4543/0.3661 vs 0.3153/0.3645/0.4685/0.3695 (lower on all four). Confounded by ~1 h extra training + lr rewarm; spelling still babble. Plan: `runs/rpw_encattn_PLAN.md`. No control, no longer run, no Discord change.
