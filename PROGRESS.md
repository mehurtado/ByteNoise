# Progress / handoff (read this first when resuming)

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
