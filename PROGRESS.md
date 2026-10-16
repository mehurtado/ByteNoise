# Progress / handoff (read this first when resuming)

Goal: most capable model on RTX 4060 Ti 8GB (see user's 10-step plan: GPU env -> reproducible runs -> trustworthy training/eval -> data+eval suite -> Transformer baseline -> parallel recurrent model -> shared loops -> bounded diffusion -> staged selection -> deliver report). Python: `.venv\Scripts\python`. Use bf16 (fp32 spills VRAM and is ~23x slower). Subagents: sonnet/haiku only, never opus. No Co-Authored-By lines in commits.

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
