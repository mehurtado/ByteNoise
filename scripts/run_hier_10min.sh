#!/bin/bash
# Hierarchical (trimmed hybrid) 10-minute from-scratch test. Settings identical to runs/rung_hyb_selfcond (natural-v1, batch 32 x 1,
# lr 3e-4, L2 recipe + self_cond, fp32, EMA 0.999) except the denoiser and a 600 s wall-clock budget (cosine follows time).
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
R=runs/hier_10min_s0
SECS=${1:-600}
$PY train.py train --preset hier --precision fp32 --data natural-v1 --max-updates 25000 --max-seconds $SECS --seed 0 --run-dir $R --set diffusion.param=logits diffusion.t_sampling=warp model.embed_norm=8 diffusion.ce_weight=0 diffusion.x0_rescale=true train.batch_size=32 train.grad_accum_steps=1 train.learning_rate=0.0003 train.eval_interval=1000 train.save_interval=500 diffusion.self_cond=true > $R.log 2>&1
