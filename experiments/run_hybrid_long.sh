#!/bin/bash
# Hybrid+logits, effective batch 128 (32 x 4 accum), lr 6e-4, natural-v2, 5 h wall-clock cosine. Resumes on crash.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
R=runs/diff_nat_v2_hybrid_bs128_s0
for i in 1 2 3 4 5 6; do
  [ -f $R/status.json ] && break
  if [ -f $R/checkpoint.pt ]; then
    $PY train.py train --resume $R >> $R.log 2>&1
  else
    $PY train.py train --preset hybrid --run-dir $R --data natural-v2 --seed 0 --max-seconds 18000 --max-updates 400000 --set diffusion.param=logits diffusion.t_sampling=warp diffusion.ce_schedule=alpha_bar model.embed_norm=8 diffusion.ce_weight=0 diffusion.x0_rescale=true train.batch_size=32 train.grad_accum_steps=4 train.learning_rate=0.0006 train.eval_interval=1000 train.eval_batches=8 train.save_interval=1000 >> $R.log 2>&1
  fi
  sleep 20
done
