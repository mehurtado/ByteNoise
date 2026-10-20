#!/bin/bash
# Coarse-depth expansion smoke test: 10-minute continuations of runs/hier_10min_s0 on natural-v1, one after the other.
# A = unexpanded hierarchy (2 coarse blocks), B = 4 coarse blocks (2 new identity-init blocks). Identical settings and lr schedule:
# warmup 100 updates to 1e-4, cosine on wall-clock to 1e-5 over 600 s. seed 1 (fresh data order; seed 0 would replay the first
# 3,754 batches of the parent run) for both.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
SECS=${1:-600}
COMMON="diffusion.param=logits diffusion.t_sampling=warp model.embed_norm=8 diffusion.ce_weight=0 diffusion.x0_rescale=true train.batch_size=32 train.grad_accum_steps=1 train.learning_rate=0.0001 train.lr_min=0.00001 train.warmup_updates=100 train.eval_interval=500 train.save_interval=500 diffusion.self_cond=true"
for spec in "hier_cont_unexp_s0:" "hier_cont_coarse4_s0:model.hier_core_blocks=4"; do
  R=runs/${spec%%:*}; EXTRA=${spec#*:}
  while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $R.gpu.txt; sleep 3; done &
  POLL=$!
  $PY train.py train --preset hier --precision fp32 --data natural-v1 --max-updates 25000 --max-seconds $SECS --seed 1 --run-dir $R \
    --init-from runs/hier_10min_s0/checkpoint.pt --set $COMMON $EXTRA > $R.log 2>&1
  kill $POLL
done
