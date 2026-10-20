#!/bin/bash
# PREPARED, NOT RUN. Warm-start a self-conditioned hybrid+logits model from the 5 h bs128 checkpoint (new x0_prev input columns
# zero-initialised, so step 0 reproduces the source model when x0_prev is zero; tests/test_warmstart.py). Fresh optimizer/schedule.
# Same recipe as scripts/run_hybrid_long.sh plus diffusion.self_cond=true; shorter budget (default 1 h) since it starts trained.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
SRC=runs/diff_nat_v2_hybrid_bs128_s0/checkpoint.pt
R=runs/diff_nat_v2_hybrid_selfcond_warm_s0
SECS=${1:-3600}
for i in 1 2 3; do
  [ -f $R/status.json ] && break
  if [ -f $R/checkpoint.pt ]; then
    $PY train.py train --resume $R >> $R.log 2>&1
  else
    $PY train.py train --preset hybrid --init-from $SRC --run-dir $R --data natural-v2 --seed 0 --max-seconds $SECS --max-updates 400000 --set diffusion.param=logits diffusion.t_sampling=warp diffusion.ce_schedule=alpha_bar model.embed_norm=8 diffusion.ce_weight=0 diffusion.x0_rescale=true diffusion.self_cond=true train.batch_size=32 train.grad_accum_steps=4 train.learning_rate=0.0003 train.eval_interval=1000 train.eval_batches=8 train.save_interval=1000 >> $R.log 2>&1
  fi
  sleep 20
done
