#!/bin/bash
# Fine-depth smoke test from the finished hierarchy six-hour checkpoint (preserved copy runs/hier_mix_6h_final_ckpt; the source is never written).
# Leg A = unchanged checkpoint continued, leg B = +1 full-resolution block at the encoder entrance and +1 at the decoder exit (identity-init,
# two coarse blocks kept). 10 min each on mix-clean-v1, identical settings/lr schedule/seed. Quick read of both legs afterwards:
# 32 samples at s=1.0 (1000 steps, same generation seeds) + text health + CE on 128 fixed val windows per source. Full evals are reserved for a one-hour candidate.
# Usage: run_hier_fine.sh [seconds]   (a leg whose status.json exists is not retrained)
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python; export PYTHONIOENCODING=utf-8
SECS=${1:-600}
SRC=runs/hier_mix_6h_final_ckpt
LOG=runs/hier_fine_driver.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
COMMON="diffusion.param=logits diffusion.t_sampling=warp model.embed_norm=8 diffusion.ce_weight=0 diffusion.x0_rescale=true diffusion.self_cond=true train.batch_size=32 train.grad_accum_steps=4 train.learning_rate=0.0001 train.lr_min=0.00001 train.warmup_updates=50 train.eval_interval=250 train.eval_batches=8 train.save_interval=200 train.doc_aligned=true"
for spec in "hier_fine_unexp_s0:" "hier_fine1_s0:model.hier_fine_blocks=1"; do
  R=runs/${spec%%:*}; EXTRA=${spec#*:}
  [ -f $R/status.json ] || { say "train $R"
  while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $R.gpu.txt; sleep 3; done &
  POLL=$!
  $PY train.py train --preset hier --precision fp32 --data mix-clean-v1 --max-updates 25000 --max-seconds $SECS --seed 1 --run-dir $R \
    --init-from $SRC/checkpoint.pt --set $COMMON $EXTRA > $R.log 2>&1
  kill $POLL; }
done
for n in hier_fine_unexp_s0 hier_fine1_s0; do
  R=runs/$n; mkdir -p $R/quick
  say "quick eval $n"; $PY scripts/mix_eval.py --run $R --out $R/quick --mix mix-clean-v1 --n 32 --scales 1.0 --ce-windows 128 > $R/quick/mix.log 2>&1 || say "quick eval $n FAILED"
done
say "DONE fine test"
