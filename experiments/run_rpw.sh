#!/bin/bash
# RPW-v1 tiny byte-diffusion candidate, from scratch on mix-clean-v1 (no personal writing). FP32 (bench: bf16 slower on this model), 32x4, seed 1.
# ONE step-based schedule over the whole hour: 100-update warmup to 4e-4, cosine to 1e-5 at update 10000 (bench 343 ms/update => ~57 min; wall time may differ).
# stage 1: train to update 1700 (~10 min), pause (checkpoint holds model, EMA, optimizer, scheduler, scaler, RNG, data position), 16 samples at s=1.0.
# stage 2 (run after review): train.py train --resume runs/rpw_s0   -> same optimizer/schedule, no warmup restart, continues to update 10000;
#          then a short eval (32 samples s=1.0, 64 CE windows per source).
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python; export PYTHONIOENCODING=utf-8
R=runs/rpw_s0; LOG=runs/rpw_driver.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
SET="model.embed_norm=8 diffusion.param=logits diffusion.t_sampling=warp diffusion.ce_weight=0 diffusion.x0_rescale=true diffusion.self_cond=true train.batch_size=32 train.grad_accum_steps=4 train.learning_rate=0.0004 train.lr_min=0.00001 train.warmup_updates=100 train.eval_interval=500 train.eval_batches=8 train.save_interval=200 train.doc_aligned=true"
if [ "$1" = "stage2" ]; then
  say "stage2 resume"; $PY train.py train --resume $R >> $R.log 2>&1
  mkdir -p $R/final_eval; say "short eval"
  $PY experiments/mix_eval.py --run $R --out $R/final_eval --mix mix-clean-v1 --n 32 --scales 1.0 --ce-windows 64 > $R/final_eval.log 2>&1 || say "eval FAILED"
  say "DONE stage2"; exit 0
fi
say "stage1 train"
$PY train.py train --preset rpw --precision fp32 --data mix-clean-v1 --max-updates 10000 --seed 1 --run-dir $R --stop-after 1700 --set $SET > $R.log 2>&1
mkdir -p $R/eval10
say "10-min samples"; $PY experiments/mix_eval.py --run $R --out $R/eval10 --mix mix-clean-v1 --n 32 --keep 16 --scales 1.0 --samples-only > $R/eval10.log 2>&1 || say "sample eval FAILED"
say "DONE stage1"
