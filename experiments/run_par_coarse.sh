#!/bin/bash
# Bounded smoke test: best flat warm-start (runs/diff_nat_v2_hybrid_selfcond_warm_s0, model+EMA) + one parallel coarse branch after flat block 3
# (zero-init gate). 10 min on mix-clean-v1 (no personal writing), seed 1, fp32, 32x4, warmup 50 -> 1e-4 -> wall-clock cosine to 1e-5 (as the hier smoke tests).
# Short eval: 16 samples at s=1.0 / 1000 steps / same generation seeds for the new checkpoint and the original warm-start, text health, word-hit,
# CE on 64 fixed val windows per source. Source run is read only; outputs go to new dirs. Usage: run_par_coarse.sh [seconds]
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python; export PYTHONIOENCODING=utf-8
SECS=${1:-600}
SRC=runs/diff_nat_v2_hybrid_selfcond_warm_s0
R=runs/par_coarse_s0; LOG=runs/par_coarse_driver.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
SET="model.embed_norm=8 diffusion.param=logits diffusion.t_sampling=warp diffusion.ce_schedule=alpha_bar diffusion.ce_weight=0 diffusion.x0_rescale=true diffusion.self_cond=true model.hybrid_par_coarse=1 model.hybrid_par_after=3 train.batch_size=32 train.grad_accum_steps=4 train.learning_rate=0.0001 train.lr_min=0.00001 train.warmup_updates=50 train.eval_interval=250 train.eval_batches=8 train.save_interval=200 train.doc_aligned=true"
[ -f $R/status.json ] || { say "train $R"
  while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $R.gpu.txt; sleep 3; done &
  POLL=$!
  $PY train.py train --preset hybrid --precision fp32 --data mix-clean-v1 --max-updates 25000 --max-seconds $SECS --seed 1 --run-dir $R \
    --init-from $SRC/checkpoint.pt --set $SET > $R.log 2>&1
  kill $POLL; }
for spec in "$R:$R/quick" "$SRC:runs/par_coarse_baseline_quick"; do
  run=${spec%%:*}; out=${spec#*:}; mkdir -p $out
  say "quick eval $run"; $PY experiments/mix_eval.py --run $run --out $out --mix mix-clean-v1 --n 32 --keep 16 --scales 1.0 --ce-windows 64 > $out.log 2>&1 || say "quick eval $run FAILED"
done
say "DONE par coarse"
