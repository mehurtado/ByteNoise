#!/bin/bash
# Clean-data retry of the mixture pilot: same recipe as run_mix_pilot.sh (warm start runs/diff_nat_v2_hybrid_selfcond_warm_s0,
# self_cond, doc_aligned, 32x4, lr 1e-4, cosine tied to a 3000 s budget, seed 0) on data mix-clean-v1, run dir runs/mix_clean_s0.
# A watcher keeps copies of the checkpoint at updates 500/1000/2000/3000 in runs/mix_clean_traj/. Then mix_eval on the final model
# (--mix mix-clean-v1) and a samples-only text-health read on each kept intermediate checkpoint. Aborts if free disk < 3 GB.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python; LOG=runs/mix_clean_driver.log; export PYTHONIOENCODING=utf-8
SRC=${1:-runs/diff_nat_v2_hybrid_selfcond_warm_s0/checkpoint.pt}; SECS=${2:-3000}; R=runs/mix_clean_s0; T=runs/mix_clean_traj
free() { powershell -NoProfile -c "[math]::Round((Get-PSDrive C).Free/1GB,2)" | tr -d '\r'; }
say() { echo "[$(date +%H:%M:%S)] $* | free $(free) GB" | tee -a $LOG; }
chk() { f=$(free); if awk "BEGIN{exit !($f < 3.0)}"; then say "ABORT low disk"; exit 1; fi; }
SET="diffusion.param=logits diffusion.t_sampling=warp diffusion.ce_schedule=alpha_bar model.embed_norm=8 diffusion.ce_weight=0 diffusion.x0_rescale=true diffusion.self_cond=true train.batch_size=32 train.grad_accum_steps=4 train.learning_rate=0.0001 train.warmup_updates=50 train.eval_interval=250 train.eval_batches=8 train.save_interval=500 train.doc_aligned=true"
if [ ! -f $R/status.json ]; then
  $PY experiments/ckpt_watcher.py --run $R --dest $T > runs/mix_clean_watcher.log 2>&1 &
  W=$!
  for k in 0 1 2 3; do
    [ -f $R/status.json ] && break
    chk
    if [ -f $R/checkpoint.pt ]; then say "resume"; $PY train.py train --resume $R >> runs/mix_clean_s0.log 2>&1
    else say "train"; $PY train.py train --preset hybrid --init-from $SRC --run-dir $R --data mix-clean-v1 --seed 0 --max-seconds $SECS --max-updates 400000 --set $SET > runs/mix_clean_s0.log 2>&1; fi
  done
  sleep 20; kill $W 2>/dev/null
fi
[ -f $R/status.json ] || { say "FAILED"; exit 1; }
chk; say "mix_eval final"; $PY experiments/mix_eval.py --run $R --out $R --mix mix-clean-v1 > runs/mix_clean_after.log 2>&1 || { say "EVAL FAILED"; exit 1; }
for d in $T/u*; do [ -d $d ] || continue; chk; say "health $d"; $PY experiments/mix_eval.py --run $d --out $d --mix mix-clean-v1 --samples-only > $d/health.log 2>&1; done
say "DONE"
