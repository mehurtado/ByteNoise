#!/bin/bash
# Stage B/C/D of the mixture pilot (run after run_mix_stageA.sh finished and the start checkpoint was chosen).
#   experiments/run_mix_pilot.sh <start checkpoint.pt> [seconds=3000]
# 1 mix_eval of the start model (BEFORE; also the GPU smoke test of mix_eval), 2 200-update stability check at lr 1e-4,
# 3 pilot runs/mix_pilot_s0 (init-from start, mix-pilot-v1, self_cond, 32x4, lr 1e-4, cosine tied to the time budget,
# sample_monitor beside it, resume on crash), 4 mix_eval AFTER. Every step logs free disk and aborts if < 1.5 GB.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python; LOG=runs/mix_pilot_driver.log
SRC=$1; SECS=${2:-3000}; R=runs/mix_pilot_s0
free() { powershell -NoProfile -c "[math]::Round((Get-PSDrive C).Free/1GB,2)" | tr -d '\r'; }
say() { echo "[$(date +%H:%M:%S)] $* | free $(free) GB" | tee -a $LOG; }
chk() { f=$(free); if awk "BEGIN{exit !($f < 1.5)}"; then say "ABORT low disk"; exit 1; fi; }
SET="diffusion.param=logits diffusion.t_sampling=warp diffusion.ce_schedule=alpha_bar model.embed_norm=8 diffusion.ce_weight=0 diffusion.x0_rescale=true diffusion.self_cond=true train.batch_size=32 train.grad_accum_steps=4 train.learning_rate=0.0001 train.warmup_updates=50 train.eval_interval=250 train.eval_batches=8 train.save_interval=500 train.doc_aligned=true"
SRCRUN=$(dirname $SRC)
chk; [ -f runs/mix_pilot_before/mix_eval.json ] || { say "mix_eval BEFORE"; $PY experiments/mix_eval.py --run $SRCRUN --out runs/mix_pilot_before > runs/mix_pilot_before.log 2>&1 || { say "BEFORE FAILED"; exit 1; }; }
chk; if [ ! -f runs/_mix_stab/status.json ]; then say "stability 200 updates"; rm -rf runs/_mix_stab
  $PY train.py train --preset hybrid --init-from $SRC --run-dir runs/_mix_stab --data mix-pilot-v1 --seed 0 --max-updates 200 --set $SET train.eval_interval=100 train.save_interval=1000 > runs/_mix_stab.log 2>&1; fi
tail -4 runs/_mix_stab.log | tee -a $LOG
if [ ! -f $R/status.json ]; then
  chk; $PY experiments/sample_monitor.py --run $R --data mix-pilot-v1 --interval 900 > runs/mix_pilot_s0_monitor.log 2>&1 &
  MON=$!
  for k in 0 1 2 3; do
    [ -f $R/status.json ] && break
    chk
    if [ -f $R/checkpoint.pt ]; then say "resume pilot"; $PY train.py train --resume $R >> runs/mix_pilot_s0.log 2>&1
    else say "train pilot"; $PY train.py train --preset hybrid --init-from $SRC --run-dir $R --data mix-pilot-v1 --seed 0 --max-seconds $SECS --max-updates 400000 --set $SET > runs/mix_pilot_s0.log 2>&1; fi
  done
  kill $MON 2>/dev/null
fi
[ -f $R/status.json ] || { say "pilot FAILED"; exit 1; }
chk; say "mix_eval AFTER"; $PY experiments/mix_eval.py --run $R --out $R > runs/mix_pilot_after.log 2>&1 || { say "AFTER FAILED"; exit 1; }
say "PILOT DONE"
