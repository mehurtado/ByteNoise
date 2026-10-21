#!/bin/bash
# Rollout-state training experiment. Plan: runs/rpw_roll_PLAN.md. usage: run_rpw_roll.sh stage1|stage2|ctrl
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python; export PYTHONIOENCODING=utf-8
LOG=runs/rpw_roll_driver.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
BASE="model.embed_norm=8 model.rpw_enc_attn=true diffusion.param=logits diffusion.t_sampling=warp diffusion.ce_weight=0 diffusion.x0_rescale=true diffusion.self_cond=true train.batch_size=32 train.grad_accum_steps=4 train.learning_rate=0.0001 train.lr_min=0.00001 train.warmup_updates=100 train.eval_interval=500 train.eval_batches=8 train.save_interval=200 train.doc_aligned=true"
ROLL="diffusion.rollout_prob=0.25 diffusion.rollout_steps=1 diffusion.rollout_jump=100"
EV="--mix mix-clean-v1 --n 32 --scales 1.0 --ce-windows 64"
if [ "$1" = "stage2" ]; then
  R=runs/rpw_roll_s0; say "A stage2 resume"; $PY train.py train --resume $R >> $R.log 2>&1
  mkdir -p $R/final_eval; say "A short eval"; $PY scripts/mix_eval.py --run $R --out $R/final_eval $EV > $R/final_eval.log 2>&1 || say "A eval FAILED"
  say "DONE stage2"; exit 0
fi
if [ "$1" = "ctrl" ]; then
  R=runs/rpw_roll_ctrl_s0; say "B control train"
  $PY train.py train --preset rpw --precision fp32 --data mix-clean-v1 --max-updates 8800 --seed 3 --run-dir $R --init-from runs/rpw_encattn_s0/checkpoint.pt --set $BASE > $R.log 2>&1
  mkdir -p $R/final_eval; say "B short eval"; $PY scripts/mix_eval.py --run $R --out $R/final_eval $EV > $R/final_eval.log 2>&1 || say "B eval FAILED"
  say "DONE ctrl"; exit 0
fi
R=runs/rpw_roll_s0; say "A stage1 train"
$PY train.py train --preset rpw --precision fp32 --data mix-clean-v1 --max-updates 8800 --seed 3 --run-dir $R --init-from runs/rpw_encattn_s0/checkpoint.pt --stop-after 1450 --set $BASE $ROLL > $R.log 2>&1
mkdir -p $R/eval10 runs/rpw_encattn_s0/eval16_paired
say "paired samples"
$PY scripts/mix_eval.py --run $R --out $R/eval10 --mix mix-clean-v1 --n 32 --keep 16 --scales 1.0 --samples-only > $R/eval10.log 2>&1 || say "A sample eval FAILED"
$PY scripts/mix_eval.py --run runs/rpw_encattn_s0 --out runs/rpw_encattn_s0/eval16_paired --mix mix-clean-v1 --n 32 --keep 16 --scales 1.0 --samples-only > runs/rpw_encattn_s0/eval16_paired.log 2>&1 || say "baseline sample eval FAILED"
say "DONE stage1"
