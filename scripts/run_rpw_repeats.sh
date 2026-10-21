#!/bin/bash
# Shared local refinement, ONE matched pair (control repeats=1, variant repeats=2), sequential. Both warm-start from runs/rpw_encattn_s0 (untouched).
# Same data, seed, optimizer, lr schedule, batch 32x4, fp32, 2400 updates. Commit code only, not runs/.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python; export PYTHONIOENCODING=utf-8
LOG=runs/rpw_repeats_driver.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
BASE="model.embed_norm=8 model.rpw_enc_attn=true diffusion.param=logits diffusion.t_sampling=warp diffusion.ce_weight=0 diffusion.x0_rescale=true diffusion.self_cond=true train.batch_size=32 train.grad_accum_steps=4 train.learning_rate=0.0001 train.lr_min=0.00001 train.warmup_updates=100 train.eval_interval=500 train.eval_batches=8 train.save_interval=200 train.doc_aligned=true"
for REP in 1 2; do
  R=runs/rpw_rep${REP}_s0; say "train rep$REP"
  $PY train.py train --preset rpw --precision fp32 --data mix-clean-v1 --max-updates 2400 --seed 5 --run-dir $R --init-from runs/rpw_encattn_s0/checkpoint.pt --set $BASE model.rpw_local_repeats=$REP > $R.log 2>&1 || { say "rep$REP train FAILED"; exit 1; }
  mkdir -p $R/eval; say "eval rep$REP"
  $PY scripts/mix_eval.py --run $R --out $R/eval --mix mix-clean-v1 --n 64 --keep 64 --scales 1.0 --ce-windows 64 > $R/eval.log 2>&1 || say "eval rep$REP FAILED"
done
say "analyze"
$PY scripts/rpw_repeats_analyze.py --a runs/rpw_rep1_s0 --b runs/rpw_rep2_s0 --out runs/rpw_rep2_s0/paired.json > runs/rpw_rep2_s0/paired.log 2>&1 || say "analyze FAILED"
say "DONE repeats"
