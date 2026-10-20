#!/bin/bash
# First hour (and, if the gate passes, the whole 6 h) of the hierarchy run on data mix-clean-v2 (mix-clean-v1 + bounded personal-writing source).
# Warm start runs/hier_cont_unexp_s0 (--init-from: keeps EMA, fresh optimizer). Intended 6 h LR schedule from the start: 200-update warmup
# to 1e-4, wall-clock cosine to 1e-5 at max_seconds=21600. batch 32x4=128, fp32, self_cond, logits head, doc_aligned, seed 0.
# A monitor (scripts/hier_snap.py) snapshots at 20/40/60 min then every 45 min, evaluates, applies stop rules (writes <run>/STOP).
# Launch (Windows-safe):  nohup bash scripts/run_hier_mix_6h.sh > runs/hier_mix_6h_driver.out 2>&1 &
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python; export PYTHONIOENCODING=utf-8
R=runs/hier_mix_6h_s0; S=${R}_snaps; LOG=runs/hier_mix_6h_driver.log; SRC=runs/hier_cont_unexp_s0/checkpoint.pt
free() { powershell -NoProfile -c "[math]::Round((Get-PSDrive C).Free/1GB,2)" | tr -d '\r'; }
say() { echo "[$(date +%H:%M:%S)] $* | free $(free) GB" | tee -a $LOG; }
SET="diffusion.param=logits diffusion.t_sampling=warp model.embed_norm=8 diffusion.ce_weight=0 diffusion.x0_rescale=true diffusion.self_cond=true train.batch_size=32 train.grad_accum_steps=4 train.learning_rate=0.0001 train.lr_min=0.00001 train.warmup_updates=200 train.eval_interval=1000 train.eval_batches=8 train.save_interval=200 train.doc_aligned=true"
mkdir -p $S
( while [ ! -f $R/status.json ]; do echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')" >> $R.gpu.txt; sleep 15; done ) &
POLL=$!
if [ ! -f $R/status.json ]; then
  $PY scripts/hier_snap.py --daemon --run $R --snaps $S > runs/hier_mix_6h_monitor.log 2>&1 &
  MON=$!
  for k in 0 1 2 3 4 5; do
    [ -f $R/status.json ] && break
    if [ -f $R/checkpoint.pt ]; then say "resume"; $PY train.py train --resume $R >> $R.log 2>&1
    else say "train"; $PY train.py train --preset hier --precision fp32 --data mix-clean-v2 --seed 0 --run-dir $R --init-from $SRC --max-seconds 21600 --max-updates 400000 --set $SET > $R.log 2>&1; fi
    sleep 20
  done
  say "training stopped: $(cat $R/status.json 2>/dev/null)"; sleep 120; kill $MON 2>/dev/null
fi
kill $POLL 2>/dev/null
# final evals: final checkpoint, best snapshot, start checkpoint (same fixed windows/noise/seeds as runs/mix_clean_s0)
E=$R/final_eval; mkdir -p $E
BEST=$(python -c "import json;b=json.load(open('$R/state.json'))['best'];print('$S/'+b if b else '')" 2>/dev/null)
for spec in "final:$R" "best:$BEST" "start:runs/hier_cont_unexp_s0"; do
  n=${spec%%:*}; d=${spec#*:}; [ -n "$d" ] && [ -f $d/checkpoint.pt ] || continue
  say "mix_eval $n"; $PY scripts/mix_eval.py --run $d --out $E/$n --mix mix-clean-v1 > $E/$n.mix.log 2>&1 || say "mix_eval $n FAILED"
  say "ladder_eval $n"; $PY scripts/ladder_eval.py --run $d --out-dir $E/$n > $E/$n.ladder.log 2>&1 || say "ladder_eval $n FAILED"
done
say "DONE"
