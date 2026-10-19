#!/bin/bash
# Rung runs (width 96, natural-v1, L2 recipe). Sequential and re-runnable: finished steps are skipped, crashed runs resume.
#   scripts/run_rungs.sh screen                 seed-0 screening runs
#   scripts/run_rungs.sh confirm R4 span30 ...  seed-1 runs of the named rungs (checkpoints of seed 1 are deleted after eval)
#   scripts/run_rungs.sh one bs128_25k          a single rung by name (seed 0)
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
LOG=runs/rungs_driver.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

SET="diffusion.param=logits diffusion.t_sampling=warp model.embed_norm=8 diffusion.ce_weight=0 diffusion.x0_rescale=true train.batch_size=32 train.grad_accum_steps=1 train.learning_rate=0.0003 train.eval_interval=2000 train.save_interval=5000 model.model_dim=96"
declare -A EXTRA UPD
# bs128: lr scaled 2x (6e-4, not sqrt-4x), 6250 updates = 800k examples = the baseline's 25000 x 32; eval every 500 updates = every 64k examples; 8 x 128 = the same 1024 val windows
EXTRA[bs128]="train.batch_size=128 train.learning_rate=0.0006 train.eval_interval=500 train.eval_batches=8 train.save_interval=2000"; UPD[bs128]=6250
# equal-updates companion (25000 updates at bs128 = 4x the examples), only if time permits
EXTRA[bs128_25k]="train.batch_size=128 train.learning_rate=0.0006 train.eval_interval=2000 train.eval_batches=8"; UPD[bs128_25k]=25000
EXTRA[R4]="model.embed_norm=4"
EXTRA[R6]="model.embed_norm=6"
EXTRA[tied_R8]="model.tied_head=true"
EXTRA[tied_R4]="model.tied_head=true model.embed_norm=4"
EXTRA[tied_R6]="model.tied_head=true model.embed_norm=6"
EXTRA[span30]="diffusion.span_noise_prob=0.3"
EXTRA[span60]="diffusion.span_noise_prob=0.6"

run_one() { # rung seed
  rung=$1; seed=$2; suf=""; [ "$seed" != "0" ] && suf="_s$seed"
  name=rung_${rung}${suf}; d=runs/$name; upd=${UPD[$rung]:-25000}
  if [ ! -f $d/ladder_eval.json ]; then
    if [ ! -f $d/status.json ]; then
      if [ -d $d ] && [ -f $d/checkpoint.pt ]; then
        say "resume $name"; $PY train.py train --resume $d >> runs/$name.log 2>&1
      else
        say "train $name"; $PY train.py train --preset compact --precision fp32 --data natural-v1 --max-updates $upd --seed $seed --run-dir $d --set $SET ${EXTRA[$rung]} > runs/$name.log 2>&1
      fi
      for k in 1 2; do
        [ -f $d/status.json ] && break
        say "$name incomplete, resuming (try $k)"; $PY train.py train --resume $d >> runs/$name.log 2>&1
      done
    fi
    [ -f $d/status.json ] || { say "$name FAILED to finish"; return 1; }
    say "eval $name"
    $PY scripts/ladder_eval.py --run $d > runs/${name}_eval.log 2>&1 || { say "eval $name FAILED"; return 1; }
  fi
  if [ "$seed" != "0" ] && [ -f $d/checkpoint.pt ] && [ -f $d/ladder_eval.json ]; then rm $d/checkpoint.pt; say "deleted $name checkpoint"; fi
  say "done $name"
}

best_radius() { # the better of R4/R6 by deterministic val CE (overall)
  $PY -c "
import json
c = {r: json.load(open(f'runs/rung_R{r}/ladder_eval.json'))['val_ce']['overall']['ce'] for r in (4, 6)}
print(min(c, key=c.get))" 2>/dev/null
}

case "$1" in
  screen)
    for r in bs128 R4 R6 tied_R8; do run_one $r 0; done
    br=$(best_radius); [ -z "$br" ] && br=4
    say "best untied radius by val CE: R$br -> tied_R$br"
    run_one tied_R$br 0
    for r in span30 span60; do run_one $r 0; done
    say "SCREEN DONE" ;;
  confirm) shift; for r in "$@"; do run_one $r 1; done; say "CONFIRM DONE" ;;
  one) run_one $2 0 ;;
  *) echo "usage: $0 screen | confirm <rung>... | one <rung>"; exit 1 ;;
esac
