#!/bin/bash
# Part 1 re-score of existing runs, then Part 2 fixed-update ladder (sequential). Re-runnable: finished steps are skipped.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python
LOG=runs/ladder_driver.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }

for r in diff_nat_logits_s0 diff_nat_logits_warp_s0 diff_nat_logits_emb_s0 diff_nat_logits_wide192_s0 diff_nat_v2_long_s0 diff_nat_gate_s0 diff_nat_hybrid_s0 diff_nat_hybrid_selfcond_s0; do
  o=runs/_ladder_rescore/$r
  [ -f $o/ladder_eval.json ] && continue
  say "rescore $r"
  $PY experiments/ladder_eval.py --run runs/$r --out-dir $o > runs/_ladder_rescore_$r.log 2>&1 || say "rescore $r FAILED (see runs/_ladder_rescore_$r.log)"
done

SET="diffusion.param=logits diffusion.t_sampling=warp model.embed_norm=8 diffusion.ce_weight=0 diffusion.x0_rescale=true train.batch_size=32 train.grad_accum_steps=1 train.learning_rate=0.0003 train.eval_interval=2000 train.save_interval=5000"
run_one() { # name preset seed extra-set
  name=$1; preset=$2; seed=$3; extra=$4; d=runs/$name
  if [ ! -f $d/ladder_eval.json ]; then
    if [ ! -f $d/status.json ]; then
      if [ -d $d ] && [ -f $d/checkpoint.pt ]; then
        say "resume $name"; $PY train.py train --resume $d >> runs/$name.log 2>&1
      else
        say "train $name"; $PY train.py train --preset $preset --precision fp32 --data natural-v1 --max-updates 25000 --seed $seed --run-dir $d --set $SET $extra > runs/$name.log 2>&1
      fi
      for k in 1 2; do
        [ -f $d/status.json ] && break
        say "$name incomplete, resuming (try $k)"; $PY train.py train --resume $d >> runs/$name.log 2>&1
      done
    fi
    [ -f $d/status.json ] || { say "$name FAILED to finish"; return 1; }
    say "eval $name"
    $PY experiments/ladder_eval.py --run $d > runs/${name}_eval.log 2>&1 || { say "eval $name FAILED"; return 1; }
  fi
  if [ "$seed" != "0" ] && [ -f $d/checkpoint.pt ] && [ -f $d/ladder_eval.json ]; then rm $d/checkpoint.pt; say "deleted $name checkpoint"; fi
  say "done $name"
}
for w in 48 96 192; do for s in 0 1; do run_one ladder_w${w}_s$s compact $s "model.model_dim=$w"; done; done
for s in 0 1; do run_one ladder_hyb_s$s hybrid $s ""; done
say "ALL DONE"
