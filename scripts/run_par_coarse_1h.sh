#!/bin/bash
# One-hour continuation of the flat warm-start + parallel coarse branch, from the end of the 10-minute smoke test (runs/par_coarse_s0, model+EMA, fresh optimizer).
# mix-clean-v1 (no personal writing), seed 2 (fresh data order), fp32, 32x4, warmup 50 -> 1e-4 -> wall-clock cosine to 1e-5 over 3600 s. Samples every ~15 min
# (sample_monitor, 16 samples, s=1.0 default sampler) in samples_log.md. Final: short paired comparison vs flat warm (32 samples, s=1.0, 64 CE windows/source). Sources are never written.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python; export PYTHONIOENCODING=utf-8
R=runs/par_coarse_1h_s0; SRC=runs/par_coarse_s0; LOG=runs/par_coarse_1h_driver.log
say() { echo "[$(date +%H:%M:%S)] $* | free $(powershell -NoProfile -c "[math]::Round((Get-PSDrive C).Free/1GB,1)" | tr -d '\r') GB" | tee -a $LOG; }
SET="model.embed_norm=8 diffusion.param=logits diffusion.t_sampling=warp diffusion.ce_schedule=alpha_bar diffusion.ce_weight=0 diffusion.x0_rescale=true diffusion.self_cond=true model.hybrid_par_coarse=1 model.hybrid_par_after=3 train.batch_size=32 train.grad_accum_steps=4 train.learning_rate=0.0001 train.lr_min=0.00001 train.warmup_updates=50 train.eval_interval=1000 train.eval_batches=8 train.save_interval=200 train.doc_aligned=true"
if [ -z "$ATTACH" ] && [ ! -f $R/status.json ]; then
  say "train"
  if [ -f $R/checkpoint.pt ]; then CMD="--resume $R"; else CMD="--preset hybrid --precision fp32 --data mix-clean-v1 --max-updates 400000 --max-seconds 3600 --seed 2 --run-dir $R --init-from $SRC/checkpoint.pt --set $SET"; fi
  $PY scripts/sample_monitor.py --run $R --data mix-clean-v1 --n 16 --interval 900 --seed 0 > $R.monitor.log 2>&1 &
  MON=$!
  $PY train.py train $CMD > $R.log 2>&1
  sleep 60; kill $MON 2>/dev/null
fi
# short final comparison only (user request): 32 samples at s=1.0 (1000 steps, paired seeds), CE on 64 fixed windows per source; no ladder_eval, no s=0.8
until [ -f $R/status.json ]; do sleep 30; done
sleep 60; for p in $(powershell -NoProfile -c "Get-CimInstance Win32_Process | ? { \$_.CommandLine -match 'sample_monitor.*par_coarse_1h' } | % { \$_.ProcessId }" | tr -d ''); do taskkill //F //PID $p > /dev/null 2>&1; done
E=$R/final_eval; mkdir -p $E
for spec in "$R:$E/new" "runs/diff_nat_v2_hybrid_selfcond_warm_s0:$E/flat_warm"; do
  run=${spec%%:*}; out=${spec#*:}; mkdir -p $out
  say "short eval $run"; $PY scripts/mix_eval.py --run $run --out $out --mix mix-clean-v1 --n 32 --scales 1.0 --ce-windows 64 > $out.log 2>&1 || say "short eval $run FAILED"
done
say "DONE"
