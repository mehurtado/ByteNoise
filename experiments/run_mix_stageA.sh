#!/bin/bash
# Stage A: wait until the warm-start jobs are gone and the GPU is idle (<15% twice, 2 min apart), then ladder_eval the warm run
# (if the warm agent has not already written ladder_eval.json) so it can be compared with runs/rung_hyb_selfcond.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python; LOG=runs/mix_pilot_driver.log
say() { echo "[$(date +%H:%M:%S)] $* | free $(powershell -NoProfile -c "[math]::Round((Get-PSDrive C).Free/1GB,2)" | tr -d '\r') GB" | tee -a $LOG; }
util() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' '; }
busy() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'selfcond_warm|plain_warm|warm_control|hybrid_selfcond_warm' -and \$_.CommandLine -notmatch 'Get-CimInstance' -and \$_.CommandLine -notmatch 'mix_pilot' }).Count" | tr -d '\r'; }
while true; do
  if [ "$(busy)" = "0" ]; then
    u1=$(util); sleep 120; u2=$(util)
    if [ "$u1" -lt 15 ] && [ "$u2" -lt 15 ] && [ "$(busy)" = "0" ]; then say "GPU free ($u1/$u2)"; break; fi
    say "GPU busy ($u1/$u2)"
  else say "warm jobs still running"; sleep 600; fi
done
W=runs/diff_nat_v2_hybrid_selfcond_warm_s0
if [ ! -f $W/ladder_eval.json ]; then say "ladder_eval warm"; $PY experiments/ladder_eval.py --run $W > runs/mix_pilot_warm_eval.log 2>&1; fi
say "STAGE A DONE"
