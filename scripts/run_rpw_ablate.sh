#!/bin/bash
# Workspace-off check on the final rpw checkpoint (waits for stage 2 + its short eval): identical short eval with the write-back dropped
# (32 samples s=1.0 same seeds, CE on the same 64 windows/source, same noise and timesteps). Compare with runs/rpw_s0/final_eval.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -q "DONE stage2" runs/rpw_driver.log 2>/dev/null; do sleep 15; done
R=runs/rpw_s0; mkdir -p $R/final_eval_nowrite
.venv/Scripts/python scripts/mix_eval.py --run $R --out $R/final_eval_nowrite --mix mix-clean-v1 --n 32 --scales 1.0 --ce-windows 64 --no-write > $R/final_eval_nowrite.log 2>&1
echo "[$(date +%H:%M:%S)] DONE ablation" >> runs/rpw_driver.log
