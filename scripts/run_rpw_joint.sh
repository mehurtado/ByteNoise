#!/bin/bash
# Joint four-byte head, 10-minute test. Plan: runs/rpw_joint_PLAN.md. Waits for the rollout chain (DONE stage2) before touching the GPU.
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python; export PYTHONIOENCODING=utf-8
LOG=runs/rpw_joint_driver.log; R=runs/rpw_joint_s0
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
while ! grep -q "DONE stage2" runs/rpw_roll_driver.log 2>/dev/null; do sleep 30; done
say "bench"
$PY - > runs/rpw_joint_bench.log 2>&1 <<'PYEOF'
import sys; sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import bench_rpw_encattn as b
from dataclasses import replace
orig=b.rpw_cfg
def mk(joint):
    def f(attn):
        c=orig(True); return replace(c,model=replace(c.model,rpw_joint_head=joint))
    return f
res={False:[],True:[]}
for j in (False,True,False,True):
    b.rpw_cfg=mk(j); res[j].append(b.bench(True))
for j,r in res.items(): print("joint" if j else "source",[round(x["ms"],1) for x in r],"peak_gb",round(max(x["peak_gb"] for x in r),2),"finite",all(x["finite"] for x in r),"params",r[0]["params"])
PYEOF
say "train"
SET="model.embed_norm=8 model.rpw_enc_attn=true model.rpw_joint_head=true diffusion.param=logits diffusion.t_sampling=warp diffusion.ce_weight=0 diffusion.x0_rescale=true diffusion.self_cond=true train.batch_size=32 train.grad_accum_steps=4 train.learning_rate=0.0001 train.lr_min=0.00001 train.warmup_updates=100 train.eval_interval=500 train.eval_batches=8 train.save_interval=200 train.doc_aligned=true"
$PY train.py train --preset rpw --precision fp32 --data mix-clean-v1 --max-updates 2400 --seed 4 --run-dir $R --init-from runs/rpw_encattn_s0/checkpoint.pt --set $SET > $R.log 2>&1
for M in joint marginal; do
  mkdir -p $R/eval_$M; say "eval $M"
  $PY scripts/mix_eval.py --run $R --out $R/eval_$M --mix mix-clean-v1 --n 32 --keep 16 --scales 1.0 --ce-windows 64 --head-mode $M > $R/eval_$M.log 2>&1 || say "eval $M FAILED"
done
say "DONE joint"
