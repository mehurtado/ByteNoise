"""Per-source deterministic val CE (same procedure as mix_eval.source_ce) split into junk-free val windows vs junk-flagged val windows
(mix_filters.is_junk), so a cleaned-data model is not judged only on val windows that contain tables.
   python scripts/mix_ce_split.py --mix mix-clean-v1 --runs RUN [RUN...] --out FILE.json"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import data, train
import mix_filters as mf
from mix_eval import source_ce

ap = argparse.ArgumentParser(); ap.add_argument("--mix", default="mix-clean-v1"); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
mix = json.load(open(os.path.join(data.DATA_ROOT, a.mix, "mix.json"))); z = np.load(os.path.join(data.DATA_ROOT, a.mix, "windows.npz"))
val = data.load_split(a.mix, "val"); V = val.numpy()
sets = {}
for s in mix["sources"]:
    st = z[f"val_{s['name']}"]; j = mf.is_junk(V[st[:, None] + np.arange(256)])
    sets[s["name"]] = {"clean": st[~j], "junk": st[j]}
    print(s["name"], "val windows", len(st), "junk", int(j.sum()), flush=True)
R = {}
for run in a.runs:
    model, cfg = train.load_model_from_run(run, "ema", "cuda"); model.eval(); R[run] = {}
    for n, d in sets.items():
        R[run][n] = {k: (source_ce(model, cfg, val, v) if len(v) >= 16 else None) for k, v in d.items()}
        R[run][n]["counts"] = {k: int(len(v)) for k, v in d.items()}
        print(run, n, {k: (round(x["overall"]["ce"], 4), round(x["t750_1000"]["ce"], 4)) if x else None for k, x in R[run][n].items() if k != "counts"}, flush=True)
    del model
json.dump(R, open(a.out, "w"), indent=1)
