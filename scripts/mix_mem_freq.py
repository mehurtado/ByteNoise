"""For the non-junk 32-byte sample windows (stride 16, as mix_eval) found verbatim in the mixture train text: how often each occurs in
the train text (1 / 2-10 / >10 = formulaic phrase, not recall) and in which source's train text the first occurrence lies.
   python scripts/mix_mem_freq.py --runs runs/A runs/B --out FILE.json     (reads <run>/mix_samples.json; CPU)"""
import argparse, collections, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import data
import mix_filters as mf

ap = argparse.ArgumentParser(); ap.add_argument("--mix", default="mix-pilot-v1"); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
mix = json.load(open(os.path.join(data.DATA_ROOT, a.mix, "mix.json")))
mem = data.load_split(a.mix, "train").numpy().tobytes()
bounds = [(s["name"], s["ranges"]["train"][1]) for s in mix["sources"]]
cache, R = {}, {}


def info(w):
    if w not in cache:
        pos = mem.find(w); src = next(n for n, hi in bounds if pos < hi)
        c, p = 0, -1
        while c < 1000:  # cap
            p = mem.find(w, p + 1)
            if p < 0: break
            c += 1
        cache[w] = (c, src)
    return cache[w]


for run in a.runs:
    S = json.load(open(os.path.join(run, "mix_samples.json")))
    for k, hx in S.items():
        samples = [bytes.fromhex(h) for h in hx]
        ws = [bytes(s[i:i + 32]) for s in samples for i in range(0, len(s) - 31, 16)][:2000]
        jm = mf.is_junk(np.stack([np.frombuffer(w, np.uint8) for w in ws]))
        hits = [w for w, j in zip(ws, jm) if (not j) and w in mem]
        b = collections.Counter(); srcs = collections.Counter(); ex = collections.Counter()
        for w in hits:
            c, src = info(w); b["1" if c == 1 else "2-10" if c <= 10 else ">10"] += 1; srcs[src] += 1
            if c > 10: ex[ascii(w.decode("utf-8", "replace"))] += 1
        R[f"{os.path.basename(run.rstrip('/'))}/{k}"] = dict(n_windows=len(ws), clean_hits=len(hits), freq_buckets=dict(b), first_hit_source=dict(srcs),
                                                           examples_gt10=ex.most_common(5), examples_unique=[ascii(w.decode("utf-8", "replace")) for w in hits if info(w)[0] == 1][:5])
        print(run, k, R[f"{os.path.basename(run.rstrip('/'))}/{k}"], flush=True)
json.dump(R, open(a.out, "w"), indent=1)
