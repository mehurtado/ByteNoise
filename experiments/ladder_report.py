"""Aggregate ladder_eval.json files: Part-1 re-score table + rank agreement, Part-2 per-run trajectory, per-config
mean/spread over seeds, paired bootstrap comparisons. Prints markdown and writes runs/ladder_summary.md/.json."""
import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def load(p):
    return json.load(open(p))


def g(R, s, key):
    b = R["samplers"].get(s)
    return None if b is None else b[key]


def ce(R, k):
    return None if not R.get("val_ce") else R["val_ce"][k]["ce"]


def f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def part1(out):
    names = ["diff_nat_logits_s0", "diff_nat_logits_warp_s0", "diff_nat_logits_emb_s0", "diff_nat_logits_wide192_s0",
             "diff_nat_v2_long_s0", "diff_nat_gate_s0", "diff_nat_hybrid_s0", "diff_nat_hybrid_selfcond_s0"]
    label = dict(diff_nat_logits_s0="L1", diff_nat_logits_warp_s0="L2", diff_nat_logits_emb_s0="L3 (R=16)",
                 diff_nat_logits_wide192_s0="wide192 (L2, 45 min)", diff_nat_v2_long_s0="v2 long (4 h)",
                 diff_nat_gate_s0="eps gate", diff_nat_hybrid_s0="eps hybrid", diff_nat_hybrid_selfcond_s0="eps hybrid+selfcond")
    rows, X = [], {}
    for n in names:
        p = f"runs/_ladder_rescore/{n}/ladder_eval.json"
        if os.path.exists(p):
            X[n] = load(p)
    L = ["## Part 1: re-score of existing runs (128 paired samples, EMA, natural-v1 word list/scorer)", "",
         "| run | params | updates | word_hit s=1.0 [CI] | n-gram bpb s=1.0 [CI] | utf8 | d4 | word_hit s=0.8 | bpb s=0.8 | val CE all | CE t>=750 | CE t500-750 | s/sample |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, R in X.items():
        b = R["samplers"]["s1.0"]
        L.append(f"| {label[n]} | {R['params']/1e6:.1f}M | {R['train'].get('updates')} | {b['word_hit']:.3f} [{b['word_hit_ci'][0]:.3f},{b['word_hit_ci'][1]:.3f}] | "
                 f"{b['ngram_bpb']:.2f} [{b['ngram_bpb_ci'][0]:.2f},{b['ngram_bpb_ci'][1]:.2f}] | {b['valid_utf8']:.3f} | {b['distinct_4gram']:.3f} | "
                 f"{f(g(R,'s0.8','word_hit'))} | {f(g(R,'s0.8','ngram_bpb'),2)} | {f(ce(R,'overall'),4)} | {f(ce(R,'t750_1000'),3)} | {f(ce(R,'t500_750'),4)} | {b['sec_per_sample_b32']:.2f} |")
    if X:
        R0 = next(iter(X.values()))
        L.append("")
        L.append("Controls (same real val windows for every run): " + "; ".join(
            f"{k}: word_hit {v['word_hit']:.3f}, bpb {v['ngram_bpb']:.2f}" for k, v in R0["controls"].items()))
        for s in ("s1.0", "s0.8"):
            ns_ = [n for n in X if X[n]["samplers"].get(s)]
            wh = [X[n]["samplers"][s]["word_hit"] for n in ns_]; bp = [-X[n]["samplers"][s]["ngram_bpb"] for n in ns_]
            if len(ns_) > 2:
                L.append(f"Rank agreement word_hit vs (-bpb), {s}, {len(ns_)} runs: Spearman {spearman(wh, bp):+.2f}. Order by word_hit: "
                         + " > ".join(label[n] for _, n in sorted(zip(wh, ns_), reverse=True)) + "; by bpb (lower better): "
                         + " > ".join(label[n] for _, n in sorted(zip(bp, ns_), reverse=True)))
    out["part1"] = {n: dict(label=label[n], s1=R["samplers"]["s1.0"], s08=R["samplers"].get("s0.8"), val_ce=R.get("val_ce"), params=R["params"]) for n, R in X.items()}
    return "\n".join(L)


CONFIGS = [("w48", "U-Net w48"), ("w96", "U-Net w96"), ("w192", "U-Net w192"), ("hyb", "hybrid+logits")]


def part2(out):
    runs = {}
    for c, _ in CONFIGS:
        for s in (0, 1):
            p = f"runs/ladder_{c}_s{s}/ladder_eval.json"
            if os.path.exists(p):
                runs[(c, s)] = load(p)
    L = ["## Part 2: fixed-update ladder (25,000 updates, natural-v1, L2 recipe)", "",
         "### Val CE (EMA, nats/byte, uniform t, metrics.jsonl) at 6k / 12k / 18k / 25k", "",
         "| run | 6k | 12k | 18k | 25k |", "|---|---|---|---|---|"]
    for (c, s), R in runs.items():
        tr = R["train"]["val_ce_traj"]
        L.append(f"| ladder_{c}_s{s} | " + " | ".join(f(tr.get(str(u), tr.get(u)), 4) for u in (6000, 12000, 18000, 25000)) + " |")
    L += ["", "### Per run (final checkpoint, EMA, 128 samples, s=1.0)", "",
          "| run | params | ms/upd | train VRAM MB | CE all | CE t>=750 | CE t500-750 | word_hit [CI] | bpb [CI] | utf8 | d4 | r8 | mem | s/sample | s=0.8 wh | s=0.8 bpb |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (c, s), R in runs.items():
        b = R["samplers"]["s1.0"]; t = R["train"]
        L.append(f"| ladder_{c}_s{s} | {R['params']/1e6:.2f}M | {f(t.get('ms_per_update'),1)} | {f(t.get('train_peak_vram_mb'),0)} | {f(ce(R,'overall'),4)} | "
                 f"{f(ce(R,'t750_1000'),3)} | {f(ce(R,'t500_750'),4)} | {b['word_hit']:.3f} [{b['word_hit_ci'][0]:.3f},{b['word_hit_ci'][1]:.3f}] | "
                 f"{b['ngram_bpb']:.2f} [{b['ngram_bpb_ci'][0]:.2f},{b['ngram_bpb_ci'][1]:.2f}] | {b['valid_utf8']:.3f} | {b['distinct_4gram']:.3f} | "
                 f"{b['repeated_8gram']:.4f} | {f(b.get('memorization'))} | {b['sec_per_sample_b32']:.2f} | {f(g(R,'s0.8','word_hit'))} | {f(g(R,'s0.8','ngram_bpb'),2)} |")

    def agg(c, fn):
        v = [fn(runs[(c, s)]) for s in (0, 1) if (c, s) in runs]
        v = [x for x in v if x is not None]
        return v

    def ms(v, d=3):
        if not v:
            return "-"
        if len(v) == 1:
            return f"{v[0]:.{d}f} (n=1)"
        return f"{np.mean(v):.{d}f} +- {abs(v[0]-v[1])/2:.{d}f}"
    metrics = [("params (M)", lambda R: R["params"] / 1e6, 2), ("ms/update", lambda R: R["train"].get("ms_per_update"), 1),
               ("val CE overall", lambda R: ce(R, "overall"), 4), ("val CE t>=750", lambda R: ce(R, "t750_1000"), 3),
               ("val CE t500-750", lambda R: ce(R, "t500_750"), 4),
               ("word_hit s=1.0", lambda R: R["samplers"]["s1.0"]["word_hit"], 3), ("n-gram bpb s=1.0 (lower better)", lambda R: R["samplers"]["s1.0"]["ngram_bpb"], 3),
               ("valid_utf8", lambda R: R["samplers"]["s1.0"]["valid_utf8"], 3), ("distinct_4gram", lambda R: R["samplers"]["s1.0"]["distinct_4gram"], 3),
               ("word_hit s=0.8", lambda R: g(R, "s0.8", "word_hit"), 3), ("n-gram bpb s=0.8", lambda R: g(R, "s0.8", "ngram_bpb"), 3),
               ("s/sample (b32)", lambda R: R["samplers"]["s1.0"]["sec_per_sample_b32"], 3), ("train peak VRAM MB", lambda R: R["train"].get("train_peak_vram_mb"), 0),
               ("sampling peak VRAM MB", lambda R: R["samplers"]["s1.0"]["sample_peak_vram_mb"], 0)]
    L += ["", "### Per configuration: mean +- half-range over the 2 seeds (|s0-s1|/2)", "",
          "| metric | " + " | ".join(n for c, n in CONFIGS if any((c, s) in runs for s in (0, 1))) + " |",
          "|---|" + "---|" * sum(any((c, s) in runs for s in (0, 1)) for c, _ in CONFIGS)]
    summ = {}
    for name, fn, d in metrics:
        cells = []
        for c, _ in CONFIGS:
            if not any((c, s) in runs for s in (0, 1)):
                continue
            v = agg(c, fn); summ.setdefault(c, {})[name] = v; cells.append(ms(v, d))
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    # paired bootstrap on per-sample statistics (sampling noise only; pools both seeds' samples per config)
    def pooled(c, key):
        v = []
        for s in (0, 1):
            if (c, s) in runs:
                v += runs[(c, s)]["samplers"]["s1.0"]["per_sample"][key] if key != "wh" else []
        return np.array(v)

    def wh_arrays(c):
        h, w = [], []
        for s in (0, 1):
            if (c, s) in runs:
                ps = runs[(c, s)]["samplers"]["s1.0"]["per_sample"]; h += ps["word_hit"]; w += ps["words"]
        return np.array(h, float), np.array(w, float)

    def pdiff(c1, c2, n=4000):
        rng = np.random.default_rng(0)
        b1, b2 = pooled(c1, "ngram_bpb"), pooled(c2, "ngram_bpb")
        h1, w1 = wh_arrays(c1); h2, w2 = wh_arrays(c2)
        if len(b1) != len(b2) or len(b1) == 0:
            return None
        idx = rng.integers(0, len(b1), (n, len(b1)))
        dbpb = (b1[idx] - b2[idx]).mean(1)
        dwh = h1[idx].sum(1) / np.maximum(w1[idx].sum(1), 1) - h2[idx].sum(1) / np.maximum(w2[idx].sum(1), 1)
        ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
        return dict(dbpb=float(b1.mean() - b2.mean()), dbpb_ci=ci(dbpb), dwh=float(h1.sum() / w1.sum() - h2.sum() / w2.sum()), dwh_ci=ci(dwh))
    L += ["", "### Paired bootstrap over the shared 128+128 samples (sampling noise only, NOT training-seed noise), A minus B", "",
          "| A vs B | d bpb [95% CI] (neg = A better) | d word_hit [95% CI] (pos = A better) |", "|---|---|---|"]
    pair = {}
    for a, b in (("w96", "w48"), ("w192", "w96"), ("w192", "w48"), ("hyb", "w96")):
        d = pdiff(a, b)
        if d:
            pair[f"{a}-{b}"] = d
            L.append(f"| {a} vs {b} | {d['dbpb']:+.3f} [{d['dbpb_ci'][0]:+.3f},{d['dbpb_ci'][1]:+.3f}] | {d['dwh']:+.4f} [{d['dwh_ci'][0]:+.4f},{d['dwh_ci'][1]:+.4f}] |")
    L += ["", "Seed-to-seed (s0 minus s1, same config, 128 paired samples each):", "", "| config | d bpb | d word_hit | d CE overall | d CE t>=750 |", "|---|---|---|---|---|"]
    for c, n in CONFIGS:
        if (c, 0) in runs and (c, 1) in runs:
            a, b = runs[(c, 0)], runs[(c, 1)]
            L.append(f"| {n} | {a['samplers']['s1.0']['ngram_bpb'] - b['samplers']['s1.0']['ngram_bpb']:+.3f} | {a['samplers']['s1.0']['word_hit'] - b['samplers']['s1.0']['word_hit']:+.4f} | "
                     f"{f(None if ce(a,'overall') is None else ce(a,'overall') - ce(b,'overall'),4)} | {f(None if ce(a,'t750_1000') is None else ce(a,'t750_1000') - ce(b,'t750_1000'),4)} |")
    out["part2"] = dict(config_summary=summ, paired=pair)
    return "\n".join(L)


if __name__ == "__main__":
    out = {}
    txt = part1(out) + "\n\n" + part2(out) + "\n"
    print(txt)
    open("runs/ladder_summary.md", "w", encoding="utf-8").write(txt)
    json.dump(out, open("runs/ladder_summary.json", "w"), indent=1)
