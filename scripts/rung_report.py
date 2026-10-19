"""Rung comparison table: width-96 ladder baseline (mean of seeds 0,1) vs each runs/rung_* config (seed 0 and, if run, seed 1).

Reads ladder_eval.json of every run (scripts/ladder_eval.py). Prints markdown and writes runs/rung_summary.md/.json.
`--flag` prints only the rung names whose seed-0 result beats the baseline mean by more than the seed spread
(CE overall <= -0.003, or word_hit s1.0 >= +0.05, or bpb s1.0 <= -0.1): the confirmation-stage candidates."""
import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CE_TOL, WH_TOL, BPB_TOL = 0.003, 0.05, 0.1
ORDER = ["bs128", "bs128_25k", "R4", "R6", "tied_R8", "tied_R4", "tied_R6", "span30", "span60"]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def ce(R, k):
    return None if not R.get("val_ce") else R["val_ce"][k]["ce"]


def f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def metrics(R):
    s1, s8, t = R["samplers"]["s1.0"], R["samplers"].get("s0.8"), R["train"]
    return dict(params=R["params"] / 1e6, ms=t.get("ms_per_update"), updates=t.get("updates"), ce=ce(R, "overall"), ce750=ce(R, "t750_1000"), ce500=ce(R, "t500_750"),
                wh=s1["word_hit"], wh_ci=s1["word_hit_ci"], bpb=s1["ngram_bpb"], bpb_ci=s1["ngram_bpb_ci"], utf8=s1["valid_utf8"], d4=s1["distinct_4gram"],
                wh8=s8 and s8["word_hit"], bpb8=s8 and s8["ngram_bpb"], sps=s1["sec_per_sample_b32"])


def mean_of(ms):
    keys = ["params", "ms", "ce", "ce750", "ce500", "wh", "bpb", "utf8", "d4", "wh8", "bpb8", "sps"]
    out = {k: (None if any(m[k] is None for m in ms) else float(np.mean([m[k] for m in ms]))) for k in keys}
    out["wh_ci"] = [float(np.mean([m["wh_ci"][i] for m in ms])) for i in (0, 1)]
    out["bpb_ci"] = [float(np.mean([m["bpb_ci"][i] for m in ms])) for i in (0, 1)]
    out["updates"] = ms[0]["updates"]
    return out


def gather():
    base = [load(f"runs/ladder_w96_s{s}/ladder_eval.json") for s in (0, 1, 2, 3)]
    base = [b for b in base if b]
    rungs = {}
    for name in ORDER:
        rs = [load(f"runs/rung_{name}{suf}/ladder_eval.json") for suf in ("", "_s1")]
        if rs[0]:
            rungs[name] = rs
    return base, rungs


SPREAD = {}  # metric -> minimal real difference, 2*sqrt(2)*SD over the baseline seeds (set in main when >= 3 seeds)


def beats(m, b):
    ce_t, wh_t, bpb_t = SPREAD.get("ce", CE_TOL), SPREAD.get("wh", WH_TOL), SPREAD.get("bpb", BPB_TOL)
    return (m["ce"] is not None and m["ce"] - b["ce"] <= -ce_t) or m["wh"] - b["wh"] >= wh_t or m["bpb"] - b["bpb"] <= -bpb_t


def row(label, m, b=None):
    d = ""
    if b is not None:
        d = f" ({m['ce'] - b['ce']:+.4f})" if m["ce"] is not None and b["ce"] is not None else ""
    wd = f" ({m['wh'] - b['wh']:+.3f})" if b else ""
    bd = f" ({m['bpb'] - b['bpb']:+.2f})" if b else ""
    return (f"| {label} | {m['params']:.2f}M | {f(m['ms'], 1)} | {f(m['ce'], 4)}{d} | {f(m['ce750'], 3)} | {m['wh']:.3f} [{m['wh_ci'][0]:.3f},{m['wh_ci'][1]:.3f}]{wd} | "
            f"{f(m['wh8'])} | {m['bpb']:.2f} [{m['bpb_ci'][0]:.2f},{m['bpb_ci'][1]:.2f}]{bd} | {f(m['bpb8'], 2)} | {m['utf8']:.3f} | {m['d4']:.3f} | {m['sps']:.2f} |")


def traj(path):
    rows = [json.loads(l) for l in open(os.path.join(path, "metrics.jsonl"))] if os.path.exists(os.path.join(path, "metrics.jsonl")) else []
    tr = [r for r in rows if not r.get("eval")]
    ev = [r for r in rows if r.get("eval")]
    ex = {r["update"]: r["examples"] for r in tr}
    return [(ex.get(r["update"], r.get("examples")), r["update"], r.get("ema_val_ce_loss")) for r in ev]


def main():
    base, rungs = gather()
    if not base:
        sys.exit("no baseline ladder_w96 results")
    bm = [metrics(b) for b in base]
    B = mean_of(bm)
    sd_rows = []
    if len(bm) >= 3:  # 2*sqrt(2)*SD: the difference of two single runs that is ~95% unlikely under seed noise alone
        for k, nm in (("ce", "ce"), ("ce750", None), ("wh", "wh"), ("wh8", None), ("bpb", "bpb"), ("bpb8", None)):
            v = [m[k] for m in bm]; sd = float(np.std(v, ddof=1))
            sd_rows.append((k, float(np.mean(v)), sd, 2 * 2 ** 0.5 * sd))
            if nm:
                SPREAD[nm] = 2 * 2 ** 0.5 * sd
    if "--flag" in sys.argv:
        print(" ".join(n for n, rs in rungs.items() if beats(metrics(rs[0]), B)))
        return
    L = ["# Rung screening vs width-96 baseline (natural-v1; EMA; 128 paired samples; deltas vs baseline mean in brackets)", "",
         "| config | params | ms/upd | val CE all | CE t>=750 | word_hit s=1.0 [CI] | word_hit s=0.8 | bpb s=1.0 [CI] | bpb s=0.8 | utf8 | d4 | s/sample |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|",
         row("w96 baseline (mean s0,s1)", B)]
    for i, b in enumerate(base):  # noqa
        L.append(row(f"  baseline s{i}", metrics(b)))
    out = {"baseline": B, "rungs": {}}
    for name, rs in rungs.items():
        for s, R in enumerate(rs):
            if R:
                m = metrics(R)
                L.append(row(f"rung_{name} s{s} ({m['updates']} upd)", m, B))
                out["rungs"][f"{name}_s{s}"] = dict(m, beats_baseline=bool(beats(m, B)))
    if sd_rows:
        L += ["", f"Baseline seed spread over {len(bm)} seeds (mean, SD, minimal real difference 2*sqrt(2)*SD for a two-run comparison):", "",
              "| metric | mean | SD | min real diff |", "|---|---|---|---|"]
        L += [f"| {k} | {m:.4f} | {sd:.4f} | {md:.4f} |" for k, m, sd, md in sd_rows]
        out["baseline_spread"] = {k: dict(mean=m, sd=sd, min_diff=md) for k, m, sd, md in sd_rows}
    L += ["", f"Confirmation rule (seed 0 vs baseline mean): CE <= -{SPREAD.get('ce', CE_TOL):.4f}, or word_hit >= +{SPREAD.get('wh', WH_TOL):.4f}, or bpb <= -{SPREAD.get('bpb', BPB_TOL):.3f}"
          + (" (4-seed SD based)" if SPREAD else " (fallback fixed thresholds)") + ". Candidates: "
          + (" ".join(n for n, rs in rungs.items() if beats(metrics(rs[0]), B)) or "none")]
    if "bs128" in rungs:  # equal-examples val-CE trajectory (quick_eval on the first 1024 val windows; t/noise draws differ between batch sizes)
        L += ["", "## Batch-size rung: EMA val CE (metrics.jsonl, quick_eval) at equal examples seen", "",
              "| examples | w96 bs32 s0 | w96 bs32 s1 | rung_bs128 |", "|---|---|---|---|"]
        T = [{e: c for e, u, c in traj(f"runs/ladder_w96_s{s}")} for s in (0, 1)]
        T128 = {e: (u, c) for e, u, c in traj("runs/rung_bs128")}
        for e in sorted(T128):
            L.append(f"| {e:,} | {f(T[0].get(e), 4)} | {f(T[1].get(e), 4)} | {f(T128[e][1], 4)} (upd {T128[e][0]}) |")
        L.append("")
        L.append("Equal updates: bs32 at 6250 updates is not logged (eval_interval 2000); compare rung_bs128 against rung_bs128_25k (25000 updates, 4x the data) if that run exists.")
    txt = "\n".join(L) + "\n"
    print(txt)
    open("runs/rung_summary.md", "w", encoding="utf-8").write(txt)
    json.dump(out, open("runs/rung_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
