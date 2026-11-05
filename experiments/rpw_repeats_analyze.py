"""Paired comparison of two runs (B minus A) on identical-seed samples + teacher-forced intact-context word-exact error.
python experiments/rpw_repeats_analyze.py --a runs/rpw_rep1_s0 --b runs/rpw_rep2_s0 --out runs/rpw_rep2_s0/paired.json
Needs <run>/eval/mix_samples.json + mix_eval.json (experiments/mix_eval.py --n 64 --keep 64 --scales 1.0 --ce-windows 64). Eval only."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import torch
import data, train, eval as ev
from gate_text import wordlist
from ladder_eval import per_sample_words
import ngram_score as ns
import diag_spelling_run as dsr

TF_T = [800, 850, 900, 925]
SCORER = {"gutenberg": "natural-v1"}


def pboot(fa, fb, n, B=4000, seed=0):
    """Paired bootstrap over sample indices: f(idx) -> stat per resample; diff = b - a."""
    rng = np.random.default_rng(seed); idx = rng.integers(0, n, (B, n))
    d = fb(idx) - fa(idx); full = np.arange(n)[None]
    pa, pb = float(fa(full)[0]), float(fb(full)[0])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(a=pa, b=pb, diff=pb - pa, lo=float(lo), hi=float(hi))


def ratio(num, den):
    return lambda i: num[i].sum(1) / np.maximum(den[i].sum(1), 1)


def mean(x):
    return lambda i: x[i].mean(1)


@torch.no_grad()
def tf_word_exact(model, X, words, per_win=6, seed=0):
    gd = model.gaussian_diffusion; table = model.embedding_table().detach(); n = X.size(1)
    rng = np.random.default_rng(seed); Xn = X.cpu().numpy(); tg = []
    for w in range(X.size(0)):
        rb = bytes(Xn[w].astype(np.uint8).tolist())
        cand = [(m.start(), m.end()) for m in dsr.LET.finditer(rb) if m.end() - m.start() >= 4 and rb[m.start():m.end()].lower() in words]
        for k in rng.permutation(len(cand))[:per_win]: tg.append((w, *cand[k]))
    W = torch.tensor([t[0] for t in tg], device="cuda"); S = np.array([t[1] for t in tg]); E = np.array([t[2] for t in tg])
    out = {}
    for t in TF_T:
        g = torch.Generator().manual_seed(1000 + t); noise = torch.randn((len(tg), n, table.size(1)), generator=g)
        ok = np.zeros(len(tg), bool)
        for i in range(0, len(tg), 128):
            sl = slice(i, i + 128); xb = X[W[sl]]; x0 = model.byte_embedder(xb)
            xt = gd.q_sample(x0, torch.full((xb.size(0),), t, device="cuda", dtype=torch.long), noise[sl].to("cuda"))
            c = (dsr.two_pass(gd, table, xt, t).argmax(-1) == xb).cpu().numpy()
            for k in range(c.shape[0]): ok[i + k] = c[k, S[i + k]:E[i + k]].all()
        out[t] = ok
    return out, len(tg)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--eval-dir", default="eval"); ap.add_argument("--pairs", default="0,11,22,33,44,55")
    a = ap.parse_args()
    mix = json.load(open(os.path.join(data.DATA_ROOT, dsr.MIXN, "mix.json"))); z = np.load(os.path.join(data.DATA_ROOT, dsr.MIXN, "windows.npz"))
    val = data.load_split(dsr.MIXN, "val")
    S = {}; E = {}
    for k, r in (("a", a.a), ("b", a.b)):
        S[k] = [bytes.fromhex(h) for h in json.load(open(os.path.join(r, a.eval_dir, "mix_samples.json")))["s1.0"]]
        E[k] = json.load(open(os.path.join(r, a.eval_dir, "mix_eval.json")))
    n = len(S["a"]); assert n == len(S["b"]); R = dict(a=a.a, b=a.b, n_samples=n, metrics={}, ce={}, tf={})
    words_all = set(); hh = tt = bpb_all = None
    for s in mix["sources"]:
        tb = data.load_split(s["dataset"], "train").numpy().tobytes(); w = wordlist(tb); words_all |= w
        kn = ns.load_or_train(SCORER.get(s["name"], s["dataset"]), verbose=False)
        h, t, bpb = {}, {}, {}
        for k in "ab": h[k], t[k] = per_sample_words(S[k], w); bpb[k] = ns.score_bpb(kn, S[k])
        R["metrics"][f"word_hit/{s['name']}"] = pboot(ratio(h["a"], t["a"]), ratio(h["b"], t["b"]), n)
        R["metrics"][f"bpb/{s['name']}"] = pboot(mean(bpb["a"]), mean(bpb["b"]), n)
        add = lambda acc, new: new if acc is None else {k: acc[k] + new[k] for k in "ab"}
        hh, tt, bpb_all = add(hh, h), add(tt, t), add(bpb_all, bpb)
        print("scored", s["name"], flush=True)
    R["metrics"]["word_hit/pooled4"] = pboot(ratio(hh["a"], tt["a"]), ratio(hh["b"], tt["b"]), n)
    R["metrics"]["bpb/mean4"] = pboot(mean(bpb_all["a"] / 4), mean(bpb_all["b"] / 4), n)
    per = {k: dict(utf8=np.array([float(ev.valid_utf8(s)) for s in S[k]]), rep8=np.array([ev.repeated_ngram_fraction(s, 8) for s in S[k]]),
                   d4=np.array([ev.distinct_ngram_ratio(s, 4) for s in S[k]])) for k in "ab"}
    for m, nm in (("utf8", "valid_utf8"), ("rep8", "repeated_8gram"), ("d4", "distinct_4gram")):
        R["metrics"][nm] = pboot(mean(per["a"][m]), mean(per["b"][m]), n)
    for s in mix["sources"]:
        R["ce"][s["name"]] = {k: E[k]["ce"][s["name"]]["overall"]["ce"] for k in "ab"}
        R["ce"][s["name"] + "/t750+"] = {k: E[k]["ce"][s["name"]]["t750_1000"]["ce"] for k in "ab"}
    model, cfg = train.load_model_from_run(a.a, "ema", "cuda"); X = dsr.get_windows(val, z, mix, cfg, 64); del model
    ok = {}
    for k, r in (("a", a.a), ("b", a.b)):
        model, _ = train.load_model_from_run(r, "ema", "cuda"); model.eval()
        ok[k], nt = tf_word_exact(model, X, words_all); del model; torch.cuda.empty_cache()
    R["tf_n_targets"] = nt
    for t in TF_T:
        ea, eb = 1.0 - ok["a"][t].astype(float), 1.0 - ok["b"][t].astype(float)
        R["tf"][str(t)] = pboot(mean(ea), mean(eb), nt)
    R["pairs"] = [dict(i=int(i), a=S["a"][int(i)].decode("utf-8", "replace")[:260], b=S["b"][int(i)].decode("utf-8", "replace")[:260]) for i in a.pairs.split(",")]
    json.dump(R, open(a.out, "w"), indent=1)
    f = lambda d: f"{d['a']:.4f} | {d['b']:.4f} | {d['diff']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
    print("| metric | A | B | B-A [95% paired CI] |\n|---|---|---|---|")
    for k, d in R["metrics"].items(): print(f"| {k} | {f(d)} |")
    print("| TF word-exact err t | A | B | B-A |\n|---|---|---|---|")
    for t, d in R["tf"].items(): print(f"| {t} | {f(d)} |")
    for k, d in R["ce"].items(): print(k, d)


if __name__ == "__main__":
    main()
