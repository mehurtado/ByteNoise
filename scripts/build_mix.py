"""Build data/mix-pilot-v1: a document-safe, leakage-checked training mixture over four sources (no text is rewritten).

Sources (train bytes mix): tinystories-pilot-v1 40%, fineweb-edu-pilot-v1 35%, finewiki-en-pilot-v1 15%, Gutenberg 10%
(natural-v2 TRAIN split only; val/test = natural-v1/v2 val/test). The mixture is VIRTUAL (no copy of the text on disk): data.load_split concatenates
the sources' split bins in the order of mix.json; windows.npz stores admissible 256-byte window starts per source and split; mix.json
stores the source table and weights; data.mix_window_starts() draws weighted-source windows from them.
Windows lie wholly inside one document (modern sources: documents.jsonl; Gutenberg: books are concatenated without recorded
offsets, so the train blob is tiled as one document - at most one window per book boundary can straddle two books).
Checks (64-byte hashed shingles, all positions for the eval check, 1/8 content-sampled for train-vs-train):
  1. every train window that shares a shingle with ANY val/test text of ANY source is dropped (includes the historical
     natural-v1/v2 val/test evaluation books);
  2. a train window of a later source sharing a (sampled) shingle with an earlier source's train text is dropped.
Run once: python scripts/build_mix.py   (refuses to overwrite).
Option  --filter --out mix-clean-v1 : after the checks above, additionally drop train windows of the three modern sources
(NOT Gutenberg, which keeps its 10 % unchanged) that mix_filters.reasons() flags as table markup / whitespace padding /
symbol-heavy / repeated-symbol runs / digit-heavy. Val/test windows are never filtered (per-source CE stays comparable).
"""
import json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import data
import mix_filters

L, K = 256, 64
OUT = "mix-pilot-v1"
SOURCES = [("tinystories", "tinystories-pilot-v1", 0.40), ("fineweb_edu", "fineweb-edu-pilot-v1", 0.35),
           ("finewiki_en", "finewiki-en-pilot-v1", 0.15), ("gutenberg", "natural-v2", 0.10)]
M = np.uint64(0x9E3779B97F4A7C15)  # odd multiplier (invertible mod 2^64)


def _inv(a):
    x = a
    for _ in range(6):
        x = x * (np.uint64(2) - a * x)
    return x


def shingle_hashes(b, k=K, chunk=1 << 22):
    """Hash of every k-byte shingle of uint8 array b (len n-k+1 uint64). Polynomial hash with a finaliser."""
    n = len(b)
    out = np.empty(max(n - k + 1, 0), dtype=np.uint64)
    pm = np.uint64(0x100000001B3)
    with np.errstate(over="ignore"):
        ip = _inv(pm)
        pw = np.cumprod(np.concatenate([[np.uint64(1)], np.full(chunk + k, pm, dtype=np.uint64)]), dtype=np.uint64)[:chunk + k]
        ipw = np.cumprod(np.concatenate([[np.uint64(1)], np.full(chunk + k, ip, dtype=np.uint64)]), dtype=np.uint64)[:chunk + k]
        for s in range(0, len(out), chunk):
            e = min(s + chunk, len(out))
            seg = b[s:e + k - 1].astype(np.uint64) + np.uint64(1)
            H = np.concatenate([[np.uint64(0)], np.cumsum(seg * pw[:len(seg)], dtype=np.uint64)])
            h = (H[k:k + (e - s)] - H[:e - s]) * ipw[:e - s]
            h ^= h >> np.uint64(29); h *= M; h ^= h >> np.uint64(32)
            out[s:e] = h
    return out


def window_ok(flag, starts):
    """flag: bool per shingle position. A window [s, s+L) is bad if any shingle position in [s, s+L-K] is flagged."""
    c = np.concatenate([np.zeros(1, np.int32), np.cumsum(flag, dtype=np.int32)])
    hi = np.minimum(starts + L - K + 1, len(flag)); lo = np.minimum(starts, len(flag))
    return (c[hi] - c[lo]) == 0


def grid_starts(lo, hi):
    return lo + np.arange((hi - lo) // L, dtype=np.int64) * L


def main(OUT=OUT, filt=False):
    outdir = os.path.join(data.DATA_ROOT, OUT)
    if os.path.exists(outdir):
        raise SystemExit(f"{outdir} exists; refusing to overwrite")
    t0 = time.time()
    raw = {s: {n: np.fromfile(os.path.join(data.DATA_ROOT, d, f"{n}.bin"), dtype=np.uint8) for n in ("train", "val", "test")}
           for s, d, _ in SOURCES}
    cat, rng_ = {}, {}
    for split in ("train", "val", "test"):
        cat[split] = np.concatenate([raw[s][split] for s, _, _ in SOURCES]); off = 0; rng_[split] = {}
        for s, _, _ in SOURCES:
            rng_[split][s] = (off, off + len(raw[s][split])); off += len(raw[s][split])
    del raw
    cand = {sp: {} for sp in cat}; stats = {}
    for s, d, _ in SOURCES:
        for sp in cat:
            lo, hi = rng_[sp][s]
            if s == "gutenberg":
                st = grid_starts(lo, hi)
            else:
                st, _ = data.doc_window_starts(d, sp, L); st = st + lo
            cand[sp][s] = st
    ev_h = np.unique(np.concatenate([shingle_hashes(cat["val"]), shingle_hashes(cat["test"])]))
    print(f"eval shingles {len(ev_h):,} ({time.time()-t0:.0f}s)", flush=True)
    keep = {}
    seen_sampled = np.empty(0, dtype=np.uint64)
    for s, d, _ in SOURCES:
        lo, hi = rng_["train"][s]
        nsh = hi - lo - K + 1
        f_eval = np.zeros(nsh, bool); f_prev = np.zeros(nsh, bool); sampled = []
        for a in range(0, nsh, 1 << 25):  # slices keep peak memory low
            e = min(a + (1 << 25), nsh)
            h = shingle_hashes(cat["train"][lo + a: lo + e + K - 1])
            f_eval[a:e] = ev_h[np.minimum(np.searchsorted(ev_h, h), len(ev_h) - 1)] == h
            m = (h & np.uint64(7)) == 0
            hm = h[m]
            if len(seen_sampled):
                f_prev[a:e][m] = seen_sampled[np.minimum(np.searchsorted(seen_sampled, hm), len(seen_sampled) - 1)] == hm
            sampled.append(hm); del h, m
        if s != SOURCES[-1][0]:
            seen_sampled = np.sort(np.concatenate([seen_sampled] + sampled))
        del sampled
        st = cand["train"][s] - lo
        ok_e, ok_p = window_ok(f_eval, st), window_ok(f_prev, st)
        keep[s] = cand["train"][s][ok_e & ok_p]
        fstat = {}
        if filt and s != "gutenberg":
            kept_pre = len(keep[s]); bad = np.zeros(kept_pre, bool); cnt = {}
            reas = {k: np.zeros(kept_pre, bool) for k in ("ws_run", "pipes", "symbols", "sym_run", "digits")}
            for a0 in range(0, kept_pre, 50000):
                Wn = cat["train"][keep[s][a0:a0 + 50000, None] + np.arange(L)]
                for k, v in mix_filters.reasons(Wn).items():
                    reas[k][a0:a0 + 50000] = v
            for v in reas.values(): bad |= v
            fstat = dict(filter_kept_before=kept_pre, filter_dropped=int(bad.sum()), filter_dropped_by_reason={k: int(v.sum()) for k, v in reas.items()})
            keep[s] = keep[s][~bad]
        stats[s] = dict(train_bytes=int(hi - lo), candidate_windows=int(len(st)), dropped_eval_overlap=int((~ok_e).sum()),
                        dropped_prev_source_overlap=int((~ok_p & ok_e).sum()), kept_windows=int(len(keep[s])),
                        shingles_shared_with_eval=int(f_eval.sum()), shingles_shared_with_earlier_sources_sampled=int(f_prev.sum()),
                        val_windows=int(len(cand["val"][s])), test_windows=int(len(cand["test"][s])), **fstat)
        print(s, stats[s], f"({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(outdir)
    arrs = {f"train_{s}": keep[s] for s, _, _ in SOURCES}
    arrs.update({f"{sp}_{s}": cand[sp][s] for sp in ("val", "test") for s, _, _ in SOURCES})
    np.savez(os.path.join(outdir, "windows.npz"), **arrs)
    mix = dict(window=L, shingle=K, filter=bool(filt), stats=stats,
               sources=[dict(name=s, dataset=d, weight=w, ranges={sp: list(map(int, rng_[sp][s])) for sp in cat}) for s, d, w in SOURCES])
    json.dump(mix, open(os.path.join(outdir, "mix.json"), "w"), indent=1)
    json.dump(dict(name=OUT, kind="mixture", note="see scripts/build_mix.py; Gutenberg = natural-v2 train only" + ("; modern-source train windows filtered by mix_filters" if filt else ""),
                   splits={sp: {"bytes": int(len(cat[sp]))} for sp in cat}), open(os.path.join(outdir, "manifest.json"), "w"), indent=1)
    print("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT); ap.add_argument("--filter", action="store_true")
    a = ap.parse_args()
    if a.filter and a.out == OUT:
        raise SystemExit("--filter needs a new --out name (never overwrite mix-pilot-v1)")
    main(a.out, a.filter)
