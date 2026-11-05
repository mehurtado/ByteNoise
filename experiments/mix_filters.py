"""Window-level junk detectors for byte windows (training windows or generated samples): whitespace runs, markdown/table
fragments (pipes, symbol-heavy), repeated symbol runs, digit-heavy templated lists. Vectorised on an (N, L) uint8 array.
Used by build_mix.py --filter (training-window filter), mix_verify.py and mix_eval.py (junk-excluded sample metrics)."""
import numpy as np

WS_RUN, PIPES, SYM_FRAC, SYM_RUN, DIGIT_FRAC = 6, 3, 0.15, 6, 0.35  # thresholds (see docs/experiments.md, Clean-data mixture retry)
_ALLOWED = set(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t\n\r.,;:'\"!?-()/&%$")
SYMBOL = np.array([b < 128 and b not in _ALLOWED for b in range(256)])  # ASCII symbols / control bytes; bytes >= 128 are not symbols
WSP = np.zeros(256, bool); WSP[[9, 10, 13, 32]] = True
DIG = np.zeros(256, bool); DIG[48:58] = True


def _maxrun(M):
    """Longest run of True per row of a bool (N, L) matrix."""
    run = np.zeros(len(M), np.int32); best = np.zeros(len(M), np.int32)
    for j in range(M.shape[1]):
        run = np.where(M[:, j], run + 1, 0); best = np.maximum(best, run)
    return best


def _sameruns(W):
    """Longest run of one repeated non-alphanumeric, non-whitespace byte per row."""
    sym = ~np.isin(W, np.frombuffer(bytes(range(48, 58)) + bytes(range(65, 91)) + bytes(range(97, 123)), np.uint8)) & ~WSP[W] & (W < 128)
    run = np.zeros(len(W), np.int32); best = np.zeros(len(W), np.int32)
    for j in range(W.shape[1]):
        same = np.zeros(len(W), bool) if j == 0 else (W[:, j] == W[:, j - 1])
        run = np.where(sym[:, j], np.where(same, run + 1, 1), 0); best = np.maximum(best, run)
    return best


def reasons(W):
    """dict reason -> bool array (N,) for a uint8 (N, L) array."""
    W = np.asarray(W, np.uint8)
    return {"ws_run": _maxrun(WSP[W]) >= WS_RUN, "pipes": (W == 124).sum(1) >= PIPES, "symbols": SYMBOL[W].mean(1) > SYM_FRAC,
            "sym_run": _sameruns(W) >= SYM_RUN, "digits": DIG[W].mean(1) > DIGIT_FRAC}


def is_junk(W):
    r = reasons(W); out = np.zeros(len(W), bool)
    for v in r.values():
        out |= v
    return out


def windows_of(b, win, stride):
    """All win-byte windows of bytes b (every `stride`) as (N, win) uint8, or empty."""
    a = np.frombuffer(b, np.uint8)
    if len(a) < win:
        return np.zeros((0, win), np.uint8)
    return np.stack([a[i:i + win] for i in range(0, len(a) - win + 1, stride)])


def junk_sample_metrics(samples, mem, ngram=8):
    """Sample metrics recomputed with junk excluded. samples: list of bytes; mem: train bytes.
    repeated_8gram_nojunk = mean over samples of (repeated 8-gram positions / positions) counting only 8-grams that are not junk
    (an 8-gram is junk if its 8 bytes lie in a whitespace run >= 6... approximated by the 8-byte window being all whitespace
    / containing '|' / >=4 symbol bytes / a symbol run >= 5). Memorization (32/64-byte windows, strides 16/32, as mix_eval) is
    split into windows containing junk (window-level reasons above) and the rest."""
    out = {}
    fr = []
    for s in samples:
        a = np.frombuffer(s, np.uint8)
        if len(a) < ngram: continue
        G = np.stack([a[i:i + ngram] for i in range(len(a) - ngram + 1)])
        bad = WSP[G].sum(1) >= 5; bad |= (G == 124).any(1); bad |= SYMBOL[G].sum(1) >= 3
        seen, rep, tot = set(), 0, 0
        for g, b in zip(G, bad):
            k = g.tobytes()
            if not b:
                tot += 1; rep += k in seen
            seen.add(k)
        if tot: fr.append(rep / tot)
    out["repeated_8gram_nojunk"] = float(np.mean(fr)) if fr else 0.0
    for win, stride in ((32, 16), (64, 32)):
        ws = [bytes(s[i:i + win]) for s in samples for i in range(0, len(s) - win + 1, stride)][:2000]
        if not ws: continue
        W = np.stack([np.frombuffer(w, np.uint8) for w in ws]); jk = is_junk(W)
        hit = np.array([w in mem for w in ws])
        out[f"mem{win}_all"] = float(hit.mean()); out[f"mem{win}_junk_windows"] = float(jk.mean())
        out[f"mem{win}_hits_junk"] = int((hit & jk).sum()); out[f"mem{win}_hits_clean"] = int((hit & ~jk).sum())
        out[f"mem{win}_nojunk"] = float(hit[~jk].mean()) if (~jk).any() else 0.0
    cat = b"".join(samples); a = np.frombuffer(cat, np.uint8)
    out["space_run8_frac"] = float(spacerun_fraction(samples))
    out["table_junk_samples"] = int(sum(s.count(b"|") > 10 for s in samples))
    return out


def spacerun_fraction(samples, k=8):
    """Fraction of all sample bytes inside runs of >= k spaces (mix_eval/EXPERIMENTS 'space runs')."""
    import re
    pat = re.compile(rb" {%d,}" % k)
    tot = sum(len(s) for s in samples)
    return sum(len(m.group()) for s in samples for m in pat.finditer(s)) / max(tot, 1)
