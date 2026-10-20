"""Byte-level order-5 interpolated Kneser-Ney scorer (a SCORING tool only, never a generator).

Trained once on a prepared dataset's train bytes and cached (default data/ngram_<data>.pkl, gitignored).
score_bpb(samples) = mean bits per byte of each sample under the model, scored at positions >= order-1 so that every
position uses the full 5-byte history window (same convention for samples, real val windows and shuffled controls).
Lower = more like train-set English.

Model: highest order uses raw counts, lower orders use continuation counts (number of distinct left extensions),
absolute discount D_n = n1/(n1+2*n2) per order, interpolated down to a uniform 1/256 base. Probabilities sum to 1 over
the 256 bytes for every context (checked by selftest()).
"""
import os, pickle, sys, time
import numpy as np

ORDER = 5
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _kgram_keys(b, n):
    k = b[: len(b) - n + 1].astype(np.uint64)
    for j in range(1, n):
        k = (k << np.uint64(8)) | b[j: len(b) - n + 1 + j].astype(np.uint64)
    return k


def train_model(train_bytes, order=ORDER):
    b = np.frombuffer(train_bytes, dtype=np.uint8)
    uk, uc = {}, {}
    for n in range(1, order + 1):
        keys, cnt = np.unique(_kgram_keys(b, n), return_counts=True)
        uk[n], uc[n] = keys, cnt
    levels = {}
    for n in range(1, order + 1):
        if n == order:
            keys, c = uk[n], uc[n].astype(np.float64)
        else:  # continuation counts from the (n+1)-grams' suffixes
            suf = uk[n + 1] % np.uint64(256 ** n)
            keys, c = np.unique(suf, return_counts=True)
            c = c.astype(np.float64)
        n1, n2 = (c == 1).sum(), (c == 2).sum()
        D = n1 / (n1 + 2.0 * n2)
        ctx = keys // np.uint64(256)
        ctx_keys, inv = np.unique(ctx, return_inverse=True)
        tot = np.bincount(inv, weights=c)
        nuniq = np.bincount(inv).astype(np.float64)
        levels[n] = dict(keys=keys, c=c.astype(np.float32), D=float(D), ctx_keys=ctx_keys,
                         tot=tot.astype(np.float64), nuniq=nuniq.astype(np.float32))
    return dict(order=order, levels=levels)


def _lookup(sorted_keys, q):
    pos = np.searchsorted(sorted_keys, q)
    pos = np.minimum(pos, len(sorted_keys) - 1)
    return pos, sorted_keys[pos] == q


def byte_probs(model, arr, ctxs=None):
    """arr: (S, L) uint8. Returns P(arr[:, i] | previous bytes) for i >= order-1, shape (S, L-order+1), float64."""
    order = model["order"]
    S, L = arr.shape
    a = arr.astype(np.uint64)
    idx = np.arange(order - 1, L)
    p = np.full((S, len(idx)), 1.0 / 256)
    for n in range(1, order + 1):
        lv = model["levels"][n]
        ctx = np.zeros((S, len(idx)), dtype=np.uint64)
        for j in range(1, n):
            ctx |= a[:, idx - j] << np.uint64(8 * (j - 1))
        full = (ctx << np.uint64(8)) | a[:, idx]
        pos, ok = _lookup(lv["keys"], full.ravel())
        c = np.where(ok, lv["c"][pos], 0.0).reshape(full.shape)
        cpos, cok = _lookup(lv["ctx_keys"], ctx.ravel())
        tot = np.where(cok, lv["tot"][cpos], 0.0).reshape(full.shape)
        nu = np.where(cok, lv["nuniq"][cpos], 0.0).reshape(full.shape)
        has = tot > 0
        safe = np.where(has, tot, 1.0)
        new = np.maximum(c - lv["D"], 0.0) / safe + lv["D"] * nu / safe * p
        p = np.where(has, new, p)
    return p


def score_bpb(model, samples):
    """samples: list of bytes (equal length). Returns per-sample bits/byte (np.ndarray)."""
    L = min(len(s) for s in samples)
    arr = np.stack([np.frombuffer(bytes(s[:L]), dtype=np.uint8) for s in samples])
    return -np.log2(byte_probs(model, arr)).mean(axis=1)


def load_or_train(data_name="natural-v1", path=None, verbose=True):
    path = path or os.path.join(ROOT, "data", f"ngram_{data_name}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    sys.path.insert(0, ROOT)
    import data
    t0 = time.time()
    m = train_model(data.load_split(data_name, "train").numpy().tobytes())
    with open(path, "wb") as f:
        pickle.dump(m, f, protocol=4)
    if verbose:
        print(f"trained n-gram scorer on {data_name} in {time.time() - t0:.0f}s -> {path}", flush=True)
    return m


def selftest(model, n_ctx=40, seed=0, data_name="natural-v1"):
    """Max |sum_w P(w|h) - 1| over random contexts drawn from real text (normalisation check)."""
    sys.path.insert(0, ROOT)
    import data
    val = data.load_split(data_name, "val").numpy()
    rng = np.random.default_rng(seed)
    worst = 0.0
    for st in rng.integers(0, len(val) - 10, n_ctx):
        ctx = val[st: st + model["order"] - 1]
        rows = np.tile(np.concatenate([ctx, [0]]).astype(np.uint8), (256, 1))
        rows[:, -1] = np.arange(256)
        worst = max(worst, abs(byte_probs(model, rows).sum() - 1.0))
    return worst


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "natural-v1"  # dataset whose train split the scorer is trained on
    m = load_or_train(name)
    print("normalisation max |sum-1| =", selftest(m, data_name=name))
