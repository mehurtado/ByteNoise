"""Bits per byte over held-out windows via ByteModel.score (exact for AR models)."""
import math

import torch

from ...datasets.store import load_split, windows
from ...models.base import CAP_SCORE
from ..harness import register_mode

LN2 = math.log(2)
BUCKETS = ((0, 16), (16, 64), (64, 256), (256, None))


@register_mode("bpb", needs={CAP_SCORE})
def run(model, spec, ctx):
    L = model.seq_len
    w = windows(load_split(ctx["data"], ctx["split"]), L + 1, L + 1, spec.get("max_windows"))
    if len(w) == 0:
        raise ValueError(f"{ctx['data']}/{ctx['split']} has fewer than {L + 1} bytes")
    sums = torch.zeros(L, dtype=torch.float64)
    exact = True
    bs = int(spec.get("batch_size", 32))
    for i in range(0, len(w), bs):
        out = model.score(w[i:i + bs])
        exact = exact and out.exact
        sums += out.nll.double().sum(0)
    n = len(w)
    res = {"bpb": sums.sum().item() / (n * L) / LN2, "exact": exact, "windows": n, "bytes": n * L, "seq_len": L, "bpb_by_context": {}}
    for lo, hi in BUCKETS:
        hi_c = min(hi, L) if hi else L
        if lo < hi_c:
            res["bpb_by_context"][f"{lo}-{hi or 'inf'}"] = sums[lo:hi_c].sum().item() / (n * (hi_c - lo)) / LN2
    return res
