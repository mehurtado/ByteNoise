"""Generation quality: text metrics over continuations (or unconditional samples when the model cannot continue)."""
import time

from ...datasets.store import load_prefix, load_split
from ...models.base import CAP_CONTINUE, CAP_GENERATE
from ..harness import register_mode
from ..metrics import text_metrics


@register_mode("generation", needs={CAP_GENERATE})
def run(model, spec, ctx):
    n = int(spec.get("n", 4))
    new_bytes = int(spec.get("new_bytes", 64))
    prompt_len = int(spec.get("prompt_len", 16))
    seeds = list(spec.get("seeds", [0, 1]))
    accepted = model.sampler_schema()  # one protocol serves several families: pass each model only the knobs it declares
    sampler = {k: v for k, v in (spec.get("sampler") or {}).items() if k in accepted}
    conditional = CAP_CONTINUE in model.capabilities
    if conditional:
        val = load_split(ctx["data"], ctx["split"])
        starts = [i * (len(val) - prompt_len) // max(n, 1) for i in range(n)]
        prompts = [bytes(val[s:s + prompt_len].tolist()) for s in starts]
    else:
        prompts = [b""] * n
    conts, shown, elapsed = [], [], 0.0
    for seed in seeds:
        for i, p in enumerate(prompts):
            t = time.perf_counter()
            c = model.generate(p, new_bytes, seed=seed * 1000 + i, **sampler)
            elapsed += time.perf_counter() - t
            conts.append(c)
            if seed == seeds[0] and len(shown) < n:
                shown.append({"prompt": p.decode("utf-8", "replace"), "text": c.decode("utf-8", "replace"), "repr": ascii(c)})
    train_bytes = None
    mem = int(spec.get("memorization_bytes", 8 << 20))
    if mem:
        try:
            train_bytes = load_prefix(ctx["data"], "train", mem)
        except FileNotFoundError:
            pass
    res = {"samples": shown, "n_generated": len(conts), "conditional": conditional, "sampler": sampler,
           **text_metrics(conts, train_bytes), "bytes_per_s": sum(map(len, conts)) / elapsed if elapsed else None}
    if train_bytes is not None:
        res["memorization_scope_bytes"] = len(train_bytes)
    return res
