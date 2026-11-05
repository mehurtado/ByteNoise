"""Controlled-task accuracy (copy/recall/rule/pattern) on freshly generated, seeded examples via ByteModel.score top1."""
import torch

from ...core import legacy
from ...models.base import CAP_SCORE
from ..harness import register_mode


def _eval_sets(T, ctx, per_regime):
    """The dataset's frozen eval_tasks.json when present (identical to legacy eval_tasks); otherwise fresh items from a seed
    distinct from the training stream's, with in-distribution items that occur in the train split excluded."""
    import data as legacy_data
    from ...datasets.store import dataset_path, load_split
    if (dataset_path(ctx["data"]) / "eval_tasks.json").exists():
        sets = legacy_data.load_eval_tasks(ctx["data"])
        return {t: {r: items[:per_regime] for r, items in regs.items()} for t, regs in sets.items()}
    train = bytes(load_split(ctx["data"], "train").numpy())
    exclude = set(line + b"\n" for line in train.split(b"\n"))
    return T.make_eval_sets(ctx["seed"] + 3, per_regime=per_regime, exclude=exclude)


@register_mode("tasks", needs={CAP_SCORE})
def run(model, spec, ctx):
    legacy.ensure()
    import tasks as T
    sets = _eval_sets(T, ctx, int(spec.get("per_regime", 100)))
    L = model.seq_len + 1
    bs = int(spec.get("batch_size", 64))
    out = {}
    for task, regimes in sets.items():
        out[task] = {}
        for regime, items in regimes.items():
            tok_ok = tok_n = ex_ok = 0
            for i in range(0, len(items), bs):
                chunk = [(b[:L], m[:L]) for b, m in items[i:i + bs]]
                width = max(len(b) for b, _ in chunk)
                x = torch.zeros(len(chunk), width, dtype=torch.long)  # right padding cannot affect earlier positions (causal)
                for r, (b, _) in enumerate(chunk):
                    x[r, :len(b)] = torch.tensor(list(b))
                top1 = model.score(x).top1  # top1[:, j] says whether byte j+1 was predicted
                for r, (b, m) in enumerate(chunk):
                    ok = [bool(top1[r, j - 1]) for j in range(1, len(b)) if m[j]]
                    tok_ok += sum(ok)
                    tok_n += len(ok)
                    ex_ok += bool(ok) and all(ok)
            out[task][regime] = {"token_acc": tok_ok / max(tok_n, 1), "exact_match": ex_ok / max(len(items), 1), "examples": len(items)}
    return {"tasks": out}
