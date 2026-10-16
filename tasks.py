"""Controlled byte-level tasks with train / in-distribution / held-out regimes.

Every example is a line of ASCII ending in "\\n". mask[i] = 1 where byte i is the answer, i.e. fully determined by
earlier bytes if the rule has been learned (teacher-forced accuracy is measured only there).
"""
import random
import string

LOWER = string.ascii_lowercase
DIGITS = string.digits


def _finish(text, answer_start, answer_end=None):
    data = (text + "\n").encode()
    end = len(data) if answer_end is None else answer_end
    mask = [0] * len(data)
    for i in range(answer_start, end):
        mask[i] = 1
    return data, mask


def gen_pattern(rng, period=None, length=None):
    p = period or rng.choice([2, 3, 4, 5])
    n = length or rng.randint(3 * p, 24)
    unit = "".join(rng.choice(LOWER) for _ in range(p))
    s = "p:" + (unit * (n // p + 1))[:n]
    return _finish(s, 2 + p, len(s))


def gen_copy(rng, length=None):
    n = length or rng.randint(3, 12)
    s = "".join(rng.choice(LOWER) for _ in range(n))
    prefix = "c:" + s + "|"
    return _finish(prefix + s, len(prefix), len(prefix) + n)


def gen_recall(rng, pairs=None):
    n = pairs or rng.randint(2, 6)
    keys = rng.sample(LOWER, n)
    vals = [rng.choice(DIGITS) for _ in keys]
    body = "".join(k + v for k, v in zip(keys, vals))
    q = rng.randrange(n)
    prefix = f"r:{body}?{keys[q]}="
    return _finish(prefix + vals[q], len(prefix), len(prefix) + 1)


def _apply_rule(kind, s):
    if kind == "rev":
        return s[::-1]
    if kind == "sort":
        return "".join(sorted(s))
    if kind == "inc":
        return "".join(LOWER[(LOWER.index(c) + 1) % 26] for c in s)
    raise ValueError(kind)


def gen_rule(rng, kind=None, length=None):
    kind = kind or rng.choice(["rev", "sort", "inc"])
    if length is None:
        length = rng.randint(3, 5) if kind == "sort" else rng.randint(3, 8)
    s = "".join(rng.choice(LOWER) for _ in range(length))
    prefix = f"v:{kind}:{s}="
    return _finish(prefix + _apply_rule(kind, s), len(prefix), len(prefix) + length)


# Training distribution: gen_*() with defaults. Held-out regimes below are never produced by the defaults.
REGIMES = {
    "pattern": {"id": lambda r: gen_pattern(r), "unseen_period": lambda r: gen_pattern(r, period=r.choice([6, 7])),
                "long": lambda r: gen_pattern(r, length=r.randint(40, 60))},
    "copy": {"id": lambda r: gen_copy(r), "long": lambda r: gen_copy(r, length=r.randint(16, 24))},
    "recall": {"id": lambda r: gen_recall(r), "long": lambda r: gen_recall(r, pairs=r.randint(8, 12))},
    "rule": {"id": lambda r: gen_rule(r), "unseen_combo": lambda r: gen_rule(r, "sort", r.randint(6, 8)),
             "long": lambda r: gen_rule(r, r.choice(["rev", "inc"]), r.randint(10, 14))},
}
TRAIN_GEN = {"pattern": gen_pattern, "copy": gen_copy, "recall": gen_recall, "rule": gen_rule}


def make_stream(n_bytes, seed, tasks=("pattern", "copy", "recall", "rule")):
    rng = random.Random(seed)
    out, total = [], 0
    while total < n_bytes:
        data, _ = TRAIN_GEN[rng.choice(tasks)](rng)
        out.append(data)
        total += len(data)
    return b"".join(out)[:n_bytes]


def make_eval_sets(seed, per_regime=200, exclude=None):
    """{task: {regime: [(bytes, mask), ...]}} with fresh examples; in-distribution ones avoid `exclude` strings."""
    rng = random.Random(seed)
    out = {}
    for task, regimes in REGIMES.items():
        out[task] = {}
        for name, fn in regimes.items():
            items, seen, tries = [], set(), 0
            while len(items) < per_regime and tries < per_regime * 50:
                tries += 1
                data, mask = fn(rng)
                if data in seen or (exclude is not None and name == "id" and data in exclude):
                    continue
                seen.add(data)
                items.append((data, mask))
            out[task][name] = items
    return out
