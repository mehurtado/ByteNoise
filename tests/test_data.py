"""CPU-only, offline tests for data.py and tasks.py."""
import hashlib
import json
import os
import random
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import data  # noqa: E402
import tasks  # noqa: E402


# ---------------------------------------------------------------- (1) assign_splits

def test_assign_splits_deterministic_and_complete():
    for n in (3, 5, 10, 37, 100):
        ids = list(range(n))
        a = data.assign_splits(ids)
        b = data.assign_splits(list(reversed(ids)))
        assert a == b
        assert set(a) == set(ids)
        assert set(a.values()) <= {"train", "val", "test"}
        counts = {s: sum(1 for v in a.values() if v == s) for s in ("train", "val", "test")}
        assert counts["val"] >= 1 and counts["test"] >= 1
        assert counts["train"] >= 1
        assert sum(counts.values()) == n


def test_assign_splits_each_doc_in_one_split():
    ids = [f"doc{i}" for i in range(50)]
    out = data.assign_splits(ids)
    assert len(out) == len(ids)
    members = {s: {i for i, v in out.items() if v == s} for s in ("train", "val", "test")}
    assert members["train"].isdisjoint(members["val"])
    assert members["train"].isdisjoint(members["test"])
    assert members["val"].isdisjoint(members["test"])
    assert members["train"] | members["val"] | members["test"] == set(ids)


# ---------------------------------------------------------------- (2) build_text_splits

def _synthetic_docs(n=20):
    shared_long = "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the hills."
    shared_variant = "  THE quick   brown fox jumps\nover the lazy dog while the sun sets slowly behind the hills. "
    docs = {}
    for i in range(n):
        paras = [
            "Chapter I",  # short, shared everywhere
            f"Unique paragraph number {i} " + "x" * 80,
            shared_long if i % 2 == 0 else shared_variant,
            f"Doc {i} repeats itself within the same document so it must be dropped after the first copy. " * 1,
            f"Doc {i} repeats itself within the same document so it must be dropped after the first copy. " * 1,
        ]
        docs[i] = ({"title": f"t{i}"}, "\n\n".join(paras))
    return docs


def test_build_text_splits_no_cross_split_paragraphs_and_train_dedup():
    docs = _synthetic_docs()
    outputs, split_of, stats = data.build_text_splits(docs)
    assert set(outputs) == {"train", "val", "test"}
    seen = {}
    for s, blob in outputs.items():
        keys = [data._norm_para(p) for p in data._paragraphs(blob.decode("utf-8"))]
        long_keys = [k for k in keys if len(k) >= 60]
        assert len(long_keys) == len(set(long_keys)), f"duplicate long paragraph inside {s}"
        for k in long_keys:
            assert k not in seen or seen[k] == s, "long paragraph in two splits"
            seen[k] = s
    # the shared paragraph survives in exactly one split (first in order: train)
    shared_key = data._norm_para("The quick brown fox jumps over the lazy dog while the sun sets slowly behind the hills.")
    assert seen[shared_key] == "train"
    assert stats["dropped_duplicate"]["val"] > 0 and stats["dropped_duplicate"]["test"] > 0
    assert stats["dropped_duplicate"]["train"] > 0


def test_build_text_splits_short_paragraphs_retained():
    docs = _synthetic_docs()
    outputs, split_of, _ = data.build_text_splits(docs)
    for s in ("train", "val", "test"):
        n_docs = sum(1 for v in split_of.values() if v == s)
        paras = data._paragraphs(outputs[s].decode("utf-8"))
        assert sum(1 for p in paras if p == "Chapter I") == n_docs
        assert outputs[s].endswith(b"\n")


# ---------------------------------------------------------------- (3) _parse_gutenberg

def test_parse_gutenberg_strips_boilerplate_and_reads_meta():
    text = (
        "The Project Gutenberg eBook of Test Book\r\n\r\n"
        "Title: A Test Book\r\nAuthor: Jane Doe\r\nLanguage: English\r\n\r\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK A TEST BOOK ***\r\n\r\n"
        "Chapter 1\r\n\r\nReal body text here.\r\n\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK A TEST BOOK ***\r\n"
        "License boilerplate that must go.\r\n"
    )
    meta, body = data._parse_gutenberg(text)
    assert meta == {"title": "A Test Book", "author": "Jane Doe", "language": "English"}
    assert "Real body text here." in body and "Chapter 1" in body
    assert "PROJECT GUTENBERG" not in body
    assert "License boilerplate" not in body
    assert "Language:" not in body
    assert "\r" not in body
    assert body == body.strip()


def test_parse_gutenberg_this_variant_and_missing_markers():
    t = "Title: X\n*** START OF THIS PROJECT GUTENBERG EBOOK X ***\nbody\n*** END OF THIS PROJECT GUTENBERG EBOOK X ***\nfoot"
    meta, body = data._parse_gutenberg(t)
    assert body == "body" and meta["title"] == "X" and meta["language"] == ""
    meta, body = data._parse_gutenberg("no markers here")
    assert body == "no markers here"


# ---------------------------------------------------------------- (4) task generators

def _check_common(line, mask):
    assert isinstance(line, bytes)
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    line.decode("ascii")
    assert len(mask) == len(line)
    assert set(mask) <= {0, 1}
    assert mask[-1] == 0
    idx = [i for i, m in enumerate(mask) if m]
    assert idx and idx == list(range(idx[0], idx[-1] + 1)), "mask must be contiguous"
    return line[:-1].decode(), idx


def check_pattern(line, mask):
    text, idx = _check_common(line, mask)
    assert text.startswith("p:")
    s = text[2:]
    p = idx[0] - 2
    assert p >= 1 and idx[-1] == len(text) - 1
    unit = s[:p]
    assert all(s[i] == unit[i % p] for i in range(len(s)))
    assert len(s) > p
    return p, len(s)


def check_copy(line, mask):
    text, idx = _check_common(line, mask)
    assert text.startswith("c:")
    src, _, dst = text[2:].partition("|")
    assert "|" not in dst
    assert dst == src
    assert idx == list(range(len(text) - len(src), len(text)))
    return len(src)


def check_recall(line, mask):
    text, idx = _check_common(line, mask)
    assert text.startswith("r:")
    body, _, rest = text[2:].partition("?")
    key, _, val = rest.partition("=")
    pairs = [(body[i], body[i + 1]) for i in range(0, len(body), 2)]
    assert len(body) % 2 == 0
    d = dict(pairs)
    assert len(d) == len(pairs), "keys unique"
    assert all(k.isalpha() and v.isdigit() for k, v in pairs)
    assert d[key] == val and len(val) == 1
    assert idx == [len(text) - 1]
    return len(pairs)


def check_rule(line, mask):
    text, idx = _check_common(line, mask)
    assert text.startswith("v:")
    kind, arg = text[2:].split(":", 1)
    s, _, ans = arg.partition("=")
    if kind == "rev":
        exp = "".join(reversed(s))
    elif kind == "sort":
        exp = "".join(sorted(s))
    elif kind == "inc":
        exp = "".join(chr((ord(c) - 97 + 1) % 26 + 97) for c in s)
    else:
        pytest.fail(f"unknown rule {kind}")
    assert ans == exp and len(ans) == len(s)
    assert idx == list(range(len(text) - len(s), len(text)))
    return kind, len(s)


CHECKERS = {"pattern": check_pattern, "copy": check_copy, "recall": check_recall, "rule": check_rule}


@pytest.mark.parametrize("task", list(CHECKERS))
def test_generators_correct_under_rule(task):
    rng = random.Random(0)
    for _ in range(300):
        CHECKERS[task](*tasks.TRAIN_GEN[task](rng))


def test_rule_inc_wraps_and_all_kinds_seen():
    rng = random.Random(1)
    kinds = set()
    for _ in range(300):
        kinds.add(check_rule(*tasks.gen_rule(rng))[0])
    assert kinds == {"rev", "sort", "inc"}
    line, mask = tasks.gen_rule(rng, "inc", 3)
    check_rule(line, mask)
    # explicit wrap
    assert tasks._apply_rule("inc", "xyz") == "yza"


def test_make_stream_is_ascii_and_exact_length():
    s = tasks.make_stream(5000, 7)
    assert len(s) == 5000
    s.decode("ascii")
    assert s == tasks.make_stream(5000, 7)
    assert s != tasks.make_stream(5000, 8)


# ---------------------------------------------------------------- (5) held-out regimes

def test_regimes_are_outside_training_distribution():
    rng = random.Random(123)
    periods, lengths = set(), []
    for _ in range(2000):
        p, n = check_pattern(*tasks.gen_pattern(rng))
        periods.add(p)
        lengths.append(n)
    assert periods == {2, 3, 4, 5}
    assert max(lengths) <= 24

    rng = random.Random(5)
    copy_max = max(check_copy(*tasks.gen_copy(rng)) for _ in range(2000))
    assert copy_max <= 12
    recall_max = max(check_recall(*tasks.gen_recall(rng)) for _ in range(2000))
    assert recall_max <= 6
    sort_max, other_max = 0, 0
    for _ in range(2000):
        kind, n = check_rule(*tasks.gen_rule(rng))
        if kind == "sort":
            sort_max = max(sort_max, n)
        other_max = max(other_max, n)
    assert sort_max <= 5 and other_max <= 8

    # held-out regimes
    R = tasks.REGIMES
    rng = random.Random(9)
    for _ in range(300):
        p, _n = check_pattern(*R["pattern"]["unseen_period"](rng))
        assert p in (6, 7)
        _p, n = check_pattern(*R["pattern"]["long"](rng))
        assert n >= 40 > 24
        assert check_copy(*R["copy"]["long"](rng)) > 12
        assert check_recall(*R["recall"]["long"](rng)) > 6
        kind, n = check_rule(*R["rule"]["unseen_combo"](rng))
        assert kind == "sort" and n >= 6 > 5
        kind, n = check_rule(*R["rule"]["long"](rng))
        assert kind in ("rev", "inc") and n > 8


# ---------------------------------------------------------------- (6) make_eval_sets

def _flat(sets):
    return {t: {r: [(d, tuple(m)) for d, m in items] for r, items in regs.items()} for t, regs in sets.items()}


def test_make_eval_sets_deterministic_valid_and_unique():
    a = tasks.make_eval_sets(42, per_regime=40)
    b = tasks.make_eval_sets(42, per_regime=40)
    assert _flat(a) == _flat(b)
    assert _flat(a) != _flat(tasks.make_eval_sets(43, per_regime=40))
    assert set(a) == set(tasks.REGIMES)
    for task, regs in a.items():
        assert set(regs) == set(tasks.REGIMES[task])
        for regime, items in regs.items():
            assert len(items) == 40, (task, regime)
            datas = [d for d, _ in items]
            assert len(set(datas)) == len(datas), f"duplicates in {task}/{regime}"
            for d, m in items:
                CHECKERS[task](d, m)


def test_make_eval_sets_id_excludes_training_lines():
    base = tasks.make_eval_sets(42, per_regime=40)
    exclude = {d for regs in base.values() for d, _ in regs["id"]}
    train = tasks.make_stream(50_000, 1)
    exclude |= {line + b"\n" for line in train.split(b"\n")}
    out = tasks.make_eval_sets(42, per_regime=40, exclude=exclude)
    for task, regs in out.items():
        ids = [d for d, _ in regs["id"]]
        assert len(ids) == 40, task
        assert not (set(ids) & exclude), f"{task} id leaks excluded lines"


# ---------------------------------------------------------------- (7) prepare_tasks round trip

def test_prepare_tasks_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(data, "git_revision", lambda: "test")
    name = "tasks-test"
    manifest = data.prepare_tasks(name, train_bytes=20_000, per_regime=10)
    out_dir = tmp_path / name
    for fn in ("train.bin", "val.bin", "test.bin", "manifest.json", "eval_tasks.json"):
        assert (out_dir / fn).exists(), fn

    on_disk = json.loads((out_dir / "manifest.json").read_text())
    assert on_disk["splits"] == manifest["splits"]
    blobs = {}
    for split in ("train", "val", "test"):
        raw = (out_dir / f"{split}.bin").read_bytes()
        blobs[split] = raw
        assert on_disk["splits"][split]["sha256"] == hashlib.sha256(raw).hexdigest()
        assert on_disk["splits"][split]["sha256"] == data.sha256_file(str(out_dir / f"{split}.bin"))
        assert on_disk["splits"][split]["bytes"] == len(raw)
        t = data.load_split(name, split)
        assert t.dtype.is_floating_point is False and t.numel() == len(raw)
        assert bytes(t.numpy().tobytes()) == raw
    assert len(blobs["train"]) == 20_000
    assert len({blobs["train"], blobs["val"], blobs["test"]}) == 3
    assert blobs["val"][:2000] != blobs["test"][:2000]
    assert data.read_manifest(name)["kind"] == "controlled_tasks"

    ev = data.load_eval_tasks(name)
    assert set(ev) == set(tasks.REGIMES)
    train_lines = {line + b"\n" for line in blobs["train"].split(b"\n")}
    for task, regs in ev.items():
        assert set(regs) == set(tasks.REGIMES[task])
        for regime, items in regs.items():
            assert len(items) == 10
            for d, m in items:
                assert isinstance(d, bytes) and isinstance(m, list)
                assert len(d) == len(m)
                CHECKERS[task](d, m)
                if regime == "id":
                    assert d not in train_lines


def test_load_split_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        data.load_split("nope-v1", "train")


def test_write_splits_manifest_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "git_revision", lambda: "test")
    m = data._write_splits(str(tmp_path / "d"), {"train": b"abc", "val": b"de", "test": b"f"}, {"name": "d"})
    assert m["splits"]["train"] == {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
    assert (tmp_path / "d" / "val.bin").read_bytes() == b"de"


# ---------------------------------------------------------------- natural-v2 (skipped when datasets are not prepared)

_V1, _V2 = os.path.join(data.DATA_ROOT, "natural-v1"), os.path.join(data.DATA_ROOT, "natural-v2")


@pytest.mark.skipif(not (os.path.exists(os.path.join(_V1, "manifest.json")) and os.path.exists(os.path.join(_V2, "manifest.json"))),
                    reason="natural-v1/v2 not prepared")
def test_natural_v2_val_test_identical_and_no_leakage():
    for s in ("val", "test"):
        assert data.sha256_file(os.path.join(_V1, f"{s}.bin")) == data.sha256_file(os.path.join(_V2, f"{s}.bin"))
    assert data.read_manifest("natural-v2")["val_test_identical_to_base"] == {"val": True, "test": True}
    assert data.load_split("natural-v2", "train").numel() > data.load_split("natural-v1", "train").numel()
    keys = {}
    for s in ("train", "val", "test"):
        text = open(os.path.join(_V2, f"{s}.bin"), "rb").read().decode("utf-8")
        keys[s] = {k for p in data._paragraphs(text) if len(k := data._norm_para(p)) >= 60}
    assert not keys["train"] & keys["val"] and not keys["train"] & keys["test"] and not keys["val"] & keys["test"]
    m = data.read_manifest("natural-v2")
    assert {d["id"] for d in m["documents"] if d["split"] == "train"}.isdisjoint(
        {d["id"] for d in m["documents"] if d["split"] != "train"})
