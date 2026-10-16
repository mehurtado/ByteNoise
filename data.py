"""Dataset preparation. Each dataset is data/<name>/{train,val,test}.bin (uint8) plus manifest.json."""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
import urllib.request

import numpy as np
import torch

import tasks as T

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "data")

# Project Gutenberg ebook ids (public domain). Title/author/language are read from each file's header and recorded in
# the manifest; entries that fail to download or are not English are skipped and listed in the manifest.
GUTENBERG_IDS = [
    1342, 11, 84, 98, 1661, 2701, 1952, 74, 76, 345, 1232, 174, 1400, 158, 161, 1260, 768, 120, 35, 36,
    1080, 205, 4300, 2554, 5200, 16, 55, 219, 829, 2148, 521, 45, 2591, 3207, 1497, 2009, 244, 215, 135, 996,
    730, 766, 1023, 1399, 2600, 1184, 43, 64, 103, 108, 1257, 2814, 5740, 3825, 41, 46, 19033, 514, 1322, 10,
]


def git_revision():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(data_name, split):
    """uint8 tensor for data/<name>/<split>.bin"""
    path = os.path.join(DATA_ROOT, data_name, f"{split}.bin")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; run: python data.py prepare {data_name.split('-')[0]}")
    return torch.from_numpy(np.fromfile(path, dtype=np.uint8).copy())


def read_manifest(data_name):
    with open(os.path.join(DATA_ROOT, data_name, "manifest.json")) as f:
        return json.load(f)


def _write_splits(out_dir, splits, manifest):
    os.makedirs(out_dir, exist_ok=True)
    manifest["splits"] = {}
    for name, blob in splits.items():
        path = os.path.join(out_dir, f"{name}.bin")
        with open(path, "wb") as f:
            f.write(blob)
        manifest["splits"][name] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    manifest["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["code_revision"] = git_revision()
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# --- Gutenberg natural text ---

def _fetch_gutenberg(book_id, cache_dir):
    path = os.path.join(cache_dir, f"pg{book_id}.txt")
    if not os.path.exists(path):
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        req = urllib.request.Request(url, headers={"User-Agent": "ByteNoise-research/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as f:
            f.write(raw)
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def _parse_gutenberg(text):
    def field(name):
        m = re.search(rf"^{name}:\s*(.+)$", text, re.M)
        return m.group(1).strip() if m else ""

    start = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.S)
    end = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", text)
    body = text[start.end():end.start() if end else len(text)] if start else text
    body = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    return {"title": field("Title"), "author": field("Author"), "language": field("Language")}, body


def _norm_para(p):
    return re.sub(r"\s+", " ", p).strip().lower()


def _paragraphs(body):
    return [p.strip("\n") for p in re.split(r"\n\s*\n", body) if p.strip()]


def assign_splits(doc_ids, val_frac=0.1, test_frac=0.1):
    """Deterministic document-level split: rank documents by hash, tail -> test, before that -> val."""
    ranked = sorted(doc_ids, key=lambda i: hashlib.sha256(f"bytenoise-split-{i}".encode()).hexdigest())
    n = len(ranked)
    n_test = max(1, math.ceil(n * test_frac))
    n_val = max(1, math.ceil(n * val_frac))
    out = {i: "train" for i in ranked}
    for i in ranked[n - n_test:]:
        out[i] = "test"
    for i in ranked[n - n_test - n_val:n - n_test]:
        out[i] = "val"
    return out


def build_text_splits(docs, min_para=60):
    """docs: {id: (meta, body)}. Split by document, then drop duplicate paragraphs within train and any paragraph
    in val/test that also appears in train (or val paragraphs duplicated in test)."""
    split_of = assign_splits(list(docs))
    paras = {s: [] for s in ("train", "val", "test")}
    for doc_id in sorted(docs):
        for p in _paragraphs(docs[doc_id][1]):
            paras[split_of[doc_id]].append((doc_id, p))

    stats = {"paragraphs_in": {s: len(v) for s, v in paras.items()}, "dropped_duplicate": {s: 0 for s in paras}}
    seen = {}
    outputs = {}
    for s in ("train", "val", "test"):
        kept = []
        for doc_id, p in paras[s]:
            key = _norm_para(p)
            if len(key) >= min_para:
                if key in seen:
                    stats["dropped_duplicate"][s] += 1
                    continue
                seen[key] = s
            kept.append(p)
        outputs[s] = ("\n\n".join(kept) + "\n").encode("utf-8")
    return outputs, split_of, stats


def prepare_gutenberg(name="natural-v1", ids=None, cache_dir=None):
    ids = ids or GUTENBERG_IDS
    cache_dir = cache_dir or os.path.join(DATA_ROOT, "raw", "gutenberg")
    docs, skipped = {}, []
    for book_id in ids:
        try:
            meta, body = _parse_gutenberg(_fetch_gutenberg(book_id, cache_dir))
        except Exception as e:
            skipped.append({"id": book_id, "reason": f"download failed: {type(e).__name__}"})
            continue
        if meta["language"] and meta["language"].lower() != "english":
            skipped.append({"id": book_id, "reason": f"language {meta['language']}"})
            continue
        if len(body) < 20000:
            skipped.append({"id": book_id, "reason": "too short"})
            continue
        title_key = re.sub(r"\W+", " ", meta["title"].lower()).strip()
        if title_key in {re.sub(r"\W+", " ", m["title"].lower()).strip() for m, _ in docs.values()}:
            skipped.append({"id": book_id, "reason": f"duplicate title {meta['title']!r}"})
            continue
        docs[book_id] = (meta, body)
    splits, split_of, stats = build_text_splits(docs)
    manifest = {
        "name": name, "kind": "natural_text", "source": "Project Gutenberg (public domain), https://www.gutenberg.org",
        "split_policy": "document-level; rank by sha256('bytenoise-split-<id>'); 10% test, 10% val, rest train; "
                        "paragraph-level exact dedup (whitespace/case-normalised, >=60 chars) so no paragraph appears in two splits",
        "documents": [{"id": i, "title": docs[i][0]["title"], "author": docs[i][0]["author"], "split": split_of[i],
                       "chars": len(docs[i][1])} for i in sorted(docs)],
        "skipped": skipped, "dedup": stats,
    }
    return _write_splits(os.path.join(DATA_ROOT, name), splits, manifest)


# --- Dummy (smoke test) ---

def prepare_dummy(name="dummy-v1", src=os.path.join(ROOT, "dataset", "USE")):
    blob = b""
    for fn in sorted(os.listdir(src)):
        if fn.endswith(".txt"):
            with open(os.path.join(src, fn), "rb") as f:
                blob += f.read()
    a, b = int(len(blob) * 0.9), int(len(blob) * 0.95)
    manifest = {"name": name, "kind": "dummy_smoke_test", "source": src,
                "split_policy": "contiguous 90/5/5 of repeated dummy text; NOT leakage-free, smoke test only"}
    return _write_splits(os.path.join(DATA_ROOT, name), {"train": blob[:a], "val": blob[a:b], "test": blob[b:]}, manifest)


# --- Controlled tasks ---

def prepare_tasks(name="tasks-v1", train_bytes=8_000_000, seed=1234, per_regime=200):
    train = T.make_stream(train_bytes, seed)
    exclude = set(line + b"\n" for line in train.split(b"\n"))
    splits = {"train": train, "val": T.make_stream(500_000, seed + 1), "test": T.make_stream(500_000, seed + 2)}
    eval_sets = T.make_eval_sets(seed + 3, per_regime, exclude=exclude)
    manifest = {"name": name, "kind": "controlled_tasks", "seed": seed, "tasks": list(T.TRAIN_GEN),
                "regimes": {t: list(r) for t, r in T.REGIMES.items()},
                "split_policy": "independent generator seeds per split; eval examples fresh, in-distribution ones exclude training lines"}
    out_dir = os.path.join(DATA_ROOT, name)
    manifest = _write_splits(out_dir, splits, manifest)
    payload = {t: {r: [{"text": d.decode(), "mask": m} for d, m in items] for r, items in regs.items()} for t, regs in eval_sets.items()}
    with open(os.path.join(out_dir, "eval_tasks.json"), "w") as f:
        json.dump(payload, f)
    return manifest


def load_eval_tasks(data_name):
    with open(os.path.join(DATA_ROOT, data_name, "eval_tasks.json")) as f:
        raw = json.load(f)
    return {t: {r: [(e["text"].encode(), e["mask"]) for e in items] for r, items in regs.items()} for t, regs in raw.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("kind", choices=["gutenberg", "dummy", "tasks"])
    a = ap.parse_args(argv)
    fn = {"gutenberg": prepare_gutenberg, "dummy": prepare_dummy, "tasks": prepare_tasks}[a.kind]
    m = fn()
    print(json.dumps({"name": m["name"], "splits": m["splits"], **({"docs": len(m["documents"]), "skipped": len(m["skipped"]), "dedup": m["dedup"]} if "documents" in m else {})}, indent=1))


if __name__ == "__main__":
    main()
