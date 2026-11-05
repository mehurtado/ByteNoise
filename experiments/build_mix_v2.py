"""Build data/mix-clean-v2 = mix-clean-v1 + a tiny, bounded-exposure 'personal' source (user-authorised writing, long run only).

The personal text is NOT in git: it is read from data/raw/personal-writing/ (gitignored) and written only to data/personal-v1/train.bin
(gitignored *.bin; val/test are empty). data/mix-clean-v2/ and data/personal-v1/ are gitignored as a whole. Nothing is printed.
Design: PASSES tilings of the text, pass k starting at byte offset k*51 (long piece) / k (short piece), 256-byte windows lying wholly
inside one piece, no cross-piece windows, bytes unmodified except a leading UTF-8 BOM dropped. The windows are an 'extra' source
(weight 0) whose slots are carved out of the Gutenberg quota (data.mix_window_starts); mix.json 'slots' sets the virtual epoch to
SLOTS windows (~ the whole 6 h run) so every personal window appears PASSES' times... exactly once per tiling, spread uniformly by the
epoch shuffle. v1's leakage/dedup results are reused unchanged; the personal text only gets a shingle check against every train/val/test
byte of the four sources, and a set of 'distinctive' 32/64-byte shingle hashes (those NOT present in the mixture text) for sample monitoring.
Run once: python experiments/build_mix_v2.py   (refuses to overwrite)."""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import data
from build_mix import shingle_hashes

SRC = os.path.join(ROOT, "data", "raw", "personal-writing")
FILES = ["afarewell.txt", "Speculative Fiction.txt"]
PASSES, L, SLOTS, STEP = 5, 256, 4_600_000, 51


def piece_windows(n, off0):
    """tilings of a piece of n bytes starting at piece offset off0, `PASSES` shifted passes -> list of piece-relative starts"""
    out = []
    for k in range(PASSES):
        o = k * (STEP if n > 2 * L else 1)
        out += [o + j * L for j in range((n - o) // L)]
    return out


def main(out="mix-clean-v2", base="mix-clean-v1", pers="personal-v1"):
    odir, pdir = os.path.join(data.DATA_ROOT, out), os.path.join(data.DATA_ROOT, pers)
    if os.path.exists(odir) or os.path.exists(pdir):
        raise SystemExit("output exists; refusing to overwrite")
    pieces = []
    for f in FILES:
        b = open(os.path.join(SRC, f), "rb").read()
        pieces.append(b[3:] if b.startswith(b"\xef\xbb\xbf") else b)
        pieces[-1].decode("utf-8")
    blob = b"\n\n".join(pieces); offs, o = [], 0
    for p in pieces:
        offs.append(o); o += len(p) + 2
    mix = json.load(open(os.path.join(data.DATA_ROOT, base, "mix.json"))); z = dict(np.load(os.path.join(data.DATA_ROOT, base, "windows.npz")))
    lo = max(s["ranges"]["train"][1] for s in mix["sources"])  # personal train bytes are appended after the last source
    rel = [(offs[i], w) for i, p in enumerate(pieces) for w in piece_windows(len(p), 0)]
    starts = np.array([lo + a + w for a, w in rel], dtype=np.int64)
    # --- shingle check against every byte of every split of the four sources
    h64, h32 = shingle_hashes(np.frombuffer(blob, np.uint8), 64), shingle_hashes(np.frombuffer(blob, np.uint8), 32)
    s64, s32 = np.unique(h64), np.unique(h32); hit64, hit32 = np.zeros(len(s64), bool), np.zeros(len(s32), bool)
    n_dup_win = 0
    for s in mix["sources"]:
        for sp in ("train", "val", "test"):
            t = data.load_split(s["dataset"], sp).numpy()
            for a in range(0, len(t), 1 << 25):
                seg = t[a: a + (1 << 25) + 63]
                for k, ss, hit in ((64, s64, hit64), (32, s32, hit32)):
                    if len(seg) < k: continue
                    h = np.unique(shingle_hashes(seg, k)); i = np.minimum(np.searchsorted(h, ss), len(h) - 1)
                    hit |= h[i] == ss
    # a personal window duplicates corpus text if any of its 64-byte shingles occurs in the corpus
    hs = {int(x): bool(v) for x, v in zip(s64, hit64)}
    for a, w in rel:
        sh = shingle_hashes(np.frombuffer(blob[a + w: a + w + L], np.uint8), 64)
        n_dup_win += any(hs[int(x)] for x in sh)
    os.makedirs(pdir)
    open(os.path.join(pdir, "train.bin"), "wb").write(blob)
    for sp in ("val", "test"): open(os.path.join(pdir, f"{sp}.bin"), "wb").close()
    json.dump(dict(name=pers, kind="personal", note="user-authorised writing; local only, never commit", splits={"train": {"bytes": len(blob)}, "val": {"bytes": 0}, "test": {"bytes": 0}}),
              open(os.path.join(pdir, "manifest.json"), "w"), indent=1)
    vlo = max(s["ranges"]["val"][1] for s in mix["sources"]); tlo = max(s["ranges"]["test"][1] for s in mix["sources"])
    mix["sources"].append(dict(name="personal", dataset=pers, weight=0.0, extra=True, carve_from="gutenberg", passes_over_pool=1,
                               ranges=dict(train=[lo, lo + len(blob)], val=[vlo, vlo], test=[tlo, tlo])))
    mix["slots"] = {"train": SLOTS}
    z["train_personal"] = starts; z["val_personal"] = np.zeros(0, np.int64); z["test_personal"] = np.zeros(0, np.int64)
    os.makedirs(odir)
    np.savez(os.path.join(odir, "windows.npz"), **z)
    json.dump(mix, open(os.path.join(odir, "mix.json"), "w"), indent=1)
    man = json.load(open(os.path.join(data.DATA_ROOT, base, "manifest.json"))); man["name"] = out
    man["splits"]["train"]["bytes"] += len(blob)
    man["note"] = "mix-clean-v1 + bounded-exposure personal source (see experiments/build_mix_v2.py); local only"
    json.dump(man, open(os.path.join(odir, "manifest.json"), "w"), indent=1)
    # distinctive shingles: personal shingles not occurring anywhere in the mixture's text (generic phrases excluded)
    np.savez(os.path.join(odir, "personal_shingles.npz"), s32=s32[~hit32], s64=s64[~hit64])
    st = dict(personal_bytes=len(blob), pieces=len(pieces), windows=int(len(starts)), window_bytes=int(len(starts)) * L, passes=PASSES,
              shingles64=int(len(s64)), shingles64_in_corpus=int(hit64.sum()), shingles32=int(len(s32)), shingles32_in_corpus=int(hit32.sum()),
              windows_with_corpus_64_shingle=int(n_dup_win))
    json.dump(st, open(os.path.join(odir, "personal_stats.json"), "w"), indent=1)
    print(json.dumps(st))


if __name__ == "__main__":
    main()
