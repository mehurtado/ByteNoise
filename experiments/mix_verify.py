"""Step-1 verification of the mixture pilot: (A) junk (whitespace-run / table / symbol) prevalence in the TRAIN windows of
each source of a mix; (B) before/after sample health recomputed with junk-like windows excluded, and which repeated 8-grams /
memorised windows are prose. Reads runs/*/mix_samples.json. CPU only.
   python experiments/mix_verify.py --mix mix-pilot-v1 --runs runs/mix_pilot_before runs/mix_pilot_s0 --out runs/mix_verify"""
import argparse, collections, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import data, eval as ev
import mix_filters as mf


def asc(b, n=150):
    return ascii(bytes(b).decode("utf-8", "replace")[:n])


def train_prevalence(mix_name, out):
    mix = json.load(open(os.path.join(data.DATA_ROOT, mix_name, "mix.json"))); z = np.load(os.path.join(data.DATA_ROOT, mix_name, "windows.npz"))
    res, rng = {}, np.random.default_rng(0)
    for s in mix["sources"]:
        tr = np.fromfile(os.path.join(data.DATA_ROOT, s["dataset"], "train.bin"), dtype=np.uint8)
        st = z[f"train_{s['name']}"] - s["ranges"]["train"][0]
        W = tr[st[:, None] + np.arange(256)] if len(st) < 400000 else None
        if W is None:
            W = np.concatenate([tr[st[i:i + 50000, None] + np.arange(256)] for i in range(0, len(st), 50000)])
        r = mf.reasons(W); j = np.zeros(len(W), bool)
        for v in r.values(): j |= v
        sp8 = (W == 32); runs8 = mf._maxrun(sp8) >= 8
        r["only_pipes_or_ws_or_sym"] = j
        res[s["name"]] = dict(windows=int(len(W)), junk=float(j.mean()), reasons={k: float(v.mean()) for k, v in r.items()},
                              any_spacerun8=float(runs8.mean()), nonalnum_frac_mean=float((~np.isin(W, np.frombuffer(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ", np.uint8))).mean()),
                              bytes_in_space_runs8=float(sum(len(m.group()) for m in __import__("re").finditer(rb" {8,}", tr[:20_000_000].tobytes())) / min(len(tr), 20_000_000)),
                              examples={k: [asc(W[i]) for i in np.where(r[k])[0][:: max(1, int(r[k].sum()) // 4 + 1)][:3]] for k in ("ws_run", "pipes", "sym_run") if r[k].any()})
        print(s["name"], {k: round(v, 4) for k, v in res[s["name"]]["reasons"].items()}, flush=True)
    return res


def prose_repeats(samples, topn=12):
    """Repeated 8-grams (position repeats an earlier 8-gram in the same sample), split junk vs prose; most common prose ones."""
    junk_pos = prose_pos = 0; cnt = collections.Counter()
    for s in samples:
        a = np.frombuffer(s, np.uint8)
        if len(a) < 8: continue
        G = np.stack([a[i:i + 8] for i in range(len(a) - 7)])
        bad = mf.WSP[G].sum(1) >= 5; bad |= (G == 124).any(1); bad |= mf.SYMBOL[G].sum(1) >= 3
        seen = set()
        for g, b in zip(G, bad):
            k = g.tobytes()
            if k in seen:
                if b: junk_pos += 1
                else: prose_pos += 1; cnt[k] += 1
            seen.add(k)
    return junk_pos, prose_pos, [(asc(k, 20), c) for k, c in cnt.most_common(topn)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", default="mix-pilot-v1"); ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R = dict(mix=a.mix)
    if not a.skip_train: R["train_prevalence"] = train_prevalence(a.mix, a.out)
    mem = data.load_split(a.mix, "train").numpy().tobytes()
    R["samples"] = {}
    for run in a.runs:
        S = json.load(open(os.path.join(run, "mix_samples.json")))
        for k, hx in S.items():
            samples = [bytes.fromhex(h) for h in hx]
            m = ev.text_metrics(samples, mem)
            blk = dict(valid_utf8=m["valid_utf8_rate"], distinct_4gram=m["distinct_4gram_ratio"], repeated_8gram=m["repeated_8gram_fraction"],
                       mem32=m["memorization_rate"], mem64=ev.memorization_rate(samples, mem, win=64, stride=32))
            blk.update(mf.junk_sample_metrics(samples, mem))
            jp, pp, top = prose_repeats(samples); blk.update(rep_positions_junk=jp, rep_positions_prose=pp, top_prose_repeats=top)
            W = np.stack([np.frombuffer(s[:256].ljust(256, b" "), np.uint8) for s in samples]); jk = mf.is_junk(W)
            blk["samples_with_junk_window"] = int(jk.sum())
            # clean memorised windows (examples)
            ws = [bytes(s[i:i + 32]) for s in samples for i in range(0, len(s) - 31, 16)][:2000]
            Wm = np.stack([np.frombuffer(w, np.uint8) for w in ws]); jm = mf.is_junk(Wm)
            hits = [w for w, j in zip(ws, jm) if (not j) and w in mem]
            blk["clean_mem32_examples"] = [asc(w, 40) for w in hits[:10]]
            blk["clean_mem32_distinct"] = len(set(hits))
            R["samples"][f"{os.path.basename(run.rstrip('/'))}/{k}"] = blk
            print(run, k, {x: (round(v, 4) if isinstance(v, float) else v) for x, v in blk.items() if not isinstance(v, list)}, flush=True)
    json.dump(R, open(os.path.join(a.out, "verify.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
