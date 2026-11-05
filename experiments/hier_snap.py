"""Snapshot monitor + fixed-seed sample/CE evaluator for the six-hour hierarchy run (runs/hier_mix_6h_s0).
  python experiments/hier_snap.py --once CKPT_DIR --tag t0 --log LOGDIR      # evaluate one checkpoint dir (no stop rules)
  python experiments/hier_snap.py --daemon --run runs/hier_mix_6h_s0 --snaps runs/hier_mix_6h_s0_snaps
Daemon: when <run>/checkpoint.pt reports training elapsed >= 20/40/60 min, then every 45 min, copy it to <snaps>/m<min>/, evaluate
(EMA weights, 64 samples s=1.0, 1000 steps, seeds 1000/1001 = first two batches of mix_eval; per-source val CE on mix-clean-v1 val
windows exactly as mix_eval; text health; personal-text shingle hit COUNTS only), append runs/.../samples_log.md, update state,
apply the stop rules (writes <run>/STOP). Personal text is never printed or stored here (only counts)."""
import argparse, json, os, shutil, sys, time, re
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import torch
import data, train, eval as ev
from gate_text import wordlist
from ladder_eval import make_samples, per_sample_words
from sampler_sweep import uniform_steps
import ngram_score as ns
import mix_filters as mf
from build_mix import shingle_hashes

MIX = "mix-clean-v1"; SCORER = {"gutenberg": "natural-v1"}; PERS = os.path.join(data.DATA_ROOT, "mix-clean-v2", "personal_shingles.npz")
TARGETS = [1200, 2400, 3600] + [3600 + 2700 * k for k in range(1, 7)]


def load_tools(with_scorers=True):
    mix = json.load(open(os.path.join(data.DATA_ROOT, MIX, "mix.json"))); z = np.load(os.path.join(data.DATA_ROOT, MIX, "windows.npz"))
    T = dict(mix=mix, z=z, val=data.load_split(MIX, "val"), mem=data.load_split(MIX, "train").numpy().tobytes(), tools={})
    if with_scorers:
        for s in mix["sources"]:
            tb = data.load_split(s["dataset"], "train").numpy().tobytes()
            T["tools"][s["name"]] = (wordlist(tb), ns.load_or_train(SCORER.get(s["name"], s["dataset"]), verbose=False)); del tb
    if os.path.exists(PERS):
        p = np.load(PERS); T["p32"], T["p64"] = np.unique(p["s32"]), np.unique(p["s64"])
    return T


def pers_hits(samples, T):
    """(#samples with any 32B hit, #samples with any 64B hit, total 64B hit shingles) -- counts only."""
    out = [0, 0, 0]
    for s in samples:
        a = np.frombuffer(s, np.uint8)
        if len(a) >= 32 and len(T["p32"]) and np.isin(shingle_hashes(a, 32), T["p32"]).any(): out[0] += 1
        if len(a) >= 64 and len(T["p64"]):
            n = int(np.isin(shingle_hashes(a, 64), T["p64"]).sum()); out[1] += n > 0; out[2] += n
    return out


@torch.no_grad()
def evaluate(ckpt_dir, T, n=64, n08=32):
    from mix_eval import source_ce
    model, cfg = train.load_model_from_run(ckpt_dir, "ema", "cuda"); model.eval()
    steps = uniform_steps(model.gaussian_diffusion.num_timesteps, 1000); R = dict(ce={}, ce750={})
    for s in T["mix"]["sources"]:
        c = source_ce(model, cfg, T["val"], T["z"][f"val_{s['name']}"]); R["ce"][s["name"]] = c["overall"]["ce"]; R["ce750"][s["name"]] = c["t750_1000"]["ce"]
    t0 = time.time(); samples, sps, peak = make_samples(model, True, 1.0, steps, n); R["gen_s"] = time.time() - t0
    s08, _, _ = make_samples(model, True, 0.8, steps, n08)
    mem = T["mem"]; m = ev.text_metrics(samples, mem)
    R.update(valid_utf8=m["valid_utf8_rate"], distinct_4gram=m["distinct_4gram_ratio"], repeated_8gram=m["repeated_8gram_fraction"],
             mem32=m["memorization_rate"], mem64=ev.memorization_rate(samples, mem, win=64, stride=32),
             space_run8=mf.spacerun_fraction(samples), junk=int(sum(s.count(b"|") > 10 for s in samples)),
             valid_utf8_s08=float(np.mean([ev.valid_utf8(s) for s in s08])), gen_peak_mb=peak)
    R["word_hit"], R["bpb"] = {}, {}
    for name, (words, kn) in T["tools"].items():
        hit, tot = per_sample_words(samples, words); R["word_hit"][name] = float(hit.sum() / max(tot.sum(), 1)); R["bpb"][name] = float(ns.score_bpb(kn, samples).mean())
    R["pers"] = pers_hits(samples, T) if "p32" in T else [0, 0, 0]
    R["pers08"] = pers_hits(s08, T) if "p32" in T else [0, 0, 0]
    R["score"] = score(R)
    del model; torch.cuda.empty_cache()
    return R, samples, s08


def score(R):  # BEST_RULE.md
    wh = float(np.mean(list(R["word_hit"].values()))); ce = float(np.mean(list(R["ce"].values())))
    return R["valid_utf8"] + wh - 5 * R["repeated_8gram"] - 20 * R["mem32"] - 5 * R["space_run8"] - 0.05 * R["junk"] - 2 * ce


def excerpts(samples, k=4, w=150):
    return [ascii(bytes(samples[i][:w]).decode("utf-8", "replace")) for i in range(0, len(samples), max(1, len(samples) // k))][:k]


def md_row(tag, info, R, samples):
    src = list(R["ce"])
    L = [f"## {tag}  ({info})", "",
         "| " + " | ".join(f"CE {s}" for s in src) + " | CE t>=750 | valid_utf8 | rep8 | dist4 | mem32 | mem64 | spc8 | junk | word_hit | bpb | pers(s32,h64,n64) | s0.8 utf8 | score |",
         "|" + "---|" * (len(src) + 13),
         "| " + " | ".join(f"{R['ce'][s]:.4f}" for s in src) + " | " + "/".join(f"{R['ce750'][s]:.3f}" for s in src) +
         f" | {R['valid_utf8']:.3f} | {R['repeated_8gram']:.4f} | {R['distinct_4gram']:.3f} | {R['mem32']:.4f} | {R['mem64']:.4f} | {R['space_run8']:.4f} | {R['junk']} | "
         + "/".join(f"{R['word_hit'][s]:.3f}" for s in src) + " | " + "/".join(f"{R['bpb'][s]:.2f}" for s in src) + f" | {R['pers']} | {R['valid_utf8_s08']:.3f} | {R['score']:.3f} |", ""]
    L += [f"- excerpt {i}: `{e}`" for i, e in enumerate(excerpts(samples))] + [""]
    return "\n".join(L)


def free_gb(p):
    return shutil.disk_usage(p).free / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once"); ap.add_argument("--tag", default="t0"); ap.add_argument("--log")
    ap.add_argument("--daemon", action="store_true"); ap.add_argument("--run"); ap.add_argument("--snaps")
    a = ap.parse_args(); T = load_tools()
    if a.once:
        R, samples, s08 = evaluate(a.once, T); os.makedirs(a.log, exist_ok=True)
        json.dump(R, open(os.path.join(a.log, f"{a.tag}.json"), "w"), indent=1)
        json.dump(dict(s10=[x.hex() for x in samples], s08=[x.hex() for x in s08]), open(os.path.join(a.log, f"{a.tag}_samples.json"), "w"))
        open(os.path.join(a.log, "samples_log.md"), "a", encoding="utf-8").write(md_row(a.tag, a.once, R, samples) + "\n"); print(json.dumps(R)); return
    run, snaps = a.run, a.snaps; os.makedirs(snaps, exist_ok=True)
    stp, lg, ck = os.path.join(run, "state.json"), os.path.join(run, "samples_log.md"), os.path.join(run, "checkpoint.pt")
    st = json.load(open(stp)) if os.path.exists(stp) else dict(done=[], hist=[], flagged=[], best=None, stop=None, pers_bad_prev=False)
    t0 = json.load(open(os.path.join(run, "t0.json"))) if os.path.exists(os.path.join(run, "t0.json")) else None
    last_m, lines_seen, bad_loss = 0.0, 0, 0
    def say(m): print(time.strftime("[%H:%M:%S]"), m, f"free {free_gb(run):.2f}GB", flush=True)
    def stop(why):
        st["stop"] = why; json.dump(st, open(stp, "w"), indent=1); open(os.path.join(run, "STOP"), "w").write(why); say("STOP: " + why)
    while not os.path.exists(os.path.join(run, "status.json")):
        time.sleep(30)
        # numerical watch on the training log
        try:
            tail = open(run + ".log", "rb").read()[-6000:].decode("utf-8", "replace").splitlines()
            ups = [l for l in tail if l.startswith("update ")]
            if any(re.search(r"\b(nan|inf)\b", l, re.I) for l in tail[-30:] if not l.startswith("eval")):
                stop("NaN/Inf in training log"); break
            g = [float(re.search(r"gnorm ([\d.e+-]+)", l).group(1)) for l in ups if "gnorm" in l]; lo = [float(re.search(r"loss ([\d.e+-]+)", l).group(1)) for l in ups]
            if g and max(g[-10:]) > 10: stop(f"gradient-norm explosion {max(g[-10:]):.1f}"); break
            if len(lo) >= 10 and np.mean(lo[-10:]) > 3.0: stop(f"loss blow-up {np.mean(lo[-10:]):.2f}"); break
        except Exception as e:
            pass
        if not os.path.exists(ck) or os.path.getmtime(ck) == last_m or time.time() - os.path.getmtime(ck) < 5: continue
        last_m = os.path.getmtime(ck)
        try:
            prog = torch.load(ck, map_location="cpu", weights_only=False)["progress"]
        except Exception:
            last_m = 0.0; continue
        todo = [t for t in TARGETS if t not in st["done"] and prog["elapsed"] >= t]
        if not todo: continue
        tgt = todo[0]; name = f"m{tgt // 60}"; d = os.path.join(snaps, name); os.makedirs(d, exist_ok=True)
        low = free_gb(run) < 3.0
        if free_gb(run) < 1.5: stop("disk < 1.5 GB"); break
        if low:  # keep only best + latest
            for x in os.listdir(snaps):
                if x not in [st["best"], "latest", name] + st["flagged"]: shutil.rmtree(os.path.join(snaps, x), ignore_errors=True)
            d = os.path.join(snaps, "latest"); os.makedirs(d, exist_ok=True)
        shutil.copy2(ck, os.path.join(d, "checkpoint.pt"))
        info = f"update {prog['updates']}, train elapsed {prog['elapsed']/60:.1f} min" + (" [disk low: latest only]" if low else "")
        say(f"snapshot {name}: {info}"); R, samples, s08 = evaluate(d, T)
        R.update(name=name, target=tgt, updates=prog["updates"], elapsed=prog["elapsed"])
        json.dump(dict(s10=[x.hex() for x in samples], s08=[x.hex() for x in s08]), open(os.path.join(d, "samples.json"), "w"))
        json.dump(R, open(os.path.join(d, "eval.json"), "w"), indent=1)
        flag = R["pers"][1] > 0 or R["pers"][0] > 1 or R["pers08"][1] > 0
        open(lg, "a", encoding="utf-8").write(md_row(f"{name}" + (" **PERSONAL-MATCH FLAG**" if flag else ""), info, R, samples) + "\n")
        st["done"].append(tgt); st["hist"].append(R)
        if flag:
            st["flagged"].append(name)
            if low: os.replace(d, os.path.join(snaps, name))
        H = st["hist"]; why = None
        two = R["pers"][1] > 1 and st["pers_bad_prev"]; st["pers_bad_prev"] = R["pers"][1] > 1
        if two: why = "64-byte personal matches in >1 of 64 samples at two consecutive snapshots"
        if R["mem32"] > 0.02: why = f"memorization32 {R['mem32']:.3f} > 0.02"
        if R["space_run8"] > 0.015: why = f"space-run {R['space_run8']:.3f} > 1.5%"
        if R["junk"] > 2: why = f"table junk {R['junk']}/64 > 2"
        if len(H) >= 4 and all(H[-k]["repeated_8gram"] > H[-k - 1]["repeated_8gram"] for k in (1, 2, 3)) and H[-1]["repeated_8gram"] > 3 * H[0]["repeated_8gram"] and H[0]["repeated_8gram"] > 0 \
                and H[-1]["repeated_8gram"] > 0.01:  # absolute floor: 0.0046 fired the ratio on noise (flat mix_clean_s0 sits at 0.004-0.012)
            why = "repeated_8gram rose 3 consecutive snapshots to >3x first value and above 0.01"
        h1 = next((h for h in H if h["target"] == 3600), None)
        late = [h for h in H if h["target"] > 3600]
        if h1 and len(late) >= 3 and all(h["valid_utf8"] < h1["valid_utf8"] for h in late[-3:]): why = "valid_utf8 below first-hour value for 3 consecutive snapshots"
        eligible = [h for h in H if h["name"] not in st["flagged"]]
        if not flag and R["score"] >= max(h["score"] for h in eligible):
            st["best"] = name
            if low: os.replace(d, os.path.join(snaps, name))
        json.dump(st, open(stp, "w"), indent=1)
        if why: stop(why); break
        # flagged personal-match snapshots are not stop reasons by themselves
    say("monitor done")


if __name__ == "__main__":
    main()
