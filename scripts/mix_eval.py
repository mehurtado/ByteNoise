"""Per-source evaluation of a diffusion run on the mix-pilot-v1 sources (EMA weights). Identical settings for every run.

1. Deterministic val CE per source: FIXED windows (evenly spaced over the source's document-safe val windows), fixed t draws and
   fixed noise (generator seeded 0 per source and region), overall and t>=750 -> comparable before/after training.
2. Fixed-seed samples (batch seeds 1000+b, 128 samples, 1000 steps) at s=1.0 and s=0.8, scored per source with that source's
   TRAIN word list and byte-level Kneser-Ney scorer (Gutenberg: word list from natural-v2 train, scorer natural-v1 train, a
   subset of v2 train that was already cached), plus real val windows of each source as the reference. Memorization (32- and
   64-byte windows), repeated_8gram, distinct_4gram, valid_utf8 are measured against the whole mixture train text.
Writes <out>/mix_eval.json, mix_eval.md, mix_samples.json.   python scripts/mix_eval.py --run runs/X [--out DIR]
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import numpy as np
import torch
import data, train, eval as ev
from gate_text import wordlist
from ladder_eval import make_samples, per_sample_words, boot_ci
from sampler_sweep import uniform_steps
import ngram_score as ns

MIX = "mix-pilot-v1"
SCORER = {"gutenberg": "natural-v1"}  # scorer training data when it differs from the source dataset


@torch.no_grad()
def source_ce(model, cfg, tokens, starts, n=512, bs=32, seed=0):
    T = cfg.diffusion.num_timesteps
    idx = np.linspace(0, len(starts) - 1, min(n, len(starts))).astype(int)
    st = torch.as_tensor(starts[idx]); L = train.window_len(cfg); ar = torch.arange(L)
    res = {}
    for name, lo, hi in (("overall", 0, T), ("t750_1000", 750 * T // 1000, T)):
        g = torch.Generator().manual_seed(seed)
        ce = acc = 0.0
        for i in range(0, len(st), bs):
            x = tokens[st[i:i + bs].unsqueeze(-1) + ar].long().to("cuda")
            t = torch.randint(lo, hi, (x.size(0),), generator=g).to("cuda")
            noise = torch.randn(x.size(0), x.size(1), cfg.model.byte_embed_dim, generator=g).to("cuda")
            _, s = model(x, t=t, noise=noise)
            ce += float(s["ce_loss"]) * x.size(0); acc += float(s["x0_acc"]) * x.size(0)
        res[name] = dict(ce=ce / len(st), x0_acc=acc / len(st))
    res["n_windows"] = len(st)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args()
    out = a.out or a.run
    os.makedirs(out, exist_ok=True)
    model, cfg = train.load_model_from_run(a.run, "ema", "cuda"); model.eval()
    gd = model.gaussian_diffusion; T = gd.num_timesteps; assert gd.param == "logits"
    mix = json.load(open(os.path.join(data.DATA_ROOT, MIX, "mix.json")))
    z = np.load(os.path.join(data.DATA_ROOT, MIX, "windows.npz"))
    val = data.load_split(MIX, "val"); L = cfg.model.seq_len
    R = dict(run=a.run, mix=MIX, ce={}, samplers={}, real={})
    for s in mix["sources"]:
        R["ce"][s["name"]] = source_ce(model, cfg, val, z[f"val_{s['name']}"])
        print(a.run, s["name"], R["ce"][s["name"]], flush=True)
    mem = data.load_split(MIX, "train").numpy().tobytes()
    tools = {}
    for s in mix["sources"]:
        tb = data.load_split(s["dataset"], "train").numpy().tobytes()
        tools[s["name"]] = (wordlist(tb), ns.load_or_train(SCORER.get(s["name"], s["dataset"]), verbose=False)); del tb

    def score(samples):
        blk = {}
        for name, (words, kn) in tools.items():
            hit, tot = per_sample_words(samples, words); bpb = ns.score_bpb(kn, samples)
            blk[name] = dict(word_hit=boot_ci(hit, tot)[0], ngram_bpb=float(bpb.mean()))
        return blk

    g = np.random.default_rng(0)
    for s in mix["sources"]:  # real reference windows per source
        pool = z[f"val_{s['name']}"]; pick = pool[g.choice(len(pool), a.n, replace=False)]
        real = [val[p:p + L].numpy().tobytes() for p in pick]
        R["real"][s["name"]] = score(real)[s["name"]]
    samples_out = {}
    for sc in (1.0, 0.8):
        samples, sps, _ = make_samples(model, True, sc, uniform_steps(T, T), a.n)
        m = ev.text_metrics(samples, mem)
        blk = dict(valid_utf8=m["valid_utf8_rate"], distinct_4gram=m["distinct_4gram_ratio"], repeated_8gram=m["repeated_8gram_fraction"],
                   memorization32=m["memorization_rate"], memorization64=ev.memorization_rate(samples, mem, win=64, stride=32),
                   per_source=score(samples))
        R["samplers"][f"s{sc}"] = blk; samples_out[f"s{sc}"] = [x.hex() for x in samples]
        print(a.run, sc, blk, flush=True)
    json.dump(R, open(os.path.join(out, "mix_eval.json"), "w"), indent=1)
    json.dump(samples_out, open(os.path.join(out, "mix_samples.json"), "w"))
    open(os.path.join(out, "mix_eval.md"), "w", encoding="utf-8").write(to_md(R))
    print("wrote", out)


def to_md(R):
    names = list(R["ce"])
    L = [f"# mix_eval: {R['run']}", "", "| source | val CE overall | CE t>=750 | x0 acc t>=750 |", "|---|---|---|---|"]
    for n in names:
        c = R["ce"][n]; L.append(f"| {n} | {c['overall']['ce']:.4f} | {c['t750_1000']['ce']:.4f} | {c['t750_1000']['x0_acc']:.3f} |")
    L += ["", "| set | " + " | ".join(f"{n} word_hit / bpb" for n in names) + " |", "|---|" + "---|" * len(names)]
    L.append("| real val | " + " | ".join(f"{R['real'][n]['word_hit']:.3f} / {R['real'][n]['ngram_bpb']:.2f}" for n in names) + " |")
    for k, b in R["samplers"].items():
        L.append(f"| samples {k} | " + " | ".join(f"{b['per_source'][n]['word_hit']:.3f} / {b['per_source'][n]['ngram_bpb']:.2f}" for n in names) + " |")
    L += ["", "| samples | valid_utf8 | distinct_4gram | repeated_8gram | memorization 32B | memorization 64B |", "|---|---|---|---|---|---|"]
    for k, b in R["samplers"].items():
        L.append(f"| {k} | {b['valid_utf8']:.3f} | {b['distinct_4gram']:.3f} | {b['repeated_8gram']:.4f} | {b['memorization32']:.4f} | {b['memorization64']:.4f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
