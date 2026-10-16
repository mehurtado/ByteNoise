"""English-likeness gate: samples from a diffusion run vs real val windows. Word list from TRAIN (count>=50, casefolded)."""
import argparse, collections, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import data, eval as ev, train

WORD = re.compile(rb"[A-Za-z]+")


def wordlist(train_bytes, min_count=50):
    c = collections.Counter(w.lower() for w in WORD.findall(train_bytes))
    return {w for w, n in c.items() if n >= min_count}


def word_hit(samples, words):
    tot = hit = 0
    for s in samples:
        for w in WORD.findall(bytes(s)):
            tot += 1
            hit += w.lower() in words
    return hit / max(tot, 1), tot


def report(name, samples, words, train_bytes):
    m = ev.text_metrics(samples, train_bytes)
    wh, nw = word_hit(samples, words)
    m["word_hit"], m["n_words"] = wh, nw
    print(f"[{name}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="natural-v1")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--which", default="ema")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = train.load_model_from_run(a.run, a.which, dev)
    model.eval()
    L = cfg.model.seq_len
    tb = data.load_split(a.data, "train").numpy().tobytes()
    words = wordlist(tb)
    print(f"wordlist size {len(words)}")
    val = data.load_split(a.data, "val")
    g = torch.Generator().manual_seed(a.seed)
    starts = torch.randint(0, len(val) - L, (a.n,), generator=g)
    real = [bytes(val[s:s + L].numpy().tobytes()) for s in starts]
    report("real_val", real, words, tb)
    torch.manual_seed(a.seed)
    model.sample(1)  # warmup
    torch.cuda.synchronize(); t0 = time.perf_counter()
    ids = model.sample(a.n).cpu().to(torch.uint8)
    torch.cuda.synchronize(); tb_ = time.perf_counter() - t0
    samples = [bytes(r.tolist()) for r in ids]
    torch.cuda.synchronize(); t0 = time.perf_counter()
    model.sample(1)
    torch.cuda.synchronize(); t1 = time.perf_counter() - t0
    report("generated", samples, words, tb)
    print(f"latency: batch{a.n} {tb_:.1f}s total = {tb_/a.n:.2f}s/sample; batch1 {t1:.1f}s")
    for i in range(5):
        print(f"--- sample {i}: {ascii(samples[i][:200])}")


if __name__ == "__main__":
    main()
