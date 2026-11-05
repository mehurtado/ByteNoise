"""Final-run sampler variants with memorization_rate (vs. the run's training set) + decoded samples.
Reuses sample_logits from sampler_sweep_logits (same RNG protocol: seed 0, warmup model.sample(1), then n samples)."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import data, train, eval as ev
from gate_text import wordlist
from sampler_sweep import uniform_steps, word_hit
from sampler_sweep_logits import sample_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/diff_nat_v2_long_s0")
    ap.add_argument("--wordlist-data", default="natural-v1")
    ap.add_argument("--mem-data", default="natural-v2")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.run, "final_sweep.json")
    words = wordlist(data.load_split(a.wordlist_data, "train").numpy().tobytes())
    mem_tb = data.load_split(a.mem_data, "train").numpy().tobytes()
    model, cfg = train.load_model_from_run(a.run, "ema", "cuda"); model.eval()
    T = model.gaussian_diffusion.num_timesteps
    full, u250 = uniform_steps(T, T), uniform_steps(T, 250)
    res = {}
    variants = [("full1000_s1.0", full, 1.0), ("full1000_s0.8", full, 0.8), ("full1000_s0.6", full, 0.6),
                ("uni250_s1.0", u250, 1.0), ("uni250_s0.8", u250, 0.8)]
    torch.manual_seed(a.seed); model.sample(1)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    cid = model.sample(a.n).cpu().to(torch.uint8); dt_c = time.perf_counter() - t0
    for name, steps, s in variants:
        torch.manual_seed(a.seed); model.sample(1)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ids = sample_logits(model, a.n, steps, s)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        samples = [bytes(r.tolist()) for r in ids]
        m = ev.text_metrics(samples, mem_tb)
        m["word_hit"], m["n_words"] = word_hit(samples, words)
        m.update(sec_per_sample=dt / a.n, nsteps=len(steps), s=s)
        if name == "full1000_s1.0":
            m["matches_model_sample_control"] = bool((ids == cid).all()); m["control_sec_per_sample"] = dt_c / a.n
        m["samples_ascii"] = [ascii(x[:250]) for x in samples[:5]]
        res[name] = m
        print(name, {k: v for k, v in m.items() if k != "samples_ascii"}, flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
