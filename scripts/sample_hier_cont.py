"""Fixed-seed EMA sample health for a run: 64 samples at s=1.0 (default sampler), seed 1. Prints metrics and writes <run>/samples_ema.txt."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import data, eval as ev, train
from gate_text import wordlist, word_hit

run, n, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64, int(sys.argv[3]) if len(sys.argv) > 3 else 1
tb = data.load_split("natural-v1", "train").numpy().tobytes()
words = wordlist(tb)
model, _ = train.load_model_from_run(run, "ema", "cuda")
torch.manual_seed(seed)
with torch.no_grad():
    ids = model.sample(n).cpu().to(torch.uint8)
samples = [bytes(r.tolist()) for r in ids]
m = ev.text_metrics(samples, tb)
m["word_hit"], _ = word_hit(samples, words)
tot = sum(len(s) for s in samples)
m["space_run8_frac"] = sum(len(x.group()) for s in samples for x in re.finditer(rb" {8,}", s)) / tot
print(json.dumps(m))
with open(os.path.join(run, "samples_ema.txt"), "w", encoding="utf-8") as f:
    for i, s in enumerate(samples):
        f.write(f"--- {i}\n{s.decode('utf-8', 'replace')}\n")
json.dump(m, open(os.path.join(run, "samples_ema_metrics.json"), "w"))
