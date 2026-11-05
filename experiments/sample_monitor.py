"""Sample monitor for a running training job: every --interval seconds, if the run's checkpoint has advanced, load it,
generate --n samples (EMA) and append metrics + decoded text to <run>/samples_log.md (latest also in samples_latest.md).
Runs beside training; loads the checkpoint from disk only, so it never touches the training process."""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import data, eval as ev, train
from gate_text import wordlist, word_hit


def snapshot(run, n, data_name, words, tb, seed, scale):
    ckpt = torch.load(os.path.join(run, train.CHECKPOINT_NAME), map_location="cpu", weights_only=False)
    upd, elapsed = ckpt["progress"]["updates"], ckpt["progress"]["elapsed"]
    del ckpt
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = train.load_model_from_run(run, "ema", dev)
    torch.manual_seed(seed)
    with torch.no_grad():
        ids = model.sample(n).cpu().to(torch.uint8)
    samples = [bytes(r.tolist()) for r in ids]
    m = ev.text_metrics(samples, tb)
    m["word_hit"], _ = word_hit(samples, words)
    del model
    torch.cuda.empty_cache()
    return upd, elapsed, m, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="natural-v1")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--interval", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", type=int, default=4, help="samples written per snapshot")
    a = ap.parse_args()
    tb = data.load_split(a.data, "train").numpy().tobytes()
    words = wordlist(tb)
    last = -1
    while True:
        try:
            ckpt = os.path.join(a.run, train.CHECKPOINT_NAME)
            if os.path.exists(ckpt) and os.path.getmtime(ckpt) > last:
                last = os.path.getmtime(ckpt)
                time.sleep(20)  # let a checkpoint write finish
                upd, el, m, samples = snapshot(a.run, a.n, a.data, words, tb, a.seed, 1.0)
                head = f"## update {upd}  ({el/60:.0f} min training, {time.strftime('%H:%M')})\n" + \
                    " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()) + "\n\n"
                body = "".join(f"```\n{s.decode('utf-8', 'replace')}\n```\n\n" for s in samples[:a.show])
                with open(os.path.join(a.run, "samples_log.md"), "a", encoding="utf-8") as f:
                    f.write(head + body)
                with open(os.path.join(a.run, "samples_latest.md"), "w", encoding="utf-8") as f:
                    f.write(head + body)
                print(f"update {upd}: word_hit={m['word_hit']:.3f} valid_utf8={m['valid_utf8_rate']:.3f}", flush=True)
        except Exception as e:  # keep monitoring through partial-write errors
            print("monitor error:", repr(e), flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
