"""Task-aware validity of free-running diffusion samples on tasks-v1.

Reports alphabet fraction, valid-prefix fraction and fully-correct fraction of complete lines, plus the alphabet
fraction of the x0-prediction at t=T-1 from pure noise. Checkers are validated on real val.bin lines.
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import data
import train
from tasks import _apply_rule

ALPHA = set(b"abcdefghijklmnopqrstuvwxyz0123456789:|?=\n")
PREFIXES = ("p:", "c:", "r:", "v:rev:", "v:sort:", "v:inc:")
LOW = "abcdefghijklmnopqrstuvwxyz"


def check_line(s):
    """True iff s (no newline) is a fully correct task line."""
    if s.startswith("p:"):
        b = s[2:]
        if not b or not re.fullmatch(r"[a-z]+", b) or len(b) < 6:
            return False
        for p in (2, 3, 4, 5):  # smallest generator period (2..5) such that b is periodic with it
            if len(b) >= 3 * p and all(b[i] == b[i % p] for i in range(len(b))):
                return True
        return False
    if s.startswith("c:"):
        m = re.fullmatch(r"c:([a-z]+)\|([a-z]+)", s)
        return bool(m) and m.group(1) == m.group(2)
    if s.startswith("r:"):
        m = re.fullmatch(r"r:(?:([a-z])(\d))+\?([a-z])=(\d)", s)
        if not m:
            return False
        body, k, v = re.fullmatch(r"r:([a-z0-9]+)\?([a-z])=(\d)", s).groups()
        pairs = [(body[i], body[i + 1]) for i in range(0, len(body), 2)]
        keys = [p[0] for p in pairs]
        return len(set(keys)) == len(keys) and dict(pairs).get(k) == v
    m = re.fullmatch(r"v:(rev|sort|inc):([a-z]+)=([a-z]+)", s)
    return bool(m) and _apply_rule(m.group(1), m.group(2)) == m.group(3)


def split_lines(b):
    """(complete interior lines, all fragments). First fragment is a window cut and the last is unterminated."""
    parts = b.decode("latin-1").split("\n")
    return parts[1:-1], parts


def stats(samples):
    tot = alpha = 0
    lines, pre, ok = 0, 0, 0
    for b in samples:
        tot += len(b)
        alpha += sum(c in ALPHA for c in b)
        complete, _ = split_lines(b)
        for ln in complete:
            lines += 1
            pre += ln.startswith(PREFIXES)
            ok += check_line(ln)
    return dict(alpha=alpha / max(tot, 1), prefix=pre / max(lines, 1), correct=ok / max(lines, 1), lines=lines)


def validate_checkers(data_name, n_bytes=200000):
    val = data.load_split(data_name, "val")[:n_bytes].numpy().tobytes()
    lines = val.decode("latin-1").split("\n")[1:-1]
    ok = sum(check_line(l) for l in lines)
    bad = [l for l in lines if not check_line(l)][:5]
    # negatives: corrupt each line's last char, should mostly fail
    neg = sum(check_line(l[:-1] + ("z" if l[-1] != "z" else "y")) for l in lines)
    return ok / len(lines), len(lines), neg / len(lines), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", default="tasks-v1")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--which", default="ema")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate", action="store_true", help="only validate checkers on val.bin")
    a = ap.parse_args()
    acc, nl, neg, bad = validate_checkers(a.data)
    print(f"[checker] val lines pass={acc:.4f} (n={nl}); last-char-corrupted pass={neg:.4f}; failing examples={bad}")
    if a.validate:
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = train.load_model_from_run(a.run, a.which, dev)
    model.eval()
    torch.manual_seed(a.seed)
    with torch.no_grad():
        ids = model.sample(a.n).cpu().to(torch.uint8)
        samples = [bytes(r.tolist()) for r in ids]
        # first step: x0-prediction from pure noise at t=T-1
        gd = model.gaussian_diffusion
        T = gd.num_timesteps
        L, E = model.cfg.model.seq_len, model.cfg.model.byte_embed_dim
        torch.manual_seed(a.seed)
        xt = torch.randn(a.n, L, E, device=dev)
        t = torch.full((a.n,), T - 1, device=dev, dtype=torch.long)
        pred = gd.denoise_model(xt, t)
        x0 = gd.predict_start_from_noise(xt, t, pred)
        pb = model.get_logits(x0).argmax(-1).cpu().to(torch.uint8)
    s = stats(samples)
    p = sum(sum(c in ALPHA for c in r.tolist()) for r in pb) / pb.numel()
    name = os.path.basename(os.path.normpath(a.run))
    print(f"{'run':<16}{'alpha_frac':>11}{'prefix_ok':>10}{'line_correct':>13}{'lines':>6}{'x0@999_alpha':>14}")
    print(f"{name:<16}{s['alpha']:>11.3f}{s['prefix']:>10.3f}{s['correct']:>13.3f}{s['lines']:>6}{p:>14.3f}")
    for i in range(3):
        print(f"sample{i}: {samples[i][:100]!r}")
    print(f"x0@999 : {bytes(pb[0].tolist())[:100]!r}")


if __name__ == "__main__":
    main()
