"""Full status of the active (or most recent) ByteNoise training run. Read-only. Usage: status.py [run_name] [--samples N]"""
import argparse, glob, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")


def jl(path, tail_bytes=None):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "rb") as f:
        if tail_bytes:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - tail_bytes))
            f.readline()
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def pick_run(name):
    if name:
        return os.path.join(RUNS, name)
    best, bt = None, 0
    for d in glob.glob(os.path.join(RUNS, "*")):
        m = os.path.join(d, "metrics.jsonl")
        if os.path.basename(d).startswith("_") or not os.path.exists(m):
            continue
        if os.path.getmtime(m) > bt:
            best, bt = d, os.path.getmtime(m)
    return best


def procs(run):
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process | ? { $_.Name -match 'python' } | % { \"$($_.ProcessId)|$($_.CommandLine)\" }"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    key = os.path.basename(run)
    return [l.split("|", 1)[0] for l in out.splitlines() if key in l and "status.py" not in l]


def gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=15).stdout.strip()
    except Exception:
        return "n/a"


def fmt_t(s):
    s = int(s)
    return f"{s//3600}h{(s%3600)//60:02d}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--samples", type=int, default=2)
    a = ap.parse_args()
    run = pick_run(a.run)
    if not run or not os.path.isdir(run):
        print("no run found"); return
    name = os.path.basename(run)
    cfg = json.load(open(os.path.join(run, "config.json"))) if os.path.exists(os.path.join(run, "config.json")) else {}
    meta = json.load(open(os.path.join(run, "meta.json"))) if os.path.exists(os.path.join(run, "meta.json")) else {}
    rows = jl(os.path.join(run, "metrics.jsonl"))
    tr = [r for r in rows if not r.get("eval")]
    ev = [r for r in rows if r.get("eval")]
    m, d, t = cfg.get("model", {}), cfg.get("diffusion", {}), cfg.get("train", {})
    now = time.time()
    mt = os.path.getmtime(os.path.join(run, "metrics.jsonl"))
    pids = procs(run)
    last = tr[-1] if tr else {}
    upd = last.get("update", 0)
    maxu, maxs = t.get("max_updates", 0), t.get("max_seconds", 0)
    done = os.path.exists(os.path.join(run, "ladder_eval.json"))
    age = now - mt
    state = "FINISHED (evaluated)" if done else ("RUNNING" if pids and age < 300 else
            ("PAUSED/STALLED (process alive, no log for %s)" % fmt_t(age) if pids else "NOT RUNNING (stopped or finished training)"))
    print(f"# {name}: {state}")
    print(f"log last written {fmt_t(age)} ago | processes: {','.join(pids) or 'none'} | GPU: {gpu()}")
    print(f"\n## Config")
    print(f"denoiser={m.get('denoiser')} params={meta.get('param_count')} dim={m.get('model_dim')} head={d.get('param')} self_cond={d.get('self_cond')} "
          f"embed_norm={m.get('embed_norm')} t_sampling={d.get('t_sampling')} ce_schedule={d.get('ce_schedule')} span_noise={d.get('span_noise_prob')} tied={m.get('tied_head')}")
    print(f"data={t.get('data')} seq_len={m.get('seq_len')} batch={t.get('batch_size')}x{t.get('grad_accum_steps')} lr={t.get('learning_rate')} "
          f"precision={t.get('precision')} seed={t.get('seed')} rev={meta.get('git_revision')}")
    print(f"\n## Progress")
    if tr:
        el = last.get("elapsed", 0)
        pct = el / maxs if maxs else (upd / maxu if maxu else None)
        rec = tr[-200:]
        ms = (rec[-1]["elapsed"] - rec[0]["elapsed"]) / max(rec[-1]["update"] - rec[0]["update"], 1) * 1000 if len(rec) > 1 else 0
        eta = ""
        if maxs:
            eta = f" | ETA {fmt_t(max(maxs - el, 0))} (time budget {fmt_t(maxs)})"
        elif maxu and ms:
            eta = f" | ETA {fmt_t((maxu - upd) * ms / 1000)}"
        print(f"update {upd}/{maxu or '-'}" + (f" ({pct*100:.1f}%)" if pct is not None else "") +
              f" | trained {fmt_t(el)} | examples {last.get('examples')} | epoch {last.get('epoch')}{eta}")
        print(f"speed {ms:.0f} ms/update | {last.get('bytes_per_s', 0)/1000:.1f} kB/s | peak VRAM {last.get('peak_vram_mb', 0)/1000:.2f} GB | lr {last.get('lr'):.2e}")
        r50 = tr[-50:]
        avg = lambda k: sum(x.get(k, 0) or 0 for x in r50) / len(r50)
        print(f"train loss (last 50 avg) {avg('loss'):.4f} | ce {avg('ce_loss'):.4f} | grad norm {avg('grad_norm'):.3f} | clip frac {avg('clip_fraction'):.2f} | emb_norm {last.get('emb_norm')} | x0_pred_norm {last.get('x0_pred_norm', 0):.2f}")
        bn = last.get("bucket_noise_loss")
        if bn:
            print("train loss by noise bucket (low t -> high t): " + " ".join("n/a" if x is None else f"{x:.4g}" for x in bn))
    print(f"\n## Validation (EMA is what samples use)")
    if ev:
        print("update | raw CE | EMA CE | EMA CE by bucket (low t -> high t)")
        step = max(1, len(ev) // 12)
        for r in ev[::step][-12:] + ([ev[-1]] if ev[-1] not in ev[::step][-12:] else []):
            b = r.get("ema_val_bucket_noise_loss") or []
            print(f"{r['update']:>6} | {r.get('val_ce_loss', float('nan')):.4f} | {r.get('ema_val_ce_loss', float('nan')):.4f} | " + " ".join(f"{x:.4g}" for x in b))
        bestr = min(ev, key=lambda r: r.get("ema_val_ce_loss", 9))
        print(f"best EMA CE {bestr['ema_val_ce_loss']:.4f} @ {bestr['update']}")
        print("note: quick-eval CE draws t/noise per batch; compare runs using ladder_eval's deterministic CE")
    else:
        print("no eval rows yet")
    sl = os.path.join(run, "samples_latest.md")
    if os.path.exists(sl):
        txt = open(sl, encoding="utf-8").read()
        print(f"\n## Latest monitor snapshot (16 samples, noisy)")
        parts = txt.split("```")
        print(parts[0].strip())
        for p in parts[1::2][:a.samples]:
            print("---\n" + p.strip()[:400])
    elog = os.path.join(run, "samples_log.md")
    if os.path.exists(elog):
        pts = []
        cur = None
        for line in open(elog, encoding="utf-8"):
            if line.startswith("## update"):
                cur = line.split()[2]
            elif cur and "word_hit=" in line:
                pts.append((cur, line.split("word_hit=")[1].split()[0]))
                cur = None
        if pts:
            print("\nword_hit trajectory (16 samples): " + ", ".join(f"{u}:{w[:5]}" for u, w in pts[-10:]))
    le = os.path.join(run, "ladder_eval.json")
    if os.path.exists(le):
        j = json.load(open(le))
        print("\n## Final eval (128 samples, 1000 steps, EMA)")
        vc = j.get("val_ce", {})
        print("val CE " + " | ".join(f"{k}: {v['ce']:.4f}" + (f" (x0 acc {v['x0_acc']:.3f})" if 'x0_acc' in v else "") for k, v in vc.items()))
        for k, v in j.get("samplers", {}).items():
            print(f"{k}: " + " ".join(f"{kk}={vv:.4g}" for kk, vv in v.items() if isinstance(vv, (int, float))))
    elif os.path.exists(os.path.join(run, "checkpoint.pt")):
        ck = os.path.join(run, "checkpoint.pt")
        print(f"\ncheckpoint written {fmt_t(now - os.path.getmtime(ck))} ago ({os.path.getsize(ck)/1e6:.0f} MB); final eval not run yet")


if __name__ == "__main__":
    main()
