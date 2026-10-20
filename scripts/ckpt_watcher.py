"""Copy intermediate checkpoints of a running job: whenever <run>/checkpoint.pt changes, read its update counter and, if it is
one of --at, copy it to <dest>/u<updates:05d>/checkpoint.pt (skips when free disk < --min-free GB). Stops when <run>/status.json exists."""
import argparse, os, shutil, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def free_gb(p):
    return shutil.disk_usage(p).free / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--dest", required=True)
    ap.add_argument("--at", type=int, nargs="+", default=[500, 1000, 2000, 3000]); ap.add_argument("--min-free", type=float, default=3.5)
    a = ap.parse_args(); ck = os.path.join(a.run, "checkpoint.pt"); last = 0.0; got = set()
    while True:
        if os.path.exists(ck) and os.path.getmtime(ck) != last and time.time() - os.path.getmtime(ck) > 3:
            last = os.path.getmtime(ck)
            try:
                u = torch.load(ck, map_location="cpu", weights_only=False)["progress"]["updates"]
                if u in a.at and u not in got:
                    if free_gb(a.run) < a.min_free:
                        print(f"skip {u}: low disk", flush=True)
                    else:
                        d = os.path.join(a.dest, f"u{u:05d}"); os.makedirs(d, exist_ok=True)
                        shutil.copy2(ck, os.path.join(d, "checkpoint.pt")); got.add(u); print(f"saved {u}", flush=True)
            except Exception as e:
                print("retry", e, flush=True); last = 0.0
        if os.path.exists(os.path.join(a.run, "status.json")):
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
