"""`python -m bytenoise.train run <experiment>`: the process the agent spawns for each job."""
import argparse
import json
import sys
import time

from ..serve import jobs
from .runner import run_experiment


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bytenoise.train")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("experiment")
    r.add_argument("--set", action="append", default=[], dest="sets")
    r.add_argument("--device", default="auto")
    r.add_argument("--job-id")
    r.add_argument("--allow-path", action="store_true")
    a = ap.parse_args(argv)
    code = 0
    try:
        run_experiment(a.experiment, a.sets, a.device, a.allow_path,
                       on_run=(lambda rid: jobs.record_run(a.job_id, rid)) if a.job_id else None)
    except BaseException as e:
        code = 1
        print(json.dumps({"error": f"{type(e).__name__}: {e}", "t": time.time()}), file=sys.stderr, flush=True)
        raise
    finally:
        if a.job_id:
            jobs.finish(a.job_id, code)


if __name__ == "__main__":
    main()
