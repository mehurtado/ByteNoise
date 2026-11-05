"""`bn`: one entry point. Every UI action is one of these commands."""
import argparse
import json
import sys

from . import remote


def _out(obj):
    print(json.dumps(obj, indent=1, ensure_ascii=False))


def cmd_runs(a):
    from .runstore import refresh, read_metrics, safe_run_path, describe_run
    if a.what == "ls":
        infos, stats = refresh()
        infos.sort(key=lambda i: i.mtime, reverse=True)
        for i in infos[: a.limit]:
            step = "-" if i.step is None else (f"{i.step}/{i.target_steps}" if i.target_steps else i.step)
            print(f"{i.id:60} {i.status:10} {(i.arch + '/' + i.denoiser).strip('/') or '-':18} {i.data or '-':22} {step!s:>14} {i.last_loss if i.last_loss is not None else '-'}")
        print(f"# {stats}", file=sys.stderr)
    elif a.what == "show":
        _out(describe_run(safe_run_path(a.run)).to_dict())
    else:
        m = read_metrics(safe_run_path(a.run), max_points=a.points)
        _out({"steps": m["steps"], "series": {k: v for k, v in m["series"].items() if not a.keys or k in a.keys}})


def cmd_models(a):
    from .models.registry import get_model_class, list_models
    _out([{"family": f, "capabilities": sorted(get_model_class(f).capabilities), "sampler": get_model_class(f).sampler_schema()} for f in list_models()])


def cmd_eval(a):
    from .evalx import report
    if a.what == "compare":
        res = {r: report.load(r, f"{a.protocol}@{a.which}.json") for r in a.runs}
        print(report.compare(res, force=a.force))
        return
    from .evalx.harness import run_suite
    from .evalx.protocol import load_protocol
    from .models.loading import load_run_model
    from .runstore import safe_run_path
    proto = load_protocol(a.protocol, a.data, allow_path=True)
    p = safe_run_path(a.run)
    model = load_run_model(p, a.which, a.device)
    res = run_suite(model, proto, a.device, only=a.modes or None)
    path = report.save(res, p.name, a.which)
    print(report.to_markdown(res, p.name))
    print(f"saved {path}", file=sys.stderr)


def cmd_gen(a):
    from .models.loading import load_run_model
    from .runstore import safe_run_path
    m = load_run_model(safe_run_path(a.run), a.which, a.device)
    sampler = dict(kv.split("=", 1) for kv in a.sampler)
    sys.stdout.buffer.write(m.generate(a.prompt.encode(), a.n, seed=a.seed, **sampler) + b"\n")


def cmd_train(a):
    if a.detach:
        from .serve import jobs
        job = jobs.submit(a.experiment, a.sets, a.device)
        blocked = jobs.schedule()
        _out({**jobs.get(job["id"]), "blocked": blocked})
        return
    from .train.runner import run_experiment
    ids = run_experiment(a.experiment, a.sets, a.device, allow_path=True)
    _out({"runs": ids})


def cmd_serve(a):
    from .serve.agent import serve
    serve(a.host, a.port, a.token, a.allow_host, [u for u in a.tailnet_users])


def cmd_data(a):
    from .datasets.store import list_datasets
    _out(list_datasets())


def cmd_viz(a):
    from . import viz
    spec = viz.compare_spec(a.runs, a.key) if len(a.runs) > 1 else viz.curve_spec(a.runs[0], a.key.split(","))
    if a.out:
        print(viz.render_png(spec, a.out))
    else:
        _out(spec)


def cmd_remote(a):
    if a.what == "add":
        remote.add_remote(a.name, a.url, a.token)
        print(f"saved remote {a.name}")
    elif a.what == "ls":
        _out({k: {"url": v["url"]} for k, v in remote.load_remotes().items()})
    elif a.what == "train":
        _out(remote.call(a.name, "jobs", {"experiment": a.arg, "sets": a.sets, "device": a.device}))
    elif a.what == "jobs":
        _out(remote.call(a.name, "jobs"))
    elif a.what == "log":
        print(remote.call(a.name, f"jobs/{a.arg}/log")["log"])
    elif a.what == "stop":
        _out(remote.call(a.name, f"jobs/{a.arg}/stop", {}))
    elif a.what == "runs":
        _out(remote.call(a.name, "runs"))
    elif a.what == "gpu":
        _out(remote.call(a.name, "gpu"))


def build_parser():
    ap = argparse.ArgumentParser(prog="bn")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("runs")
    r.add_argument("what", choices=["ls", "show", "metrics"])
    r.add_argument("run", nargs="?")
    r.add_argument("--limit", type=int, default=40)
    r.add_argument("--points", type=int, default=200)
    r.add_argument("--keys", nargs="*")
    r.set_defaults(fn=cmd_runs)

    sub.add_parser("models").set_defaults(fn=cmd_models)

    e = sub.add_parser("eval")
    e.add_argument("what", choices=["run", "compare"])
    e.add_argument("runs", nargs="+")
    e.add_argument("--protocol", required=True)
    e.add_argument("--data")
    e.add_argument("--which", default="ema", choices=["ema", "raw"])
    e.add_argument("--device", default="cpu")
    e.add_argument("--modes", nargs="*")
    e.add_argument("--force", action="store_true")
    e.set_defaults(fn=lambda a: cmd_eval(_single_run(a)))

    g = sub.add_parser("gen")
    g.add_argument("run")
    g.add_argument("--prompt", default="")
    g.add_argument("-n", type=int, default=128)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--which", default="ema", choices=["ema", "raw"])
    g.add_argument("--device", default="cpu")
    g.add_argument("--sampler", nargs="*", default=[], metavar="K=V")
    g.set_defaults(fn=cmd_gen)

    t = sub.add_parser("train")
    t.add_argument("experiment")
    t.add_argument("--set", action="append", default=[], dest="sets")
    t.add_argument("--device", default="cpu")
    t.add_argument("--detach", action="store_true", help="queue as a background job (same queue as the agent)")
    t.set_defaults(fn=cmd_train)

    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--token")
    s.add_argument("--allow-host", action="append", default=[], help="extra Host header value, e.g. desktop-667n4qr.tailb3695b.ts.net")
    s.add_argument("--tailnet-users", action="append", default=[], help="accept this Tailscale-User-Login from loopback (tailscale serve)")
    s.set_defaults(fn=cmd_serve)

    sub.add_parser("data").set_defaults(fn=cmd_data)

    v = sub.add_parser("viz")
    v.add_argument("runs", nargs="+")
    v.add_argument("--key", default="loss")
    v.add_argument("--out")
    v.set_defaults(fn=cmd_viz)

    rm = sub.add_parser("remote")
    rm.add_argument("what", choices=["add", "ls", "train", "jobs", "log", "stop", "runs", "gpu"])
    rm.add_argument("name", nargs="?")
    rm.add_argument("arg", nargs="?", help="url (add), experiment (train), job id (log/stop)")
    rm.add_argument("--token")
    rm.add_argument("--set", action="append", default=[], dest="sets")
    rm.add_argument("--device", default="cpu")
    rm.set_defaults(fn=lambda a: cmd_remote(_remote_args(a)))
    return ap


def _single_run(a):
    if a.what == "run":
        if len(a.runs) != 1:
            raise SystemExit("eval run takes exactly one run")
        a.run = a.runs[0]
    return a


def _remote_args(a):
    if a.what == "add":
        a.url, a.arg = a.arg, None
        if not a.name or not a.url:
            raise SystemExit("usage: bn remote add NAME URL [--token T]")
    elif a.what != "ls" and not a.name:
        raise SystemExit("remote name required")
    return a


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        a.fn(a)
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
