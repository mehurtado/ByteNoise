"""Plot specs are plain dicts; the UI draws them on canvas, `bn viz` renders the same specs with matplotlib when installed."""
from ..runstore import read_metrics, safe_run_path


def curve_spec(run_id, keys=("loss",), max_points=1500):
    m = read_metrics(safe_run_path(run_id), max_points=max_points)
    missing = [k for k in keys if k not in m["series"]]
    if missing:
        raise KeyError(f"run {run_id} has no series {missing}; available: {sorted(m['series'])}")
    return {"title": run_id, "x": m["steps"], "series": [{"name": k, "y": m["series"][k]} for k in keys]}


def compare_spec(run_ids, key="loss", max_points=800):
    series = []
    for rid in run_ids:
        m = read_metrics(safe_run_path(rid), max_points=max_points)
        if key in m["series"]:
            series.append({"name": rid, "x": m["steps"], "y": m["series"][key]})
    if not series:
        raise KeyError(f"none of the runs has a {key!r} series")
    return {"title": f"{key} across runs", "series": series}


def render_png(spec, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("PNG export needs matplotlib (pip install matplotlib); the web UI renders the same specs without it") from e
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for s in spec["series"]:
        xs = s.get("x", spec.get("x"))
        pts = [(x, y) for x, y in zip(xs, s["y"]) if y is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], label=s["name"], linewidth=1.2)
    ax.set_title(spec["title"])
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
