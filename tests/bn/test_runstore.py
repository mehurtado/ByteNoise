import json
import os
import time

import pytest

from bytenoise.runstore import RunWriter, describe_run, read_metrics, refresh, safe_run_path, scan_runs
from bytenoise.runstore.reader import tail_rows


def make_legacy(home, name, rows=3, status=True, cfg=True):
    d = home / "runs" / name
    d.mkdir(parents=True)
    with open(d / "metrics.jsonl", "w") as f:
        for i in range(1, rows + 1):
            f.write(json.dumps({"update": i, "loss": 5.0 / i, "elapsed": i * 1.5, "bucket_noise_loss": [1.0, 2.0], "eval": False}) + "\n")
        f.write(json.dumps({"update": rows, "eval": True, "val_bpb": 2.5}) + "\n")
    if cfg:
        (d / "config.json").write_text(json.dumps({"arch": "diffusion", "model": {"denoiser": "hybrid"}, "train": {"data": "d1", "max_updates": 9}}))
    if status:
        (d / "status.json").write_text(json.dumps({"stopped": "max_updates", "updates": rows, "elapsed": 4.5}))
    return d


def test_describe_legacy_run(home):
    make_legacy(home, "r1")
    i = describe_run(home / "runs" / "r1")
    assert (i.kind, i.status, i.arch, i.denoiser, i.data) == ("legacy", "done", "diffusion", "hybrid", "d1")
    assert i.step == 3 and i.target_steps == 9 and i.last_loss == pytest.approx(5 / 3)


def test_running_vs_incomplete_by_mtime(home):
    d = make_legacy(home, "live", status=False)
    assert describe_run(d).status == "running"
    old = time.time() - 3600
    for f in d.iterdir():
        os.utime(f, (old, old))
    assert describe_run(d).status == "incomplete"


def test_tail_rows_survives_truncated_last_line(home):
    f = home / "runs" / "m.jsonl"
    f.write_text('{"update": 1}\n{"update": 2}\n{"update": 3, "lo')
    assert [r["update"] for r in tail_rows(f)] == [1, 2]


def test_read_metrics_flattens_and_separates_eval(home):
    d = make_legacy(home, "r2", rows=5)
    m = read_metrics(d)
    assert m["series"]["bucket_noise_loss.1"][0] == 2.0
    assert m["series"]["eval.val_bpb"][-1] == 2.5 and m["series"]["eval.val_bpb"][0] is None
    assert all(len(v) == len(m["steps"]) for v in m["series"].values())


def test_read_metrics_downsamples_keeping_last(home):
    d = make_legacy(home, "big", rows=500)
    m = read_metrics(d, max_points=50)
    assert len(m["steps"]) <= 52 and m["steps"][-1] == 500


@pytest.mark.parametrize("bad", ["../x", "..", ".", "a/b", "a\\b", "", "nonexistent"])
def test_safe_run_path_rejects(home, bad):
    with pytest.raises((ValueError, FileNotFoundError)):
        safe_run_path(bad)


def test_scan_skips_non_runs_and_survives_corrupt(home):
    make_legacy(home, "ok")
    (home / "runs" / "emptydir").mkdir()
    (home / "runs" / "x.out").write_text("log")
    bad = home / "runs" / "corrupt"
    bad.mkdir()
    (bad / "status.json").write_text("{not json")
    infos, skipped = scan_runs()
    assert {i.id for i in infos} >= {"ok"}
    assert dict(skipped)["emptydir"] == "no run files" and dict(skipped)["x.out"] == "not a directory"


def test_index_is_incremental_and_prunes(home):
    make_legacy(home, "a")
    make_legacy(home, "b")
    assert refresh()[1]["reparsed"] == 2
    assert refresh()[1]["reparsed"] == 0
    with open(home / "runs" / "a" / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"update": 99, "loss": 0.1}) + "\n")
    assert refresh()[1]["reparsed"] == 1
    import shutil
    shutil.rmtree(home / "runs" / "b")
    infos, st = refresh()
    assert [i.id for i in infos] == ["a"]


def test_indexing_never_writes_into_runs(home):
    make_legacy(home, "a")
    before = sorted(p.name for p in (home / "runs").rglob("*"))
    refresh()
    assert sorted(p.name for p in (home / "runs").rglob("*")) == before


def test_native_writer_roundtrip(home):
    w = RunWriter.create("My Exp!", {"experiment": "e", "model": {"family": "ar_lm"}})
    assert w.path.parent == home / "runs" and "my-exp" in w.path.name
    w.update(status="running")
    (w.path / "train").mkdir()
    (w.path / "train" / "metrics.jsonl").write_text(json.dumps({"update": 4, "loss": 1.0}) + "\n")
    i = describe_run(w.path)
    assert i.kind == "native" and i.status == "running" and i.family == "ar_lm" and i.step == 4
    assert RunWriter.create("My Exp!", {}).path != w.path
