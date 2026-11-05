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


def test_eval_row_merges_into_its_training_step(home):
    d = home / "runs" / "ev"
    d.mkdir()
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"update": 1, "loss": 3.0}, {"update": 2, "loss": 2.0}, {"update": 2, "eval": True, "val_bpb": 7.0}, {"update": 3, "loss": 1.0}]))
    m = read_metrics(d)
    assert m["steps"] == [1, 2, 3] and m["series"]["loss"] == [3.0, 2.0, 1.0] and m["series"]["eval.val_bpb"] == [None, 7.0, None]


def test_step_falls_back_to_status_when_rows_have_none(home):
    d = home / "runs" / "st"
    d.mkdir()
    (d / "metrics.jsonl").write_text('{"loss": 1.0}\n')
    (d / "status.json").write_text('{"stopped": "max_seconds", "updates": 12}')
    (d / "config.json").write_text('{"train": {"max_updates": 50}}')
    i = describe_run(d)
    assert i.step == 12 and i.status == "timed_out"


def test_remote_unreachable_is_a_clean_error(tmp_path, monkeypatch):
    from bytenoise import remote
    monkeypatch.setenv("BYTENOISE_REMOTES", str(tmp_path / "r.json"))
    remote.add_remote("x", "http://127.0.0.1:1")
    with pytest.raises(RuntimeError, match="cannot reach"):
        remote.call("x", "health", timeout=2)


def test_native_run_killed_hard_is_incomplete(home):
    w = RunWriter.create("k", {"experiment": "e", "model": {"family": "ar_lm"}})
    w.update(status="running", started=time.time() - 7200)
    assert describe_run(w.path).status == "incomplete"


def test_resumed_legacy_run_is_not_reported_done(home):
    d = make_legacy(home, "resumed")
    old = time.time() - 3600
    for f in d.iterdir():
        os.utime(f, (old, old))
    assert describe_run(d).status == "done"
    os.utime(d / "metrics.jsonl", None)  # training resumed: metrics are now newer than status.json
    assert describe_run(d).status == "running"


def test_index_does_not_freeze_running_status_or_checkpoint(home):
    d = make_legacy(home, "live", status=False)
    assert refresh()[0][0].status == "running"
    old = time.time() - 3600
    for f in d.iterdir():
        os.utime(f, (old, old))
    assert refresh()[0][0].status == "incomplete"
    (d / "checkpoint.pt").write_bytes(b"x")
    assert refresh()[0][0].has_checkpoint is True


def test_corrupt_index_is_rebuilt(home, tmp_path):
    from bytenoise.core.paths import cache_dir
    make_legacy(home, "a")
    refresh()
    (cache_dir() / "runs_index.db").write_bytes(b"not a database" * 100)
    assert [i.id for i in refresh()[0]] == ["a"]


def test_nonfinite_metrics_become_null_and_evals_survive_thinning(home):
    d = home / "runs" / "nan"
    d.mkdir()
    rows = [{"update": i, "loss": float("nan") if i == 3 else 1.0, "bucket_noise_loss": [1.0, None]} for i in range(1, 3001)]
    rows += [{"update": 500 * k, "eval": True, "val_bpb": 7.0} for k in range(1, 7)]
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    m = read_metrics(d, max_points=100)
    json.dumps(m, allow_nan=False)
    assert sum(v is not None for v in m["series"]["eval.val_bpb"]) == 6
    assert "bucket_noise_loss.0" in m["series"]
    with pytest.raises(ValueError):
        read_metrics(d, max_points=0)
