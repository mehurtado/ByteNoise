import json

import pytest

from bytenoise.core import legacy
from bytenoise.runstore import describe_run, read_metrics
from bytenoise.train.experiment import load_experiment
from bytenoise.train.runner import run_experiment


def test_experiment_resolution_and_stage_inheritance(home):
    exp = load_experiment("debug_smoke")
    s0, s1 = exp.stages
    assert s0.config["train"]["data"] == "dummy-v1" and s0.config["train"]["max_updates"] == 10
    assert s1.config["train"]["max_updates"] == 6 and s1.config["train"]["learning_rate"] == 1e-4
    assert s1.config["model"] == s0.config["model"] and s1.init_from == "previous"
    assert s0.config_hash != s1.config_hash


def test_set_overrides_win_and_unknown_fields_fail(home):
    assert load_experiment("debug_smoke", ["train.max_updates=4"]).stages[0].config["train"]["max_updates"] == 4
    with pytest.raises(ValueError):
        load_experiment("debug_smoke", ["train.not_a_field=1"])


def test_bad_experiment_keys(home, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: debug\nbogus: 1\n")
    with pytest.raises(ValueError):
        load_experiment(str(bad), allow_path=True)
    bad.write_text("model: debug\nstages: [{init_from: previous}]\n")
    with pytest.raises(ValueError):
        load_experiment(str(bad), allow_path=True)


def test_two_stage_run_and_golden_match_with_legacy_trainer(home, tmp_path):
    ids = run_experiment("debug_smoke", device="cpu")
    assert len(ids) == 2
    infos = [describe_run(home / "runs" / i) for i in ids]
    assert [i.status for i in infos] == ["done", "done"] and infos[1].parent == ids[0]
    assert all((home / "runs" / i / "run.json").exists() and (home / "runs" / i / "events.jsonl").exists() for i in ids)
    assert infos[0].step == 10 and infos[1].step == 6

    # golden: stage 0 must reproduce the original trainer exactly (CPU, same seed)
    legacy.ensure()
    import train as lt
    from config import Config
    cfg = Config.from_dict(json.loads((home / "runs" / ids[0] / "config.resolved.json").read_text()))
    direct = tmp_path / "direct"
    lt.run_training(cfg, str(direct), mode="fresh", device="cpu", quiet=True)
    a = read_metrics(home / "runs" / ids[0])["series"]["loss"]
    b = read_metrics(direct)["series"]["loss"]
    assert len(a) == len(b) and a == pytest.approx(b, rel=1e-5)


def test_failed_stage_is_recorded(home, tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("model: debug\nrecipe: smoke\ndata: does-not-exist\n")
    with pytest.raises(FileNotFoundError):
        run_experiment(str(p), device="cpu", allow_path=True)
    runs = sorted((home / "runs").iterdir())
    assert len(runs) == 1 and json.loads((runs[0] / "run.json").read_text())["status"] == "failed"


def test_init_from_unknown_run_rejected(home, tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("model: debug\nrecipe: smoke\ndata: dummy-v1\nstages: [{name: a, init_from: ../etc}]\n")
    with pytest.raises(ValueError):
        run_experiment(str(p), device="cpu", allow_path=True)
