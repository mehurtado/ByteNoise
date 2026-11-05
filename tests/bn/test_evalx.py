import math

import pytest

from bytenoise.evalx import metrics, report
from bytenoise.evalx.harness import run_suite
from bytenoise.evalx.protocol import load_protocol, make_protocol
from bytenoise.models import build_model


def test_metrics_parity_with_legacy_eval(home):
    from bytenoise.core import legacy
    legacy.ensure()
    import eval as legacy_eval
    samples = [b"hello hello hello hello world", bytes(range(40)), b"\xff\xfe broken utf8", b"abcabcabcabcabcabc" * 3]
    train = b"hello hello hello hello world and more text " * 4
    assert metrics.text_metrics(samples, train) == legacy_eval.text_metrics(samples, train)


def test_ar_quick_protocol_all_modes(home):
    m = build_model({"family": "ar_lm", "preset": "tf_debug", "seed": 0})
    res = run_suite(m, load_protocol("quick"))
    r = res["results"]
    assert math.isclose(r["bpb"]["bpb"], 8.0, abs_tol=0.5)  # untrained byte model ~ log2(256)
    assert r["bpb"]["exact"] is True
    assert set(r["tasks"]["tasks"]) == {"pattern", "copy", "recall", "rule"}
    assert r["generation"]["conditional"] is True and r["generation"]["n_generated"] == 2


def test_diffusion_skips_score_modes_but_generates(home):
    m = build_model({"family": "diffusion", "preset": "debug", "seed": 0})
    r = run_suite(m, load_protocol("quick"))["results"]
    assert "skipped" in r["bpb"] and "skipped" in r["tasks"]
    assert r["generation"]["conditional"] is False


def test_protocol_hash_tracks_spec(home):
    a = load_protocol("quick")
    assert a.hash == load_protocol("quick").hash
    assert load_protocol("quick", data="other").hash != a.hash
    spec = dict(a.spec)
    spec["seed"] = 99
    assert make_protocol(spec).hash != a.hash


def test_protocol_validation():
    with pytest.raises(ValueError):
        make_protocol({"modes": {}})
    with pytest.raises(ValueError):
        make_protocol({"modes": {"bpb": {}}, "bogus": 1, "data": "x"})
    with pytest.raises(ValueError):
        make_protocol({"modes": {"bpb": {}}})


def test_unknown_mode_is_an_error(home):
    m = build_model({"family": "ar_lm", "preset": "tf_debug"})
    with pytest.raises(ValueError):
        run_suite(m, make_protocol({"data": "dummy-v1", "modes": {"nope": {}}}))


def test_report_save_load_compare(home):
    m = build_model({"family": "ar_lm", "preset": "tf_debug", "seed": 0})
    res = run_suite(m, load_protocol("quick"))
    p = report.save(res, "run-a", "ema")
    assert p.parent == home / "var" / "evals" / "run-a"
    back = report.load("run-a", p.name)
    assert back["protocol"]["hash"] == res["protocol"]["hash"]
    table = report.compare({"A": res, "B": back})
    assert "bpb.bpb" in table and "warning" not in table
    other = run_suite(m, load_protocol("quick", data="dummy-v1") if False else make_protocol({**res["protocol"]["spec"], "seed": 7}))
    with pytest.raises(ValueError):
        report.compare({"A": res, "B": other})
    assert "warning" in report.compare({"A": res, "B": other}, force=True)


@pytest.mark.parametrize("run,tag", [("../x", "ema"), ("a", "../e"), ("a b", "ema")])
def test_report_save_rejects_unsafe_names(home, run, tag):
    m = build_model({"family": "ar_lm", "preset": "tf_debug"})
    res = run_suite(m, load_protocol("quick"), only=["bpb"])
    with pytest.raises(ValueError):
        report.save(res, run, tag)
