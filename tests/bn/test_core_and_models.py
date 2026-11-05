import pytest
import torch

from bytenoise.core.config import apply_sets, deep_merge, resolve_ref, stable_hash
from bytenoise.core.paths import home
from bytenoise.models import UnsupportedError, build_model, list_models


def test_apply_sets_nested_and_scalar_types():
    out = apply_sets({"train": {"lr": 1}}, ["train.lr=0.5", "train.name=abc", "a.b.c=true"])
    assert out == {"train": {"lr": 0.5, "name": "abc"}, "a": {"b": {"c": True}}}


def test_apply_sets_rejects_missing_equals():
    with pytest.raises(ValueError):
        apply_sets({}, ["nokey"])


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"c": 2}})
    assert base == {"a": {"b": 1}}


def test_hash_is_order_independent():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


@pytest.mark.parametrize("bad", ["../models/debug", "..\\x", "a/b", "", "-x"])
def test_resolve_ref_rejects_paths_from_network(bad):
    with pytest.raises(ValueError):
        resolve_ref("models", bad)


def test_home_must_be_absolute(monkeypatch):
    monkeypatch.setenv("BYTENOISE_HOME", "relative/dir")
    with pytest.raises(RuntimeError):
        home()


def test_registered_families():
    assert {"diffusion", "ar_lm"} <= set(list_models())


@pytest.mark.parametrize("cfg", [{"family": "diffusion", "preset": "debug"}, {"family": "ar_lm", "preset": "tf_debug"}])
def test_adapter_conformance(home, cfg):
    m = build_model({**cfg, "seed": 0})
    assert m.seq_len == 32 and m.num_params() > 0
    batch = torch.randint(0, 256, (4, m.seq_len + (1 if cfg["family"] == "ar_lm" else 0)), dtype=torch.uint8)
    out = m.loss(batch)
    assert torch.isfinite(out.total)
    a = m.generate(b"hello" if "continue" in m.capabilities else b"", 16, seed=3)
    b = m.generate(b"hello" if "continue" in m.capabilities else b"", 16, seed=3)
    assert isinstance(a, bytes) and len(a) == 16 and a == b  # seed-deterministic
    if "score" in m.capabilities:
        s = m.score(torch.randint(0, 256, (3, m.seq_len + 1), dtype=torch.uint8))
        assert s.nll.shape == (3, m.seq_len) and s.exact
    else:
        with pytest.raises(UnsupportedError):
            m.score(torch.zeros(1, 33, dtype=torch.uint8))


def test_diffusion_rejects_prompt_and_overlong(home):
    m = build_model({"family": "diffusion", "preset": "debug"})
    with pytest.raises(UnsupportedError):
        m.generate(b"abc", 8)
    with pytest.raises(ValueError):
        m.generate(b"", m.seq_len + 1)


def test_sampler_validation(home):
    m = build_model({"family": "ar_lm", "preset": "tf_debug"})
    with pytest.raises(ValueError):
        m.generate(b"x", 4, bogus=1)
    with pytest.raises(ValueError):
        m.generate(b"x", 4, temperature=-1)


def test_family_preset_mismatch(home):
    with pytest.raises(ValueError):
        build_model({"family": "ar_lm", "preset": "debug"})
