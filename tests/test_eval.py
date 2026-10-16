import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import data
import eval as E
from main import ByteDiffusionLM
from models_lm import ByteLM


@pytest.fixture
def ds(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA_ROOT", str(tmp_path))
    d = tmp_path / "tiny"
    d.mkdir()
    g = torch.Generator().manual_seed(0)
    for s in ("train", "val", "test"):
        (d / f"{s}.bin").write_bytes(bytes(torch.randint(97, 123, (3000,), generator=g).tolist()))
    items = [{"text": "ab=cd;ef", "mask": [0, 0, 0, 1, 1, 0, 0, 0]}, {"text": "xy=zw;qq", "mask": [0, 0, 0, 1, 1, 0, 0, 0]},
             {"text": "k=v", "mask": [0, 0, 1]}]
    (d / "eval_tasks.json").write_text(json.dumps({"copy": {"id": items}}))
    return "tiny"


def lm(preset):
    torch.manual_seed(0)
    return ByteLM(config.get_preset(preset)).eval()


@pytest.mark.parametrize("preset", ["tf_debug", "rec_debug"])
def test_bpb_near_8_and_deterministic(ds, preset):
    m = lm(preset)
    a, b = E.eval_lm(m, ds), E.eval_lm(m, ds)
    assert abs(a["bpb"] - 8) < 1
    assert a == b
    assert len(a["bpb_by_context"]) == 4 if m.lm.seq_len > 256 else len(a["bpb_by_context"]) >= 1
    assert E.eval_lm(m, ds, max_windows=2)["windows"] == 2


def test_bpb_four_buckets(ds):
    cfg = config.get_preset("tf_debug")
    from dataclasses import replace
    m = ByteLM(replace(cfg, lm=replace(cfg.lm, seq_len=300))).eval()
    assert len(E.eval_lm(m, ds)["bpb_by_context"]) == 4


class Oracle(torch.nn.Module):
    """Logits are a one-hot of the true next byte, looked up from known examples."""

    def __init__(self, seqs, lie=False):
        super().__init__()
        self.lm, self.seqs, self.lie = config.LMConfig(seq_len=64), seqs, lie

    def logits(self, idx):
        out = torch.zeros(*idx.shape, 256)
        for r in range(idx.size(0)):
            row = bytes(idx[r].tolist())
            s = next(s for s in self.seqs if row[:len(s) - 1] == s[:-1][:len(row)])
            for j in range(idx.size(1)):
                out[r, j, 7 if self.lie else (s[j + 1] if j + 1 < len(s) else 0)] = 1
        return out


def test_eval_tasks(ds):
    seqs = [b"ab=cd;ef", b"xy=zw;qq", b"k=v"]
    r = E.eval_tasks(Oracle(seqs), ds)["copy"]["id"]
    assert r["token_acc"] == 1.0 and r["exact_match"] == 1.0 and r["examples"] == 3
    r = E.eval_tasks(Oracle(seqs, lie=True), ds)["copy"]["id"]
    assert r["token_acc"] == 0.0 and r["exact_match"] == 0.0
    assert E.eval_tasks(lm("tf_debug"), ds)["copy"]["id"]["exact_match"] < 0.5


def test_eval_tasks_missing(ds, tmp_path):
    os.remove(tmp_path / "tiny" / "eval_tasks.json")
    assert E.eval_tasks(lm("tf_debug"), ds) == {}


class Gen(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.lm, self.fn = config.LMConfig(seq_len=64), fn

    def generate(self, prompt, max_new, temperature=1.0, top_k=0, generator=None):
        tail = torch.tensor(list(self.fn(max_new)), dtype=torch.long).repeat(prompt.size(0), 1)
        return torch.cat([prompt, tail], 1)


def test_memorization(ds, tmp_path):
    train = (tmp_path / "tiny" / "train.bin").read_bytes()
    copy = E.eval_generation(Gen(lambda n: train[100:100 + n]), ds, n=2, new_bytes=100, seeds=(0,))
    assert copy["memorization_rate"] == 1.0
    rnd = E.eval_generation(Gen(lambda n: bytes(torch.randint(0, 256, (n,)).tolist())), ds, n=2, new_bytes=100, seeds=(0,))
    assert rnd["memorization_rate"] < 0.05 and rnd["latency"]["bytes_per_s"] > 0


def test_generation_real_model(ds):
    r = E.eval_generation(lm("tf_debug"), ds, n=2, new_bytes=10, seeds=(0, 1))
    assert len(r["samples"]) == 2 and r["n_generated"] == 4


def test_text_metrics():
    assert E.repeated_ngram_fraction(b"a" * 20, 8) == 12 / 13
    assert E.repeated_ngram_fraction(bytes(range(20)), 8) == 0.0
    assert E.distinct_ngram_ratio(b"a" * 10, 4) == 1 / 7
    assert E.distinct_ngram_ratio(bytes(range(10)), 4) == 1.0
    assert E.distinct_ngram_ratio(b"abc", 4) is None
    assert E.valid_utf8("hé".encode()) and not E.valid_utf8(b"\xff\xfe")
    assert E.printable_fraction(b"ab\x00\x01") == 0.5
    assert E.memorization_rate([b"x" * 64], b"y" * 100) == 0.0


def test_eval_diffusion_deterministic(ds):
    torch.manual_seed(0)
    m = ByteDiffusionLM(config.get_preset("debug")).eval()
    a, b = E.eval_diffusion(m, ds, windows=8), E.eval_diffusion(m, ds, windows=8)
    assert a == b and len(a["buckets"]) == 4 and a["buckets"][3]["t_range"][1] == 50
    assert "seconds_per_sample" in E.eval_generation(m, ds, n=2, seeds=(0,))["latency"]
