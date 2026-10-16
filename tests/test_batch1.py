import os
import sys
import tempfile
import unittest
from dataclasses import replace

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PRESETS, get_preset, override, validate, ModelConfig
from main import ByteDiffusionLM, EMA
import train as T


def make_bytes(n_windows=20, seq_len=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n_windows * seq_len,), generator=g).tolist()


def debug_cfg(**train_kw):
    cfg = get_preset("debug")
    return override(cfg, max_updates=8, save_interval=1000, log_interval=1000, **train_kw)


class ParamCounts(unittest.TestCase):
    def test_known_counts(self):
        self.assertEqual(sum(p.numel() for p in ByteDiffusionLM(PRESETS["original"]).parameters()), 57_603_328)
        self.assertEqual(sum(p.numel() for p in ByteDiffusionLM(PRESETS["compact"]).parameters()), 14_430_848)


class Validation(unittest.TestCase):
    def bad(self, **model_kw):
        cfg = replace(get_preset("debug"), model=replace(get_preset("debug").model, **model_kw))
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_heads(self):
        self.bad(attn_heads=3)

    def test_groups(self):
        self.bad(model_dim=36, attn_heads=2)

    def test_seq_len(self):
        self.bad(seq_len=33)

    def test_presets_valid(self):
        for p in PRESETS.values():
            validate(p)


class LossAccounting(unittest.TestCase):
    def test_group_sizes(self):
        self.assertEqual(T.group_sizes(10, 4), [4, 4, 2])
        self.assertEqual(T.group_sizes(8, 4), [4, 4])

    def test_unscaled_losses_returned(self):
        vals = iter([1.0, 2.0, 3.0, 4.0, 5.0])
        w = nn.Parameter(torch.ones(1))

        def loss_fn(b):
            return w * next(vals), {}

        losses, _ = T.accumulate_group(loss_fn, [0] * 5)
        self.assertEqual(losses, [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(sum(losses) / len(losses), 3.0)

    def test_partial_group_matches_big_batch(self):
        torch.manual_seed(0)
        x = torch.randn(8, 3)
        for sizes in ([3, 3, 2], [4, 4]):
            pos = 0
            for n in sizes:
                chunk = x[pos:pos + n]
                pos += n
                w1 = nn.Parameter(torch.ones(3))
                w2 = nn.Parameter(torch.ones(3))
                (chunk * w1).pow(2).mean().backward()
                T.accumulate_group(lambda b: ((b.unsqueeze(0) * w2).pow(2).mean(), {}), list(chunk))
                self.assertTrue(torch.allclose(w1.grad, w2.grad, atol=1e-6))

    def test_nonfinite_reports_timestep(self):
        stats = {"t": torch.tensor([3, 17]), "per_sample_noise_loss": torch.tensor([1.0, float("nan")])}
        with self.assertRaises(FloatingPointError) as cm:
            T.check_finite_loss(torch.tensor(float("nan")), stats)
        self.assertIn("[17]", str(cm.exception))

    def test_bucket_losses(self):
        sums, counts = T.bucket_losses(torch.tensor([0, 49, 25]), torch.tensor([1.0, 3.0, 5.0]), 50, 2)
        self.assertEqual(sums.tolist(), [1.0, 8.0])
        self.assertEqual(counts.tolist(), [1.0, 2.0])


class EMATest(unittest.TestCase):
    def test_formula_and_warmup(self):
        model = nn.Sequential(nn.Linear(1, 1, bias=False))
        model[0].weight.data.fill_(1.0)
        ema = EMA(model, decay=0.9, warmup=True)
        model[0].weight.data.fill_(3.0)
        self.assertAlmostEqual(ema.current_decay(), 0.1)
        ema.update(model)
        self.assertAlmostEqual(ema.model[0].weight.item(), 0.1 * 1.0 + 0.9 * 3.0, places=5)
        self.assertAlmostEqual(ema.current_decay(), 2 / 11)
        ema2 = EMA(model, decay=0.9, warmup=False)
        self.assertEqual(ema2.current_decay(), 0.9)


class RunProtocol(unittest.TestCase):
    def test_resume_matches_uninterrupted(self):
        data = make_bytes()
        with tempfile.TemporaryDirectory() as d:
            full = T.run_training(debug_cfg(), os.path.join(d, "full"), "fresh", "cpu", data=data, quiet=True)
            part = os.path.join(d, "part")
            T.run_training(debug_cfg(), part, "fresh", "cpu", data=data, stop_after=4, quiet=True)
            res = T.run_training(None, part, "resume", "cpu", data=data, quiet=True)
            self.assertEqual(res["progress"]["updates"], 8)
            for a, b in zip(full["model"].parameters(), res["model"].parameters()):
                self.assertTrue(torch.equal(a, b))
            for a, b in zip(full["ema"].model.parameters(), res["ema"].model.parameters()):
                self.assertTrue(torch.equal(a, b))

    def test_exact_update_count(self):
        with tempfile.TemporaryDirectory() as d:
            r = T.run_training(debug_cfg(), os.path.join(d, "r"), "fresh", "cpu", data=make_bytes(), quiet=True)
            self.assertEqual(r["progress"]["updates"], 8)
            self.assertEqual(r["ema"].num_updates.item(), 8)

    def test_fresh_refuses_existing_run(self):
        with tempfile.TemporaryDirectory() as d:
            run = os.path.join(d, "r")
            T.run_training(override(debug_cfg(), max_updates=2), run, "fresh", "cpu", data=make_bytes(), quiet=True)
            with self.assertRaises(T.CheckpointError):
                T.run_training(debug_cfg(), run, "fresh", "cpu", data=make_bytes(), quiet=True)

    def test_resume_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(T.CheckpointError):
                T.run_training(None, d, "resume", "cpu", data=make_bytes(), quiet=True)

    def test_resume_rejects_changed_data(self):
        with tempfile.TemporaryDirectory() as d:
            run = os.path.join(d, "r")
            T.run_training(debug_cfg(), run, "fresh", "cpu", data=make_bytes(), stop_after=2, quiet=True)
            with self.assertRaises(T.CheckpointError):
                T.run_training(None, run, "resume", "cpu", data=make_bytes(seed=1), quiet=True)

    def test_incompatible_architecture_rejected(self):
        small = ByteDiffusionLM(get_preset("debug"))
        other_cfg = replace(get_preset("debug"), model=replace(get_preset("debug").model, model_dim=64, attn_heads=4))
        other = ByteDiffusionLM(other_cfg)
        with self.assertRaises(T.CheckpointError) as cm:
            T.check_compat(other, small.state_dict())
        self.assertIn("incompatible", str(cm.exception))

    def test_generate_roundtrip_and_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            run = os.path.join(d, "r")
            T.run_training(override(debug_cfg(), max_updates=3), run, "fresh", "cpu", data=make_bytes(), quiet=True)
            out = T.generate(run_dir=run, which="both", seeds=[0, 1], device="cpu")
            self.assertEqual(set(out), {"raw", "ema"})
            self.assertEqual(len(out["ema"]), 2)
            lines = open(os.path.join(run, "metrics.jsonl")).read().strip().splitlines()
            self.assertEqual(len(lines), 3)
            for k in ("noise_loss", "ce_loss", "bucket_noise_loss", "grad_norm", "lr", "examples", "emb_norm", "x0_pred_norm"):
                self.assertIn(k, lines[0])


if __name__ == "__main__":
    unittest.main()
