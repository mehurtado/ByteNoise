import os, sys, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import get_preset, validate
from models_lm import ByteLM, scan_parallel, scan_sequential, positive_activation, log_positive_activation


def lm_cfg(name, **kw):
    cfg = get_preset(name)
    return replace(cfg, lm=replace(cfg.lm, **kw))


class Scan(unittest.TestCase):
    def test_parallel_matches_sequential(self):
        torch.manual_seed(0)
        zl, cl = torch.randn(3, 200, 8) * 2, torch.randn(3, 200, 8) * 2
        seq = scan_sequential(torch.sigmoid(zl), positive_activation(cl))
        par = scan_parallel(zl, log_positive_activation(cl))
        self.assertTrue(torch.allclose(seq, par, rtol=1e-3, atol=1e-4), (seq - par).abs().max())

    def test_log_activation_consistent(self):
        x = torch.randn(1000) * 5
        self.assertTrue(torch.allclose(positive_activation(x).log(), log_positive_activation(x), atol=1e-5))

    def test_parallel_matches_long_sequence(self):
        torch.manual_seed(1)
        zl, cl = torch.randn(1, 2048, 4), torch.randn(1, 2048, 4)
        seq = scan_sequential(torch.sigmoid(zl), positive_activation(cl))
        par = scan_parallel(zl, log_positive_activation(cl))
        self.assertTrue(torch.allclose(seq, par, rtol=1e-3, atol=1e-4))


class Causality(unittest.TestCase):
    def check(self, cfg, parallel=True):
        torch.manual_seed(0)
        m = ByteLM(cfg).eval()
        m.parallel = parallel
        x = torch.randint(0, 256, (2, 24))
        base = m.logits(x)
        x2 = x.clone()
        x2[:, 15:] = torch.randint(0, 256, (2, 9))
        out = m.logits(x2)
        self.assertTrue(torch.allclose(base[:, :15], out[:, :15], atol=1e-5))
        self.assertFalse(torch.allclose(base[:, 15:], out[:, 15:], atol=1e-5))

    def test_transformer_causal(self):
        self.check(get_preset("tf_debug"))

    def test_recurrent_causal_parallel(self):
        self.check(get_preset("rec_debug"), True)

    def test_recurrent_causal_sequential(self):
        self.check(get_preset("rec_debug"), False)

    def test_looped_causal(self):
        self.check(lm_cfg("rec_debug", loops=3, iter_embedding=True))
        self.check(lm_cfg("tf_debug", loops=2))


class Shapes(unittest.TestCase):
    def test_forward_and_generate(self):
        for name in ("tf_debug", "rec_debug"):
            m = ByteLM(get_preset(name))
            x = torch.randint(0, 256, (3, 33))
            loss, stats = m(x)
            self.assertEqual(m.logits(x[:, :-1]).shape, (3, 32, 256))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            out = m.generate(x[:, :4], 6, temperature=0.0)
            self.assertEqual(out.shape, (3, 10))

    def test_parallel_and_sequential_model_agree(self):
        m = ByteLM(get_preset("rec_debug")).eval()
        x = torch.randint(0, 256, (2, 32))
        a = m.logits(x)
        m.parallel = False
        b = m.logits(x)
        self.assertTrue(torch.allclose(a, b, atol=1e-3))

    def test_sharing_reduces_params(self):
        a = sum(p.numel() for p in ByteLM(lm_cfg("tf_debug", n_layers=1, loops=4)).parameters())
        b = sum(p.numel() for p in ByteLM(lm_cfg("tf_debug", n_layers=4, loops=1)).parameters())
        self.assertLess(a, b)

    def test_initial_loss_near_uniform(self):
        m = ByteLM(get_preset("tf_debug"))
        loss, _ = m(torch.randint(0, 256, (8, 33)))
        self.assertAlmostEqual(loss.item(), 5.545, delta=0.3)


if __name__ == "__main__":
    unittest.main()
