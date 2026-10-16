import os, sys, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import Config, get_preset
from main import ByteDiffusionLM, BiRNNDenoiser
from models_lm import scan_parallel, scan_sequential, log_positive_activation, positive_activation


def small():
    return BiRNNDenoiser(32, 8, 2, 7, 2.667, 0.0).eval()


class BiRNN(unittest.TestCase):
    def test_shape(self):
        x, t = torch.randn(3, 20, 8), torch.randint(0, 1000, (3,))
        self.assertEqual(small()(x, t).shape, (3, 20, 8))

    def test_bidirectional_and_time(self):
        torch.manual_seed(0)
        m, t = small(), torch.tensor([10])
        x = torch.randn(1, 20, 8)
        y = m(x, t)
        x2 = x.clone(); x2[:, 15:] += 1.0           # change only later positions
        y2 = m(x2, t)
        self.assertGreater((y2[:, :15] - y[:, :15]).abs().max().item(), 1e-5)   # earlier outputs see the future
        x3 = x.clone(); x3[:, :5] += 1.0            # and earlier positions affect later outputs
        self.assertGreater((m(x3, t)[:, 10:] - y[:, 10:]).abs().max().item(), 1e-5)
        self.assertGreater((m(x, torch.tensor([900])) - y).abs().max().item(), 1e-5)

    def test_scan_matches_sequential(self):
        torch.manual_seed(0)
        zl, cl = torch.randn(2, 30, 5), torch.randn(2, 30, 5)
        a = scan_parallel(zl, log_positive_activation(cl))
        b = scan_sequential(torch.sigmoid(zl), positive_activation(cl))
        self.assertTrue(torch.allclose(a, b, atol=1e-4, rtol=1e-4))

    def test_full_model_loss_and_sampler_path(self):
        cfg = get_preset("debug")
        cfg = replace(cfg, model=replace(cfg.model, denoiser="birnn", model_dim=32, rnn_layers=2))
        m = ByteDiffusionLM(cfg)
        loss, _ = m(torch.randint(0, 256, (2, 32)))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(m.sample(1).shape, (1, 32))

    def test_old_config_loads_and_hash_stable(self):
        cfg = get_preset("debug")
        d = cfg.to_dict()
        for k in ("denoiser", "rnn_layers", "rnn_conv", "rnn_ffn_mult"):
            del d["model"][k]
        old = Config.from_dict(d)
        self.assertEqual(old.model.denoiser, "unet")
        self.assertEqual(old.architecture_hash(), cfg.architecture_hash())
        self.assertEqual(Config.from_dict(cfg.to_dict()), cfg)
        b = get_preset("birnn")
        self.assertNotEqual(b.architecture_hash(), get_preset("compact").architecture_hash())
        self.assertEqual(Config.from_dict(b.to_dict()), b)


if __name__ == "__main__":
    unittest.main()
