import os, sys, tempfile, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import get_preset
from main import ByteDiffusionLM, EMA
from train import expand_state_dict, warm_start


def build(sc):
    cfg = get_preset("hybrid")
    m = replace(cfg.model, hybrid_layers=2, model_dim=64, hybrid_heads=4, seq_len=32)
    return ByteDiffusionLM(replace(cfg, model=m, diffusion=replace(cfg.diffusion, self_cond=sc, param="logits", ce_weight=0.0, x0_rescale=True)))


class WarmStart(unittest.TestCase):
    def test_step0_equals_original(self):
        torch.manual_seed(0)
        old = build(False).eval(); new = build(True).eval()
        ema_old = EMA(old, decay=0.999, warmup=True); ema_new = EMA(new, decay=0.999, warmup=True)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.pt")
            torch.save({"model_state_dict": old.state_dict(), "ema_state_dict": ema_old.state_dict()}, p)
            warm_start(p, new, ema_new)
        e = old.cfg.model.byte_embed_dim
        x = torch.randn(2, 32, e); t = torch.tensor([5, 900])
        go, gn = old.gaussian_diffusion, new.gaussian_diffusion
        with torch.no_grad():
            a = go.denoise(x, t)
            self.assertTrue(torch.equal(a, gn.denoise(x, t)))                                  # x0_prev=None (zeros)
            self.assertTrue(torch.equal(a, gn.denoise(x, t, x0_prev=torch.zeros_like(x))))     # explicit zeros
            self.assertTrue(torch.equal(a, gn.denoise(x, t, x0_prev=torch.randn_like(x))))      # zero weights: x0_prev ignored at step 0
        w = new.denoise_unet.inp.weight
        self.assertTrue(torch.equal(w[:, :e], old.denoise_unet.inp.weight)); self.assertEqual(float(w[:, e:].abs().sum()), 0.0)
        self.assertTrue(torch.equal(ema_new.model.denoise_unet.inp.weight[:, :e], ema_old.model.denoise_unet.inp.weight))

    def test_bad_shapes_rejected(self):
        old = build(False); new = build(True)
        sd = old.state_dict(); k = "denoise_unet.out.weight"; sd[k] = sd[k][:-1]
        with self.assertRaises(Exception):
            expand_state_dict(sd, new.state_dict())

    def test_new_channels_get_gradient(self):
        new = build(True).train(); e = new.cfg.model.byte_embed_dim
        new.denoise_unet.inp.weight.data[:, e:] = 0
        loss, _ = new(torch.randint(0, 256, (8, 32))); loss.backward()
        self.assertGreater(float(new.denoise_unet.inp.weight.grad[:, e:].abs().sum()), 0.0)

if __name__ == "__main__":
    unittest.main()
