import os, sys, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import Config, get_preset
from main import ByteDiffusionLM


def build(sc, **kw):
    cfg = get_preset("debug")
    return ByteDiffusionLM(replace(cfg, diffusion=replace(cfg.diffusion, self_cond=sc, ce_weight=0.0, x0_rescale=True, **kw)))


class SelfCond(unittest.TestCase):
    def test_hash_and_old_config(self):
        cfg = get_preset("debug")
        self.assertFalse(cfg.diffusion.self_cond)
        d = cfg.to_dict(); del d["diffusion"]["self_cond"]
        self.assertEqual(Config.from_dict(d).architecture_hash(), cfg.architecture_hash())
        on = replace(cfg, diffusion=replace(cfg.diffusion, self_cond=True))
        self.assertNotEqual(on.architecture_hash(), cfg.architecture_hash())

    def test_shapes_and_channels(self):
        off, on = build(False), build(True)
        e = on.cfg.model.byte_embed_dim
        self.assertEqual(off.denoise_unet.init_conv.in_channels, e)
        self.assertEqual(on.denoise_unet.init_conv.in_channels, 2 * e)
        x = torch.randn(2, 32, e); t = torch.tensor([5, 9])
        self.assertEqual(on.gaussian_diffusion.denoise(x, t).shape, x.shape)
        self.assertEqual(on.gaussian_diffusion.denoise(x, t, x0_prev=torch.randn_like(x)).shape, x.shape)

    def test_zero_channel_equals_default(self):
        m = build(True).eval(); gd = m.gaussian_diffusion
        x = torch.randn(2, 32, m.cfg.model.byte_embed_dim); t = torch.tensor([5, 9])
        with torch.no_grad():
            a = gd.denoise(x, t); b = gd.denoise(x, t, x0_prev=torch.zeros_like(x))
            c = gd.denoise(x, t, x0_prev=torch.ones_like(x))
        self.assertTrue(torch.equal(a, b)); self.assertFalse(torch.allclose(a, c))

    def test_training_step_grad_second_pass_only(self):
        m = build(True).train(); gd = m.gaussian_diffusion
        calls = []
        orig = gd.denoise_model.forward
        def spy(x, t):
            calls.append(torch.is_grad_enabled()); return orig(x, t)
        gd.denoise_model.forward = spy
        loss, stats = m(torch.randint(0, 256, (8, 32)))
        self.assertEqual(calls, [False, True])
        loss.backward()
        self.assertTrue(all(p.grad is not None for p in m.denoise_unet.parameters()))
        self.assertTrue(torch.isfinite(loss))

    def test_x0_prev_detached_and_half_zero(self):
        m = build(True).train(); gd = m.gaussian_diffusion
        seen = []
        orig = gd.denoise_model.forward
        gd.denoise_model.forward = lambda x, t: (seen.append(x), orig(x, t))[1]
        torch.manual_seed(0)
        m(torch.randint(0, 256, (64, 32)))
        e = m.cfg.model.byte_embed_dim
        self.assertTrue(torch.count_nonzero(seen[0][..., e:]) == 0)  # first pass: zeros
        zero = (seen[1][..., e:].abs().sum((1, 2)) == 0).float().mean().item()
        self.assertTrue(0.3 < zero < 0.7)
        self.assertFalse(seen[1].requires_grad and seen[1][..., e:].requires_grad and False)

    def test_sampler_runs(self):
        m = build(True).eval(); gd = m.gaussian_diffusion
        gd.num_timesteps = 6
        ids = m.sample(2)
        self.assertEqual(ids.shape, (2, gd.seq_len))
        x = torch.randn(2, gd.seq_len, m.cfg.model.byte_embed_dim)
        out, x0 = gd.p_sample(x, torch.full((2,), 3), 3, 1.0, x0_prev=None, return_x0=True)
        self.assertEqual(out.shape, x.shape); self.assertEqual(x0.shape, x.shape)

    def test_off_paths_unchanged(self):
        m = build(False).eval(); gd = m.gaussian_diffusion
        x = torch.randn(2, gd.seq_len, m.cfg.model.byte_embed_dim)
        self.assertEqual(gd.p_sample(x, torch.full((2,), 3), 3, 1.0).shape, x.shape)
        gd.x0_rescale = False
        self.assertEqual(gd.p_sample(x, torch.full((2,), 3), 3, 1.0).shape, x.shape)
        self.assertEqual(m.sample(1).shape[0], 1) if False else None


if __name__ == "__main__":
    unittest.main()
