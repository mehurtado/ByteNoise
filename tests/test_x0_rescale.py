import os, sys, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import get_preset
from main import ByteDiffusionLM


def build(rescale):
    cfg = get_preset("debug")
    return ByteDiffusionLM(replace(cfg, diffusion=replace(cfg.diffusion, x0_rescale=rescale))).eval()


class X0Rescale(unittest.TestCase):
    def test_default_off_and_hash_unchanged(self):
        cfg = get_preset("debug")
        self.assertFalse(cfg.diffusion.x0_rescale)
        on = replace(cfg, diffusion=replace(cfg.diffusion, x0_rescale=True))
        self.assertEqual(cfg.architecture_hash(), on.architecture_hash())

    def test_old_config_dict_loads(self):
        from config import Config
        d = get_preset("debug").to_dict()
        del d["diffusion"]["x0_rescale"]
        self.assertFalse(Config.from_dict(d).diffusion.x0_rescale)

    def test_default_sampler_unchanged(self):
        torch.manual_seed(0)
        m = build(False)
        gd = m.gaussian_diffusion
        x = torch.randn(2, 32, 16)
        t = torch.full((2,), 10, dtype=torch.long)
        torch.manual_seed(1)
        a = gd.p_sample(x, t, 10)
        torch.manual_seed(1)
        b = gd.p_sample(x, t, 10, target_norm=5.0)  # target_norm ignored when off
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(gd.clip_denoised)

    def test_rescaled_step_norm(self):
        torch.manual_seed(0)
        m = build(True)
        gd = m.gaussian_diffusion
        R = m.mean_embed_norm()
        x = torch.randn(2, 32, 16) * 50  # huge input: raw x0_hat would blow up
        t = torch.full((2,), 999 if gd.num_timesteps > 999 else gd.num_timesteps - 1, dtype=torch.long)
        out = gd.p_sample(x, t, 0, R)  # t_index 0 -> posterior mean, no noise
        # at the last step posterior mean == x0_hat, so norms equal R
        t0 = torch.zeros(2, dtype=torch.long)
        out = gd.p_sample(x, t0, 0, R)
        self.assertTrue(torch.allclose(out.norm(dim=-1), torch.full((2, 32), R), rtol=1e-3))
        with self.assertRaises(ValueError):
            gd.p_sample(x, t0, 0)

    def test_sample_runs_without_clip(self):
        m = build(True)
        ids = m.sample(2)
        self.assertEqual(tuple(ids.shape), (2, 32))


if __name__ == "__main__":
    unittest.main()
