import os, sys, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import Config, get_preset
from main import ByteDiffusionLM


def build(cond):
    cfg = get_preset("debug")
    return ByteDiffusionLM(replace(cfg, diffusion=replace(cfg.diffusion, cond_prompt=cond, ce_weight=0.1)))


class CondPrompt(unittest.TestCase):
    def test_default_off_hash_and_old_config(self):
        cfg = get_preset("debug")
        self.assertFalse(cfg.diffusion.cond_prompt)
        d = cfg.to_dict(); del d["diffusion"]["cond_prompt"]
        self.assertEqual(Config.from_dict(d).architecture_hash(), cfg.architecture_hash())
        on = replace(cfg, diffusion=replace(cfg.diffusion, cond_prompt=True))
        self.assertNotEqual(on.architecture_hash(), cfg.architecture_hash())

    def test_clean_positions_not_noised_and_no_loss(self):
        m = build(True).eval()
        gd = m.gaussian_diffusion
        seen = {}
        orig = gd.denoise_model.forward
        gd.denoise_model.forward = lambda x, t: (seen.__setitem__("x", x), orig(x, t))[1]
        x = torch.randint(0, 256, (4, 32))
        mask = torch.zeros(4, 32, dtype=torch.bool); mask[:, :10] = True
        t = torch.full((4,), 25).long()
        emb = m.byte_embedder(x)
        loss, st = m(x, t=t, noise=torch.randn_like(emb), clean_mask=mask)
        xin = seen["x"]
        self.assertTrue(torch.equal(xin[:, :10, :-1], emb[:, :10]))       # clean = exact embedding
        self.assertFalse(torch.allclose(xin[:, 10:, :-1], emb[:, 10:]))   # rest noised
        self.assertTrue(torch.equal(xin[..., -1], mask.float()))          # mask channel
        # loss ignores clean positions: changing noise there must not change the loss
        n1 = torch.randn_like(emb); n2 = n1.clone(); n2[:, :10] = torch.randn_like(n2[:, :10])
        torch.manual_seed(0); l1, _ = m(x, t=t, noise=n1, clean_mask=mask)
        torch.manual_seed(0); l2, _ = m(x, t=t, noise=n2, clean_mask=mask)
        self.assertAlmostEqual(l1.item(), l2.item(), places=6)
        # fully clean window -> zero loss
        l0, _ = m(x, t=t, clean_mask=torch.ones_like(mask))
        self.assertEqual(l0.item(), 0.0)

    def test_training_samples_prefix_and_off_mode_unaffected(self):
        m = build(True).train()
        mask = m.gaussian_diffusion.sample_clean_mask(64, "cpu")
        self.assertTrue((mask.sum(1) <= 128).all() and (mask[:, :-1].int() >= mask[:, 1:].int()).all())  # prefix shaped
        loss, _ = m(torch.randint(0, 256, (4, 32)))
        self.assertTrue(torch.isfinite(loss))
        off = build(False)
        self.assertEqual(off.denoise_unet.init_conv.in_channels, 16)
        self.assertEqual(m.denoise_unet.init_conv.in_channels, 17)


if __name__ == "__main__":
    unittest.main()
