import os, sys, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import Config, get_preset, validate
from main import ByteDiffusionLM


def cfg_for(sc=True):
    c = get_preset("hier")
    dif = replace(c.diffusion, param="logits", self_cond=sc, ce_weight=0.0, x0_rescale=True, t_sampling="warp", ce_schedule="alpha_bar")
    return replace(c, diffusion=dif, model=replace(c.model, embed_norm=8, dropout=0.0))


def build(sc=True):
    torch.manual_seed(0)
    return ByteDiffusionLM(cfg_for(sc)).eval()


def perturb_film(net):  # FiLM starts at zero (output is timestep-blind at init); give it weight so time dependence is visible
    for mod in net.modules():
        if hasattr(mod, "film"):
            torch.nn.init.normal_(mod.film.weight, std=0.05)


class Hier(unittest.TestCase):
    def test_param_count_and_preset(self):
        m = build(True)
        self.assertEqual(sum(p.numel() for p in m.parameters()), 14_062_272)  # hybrid: 14,070,528
        validate(get_preset("hier"))
        with self.assertRaises(ValueError):
            c = get_preset("hier"); validate(replace(c, model=replace(c.model, model_dim=200)))

    def test_shapes_and_lengths(self):
        net = build().denoise_unet
        t = torch.tensor([10, 500])
        with torch.no_grad():
            self.assertEqual(net(torch.randn(2, 256, 128), t).shape, (2, 256, 256))
            self.assertEqual(net(torch.randn(2, 512, 128), t).shape, (2, 512, 256))
        with self.assertRaises(ValueError):
            net(torch.randn(2, 250, 128), t)

    def test_timestep_dependence(self):
        net = build().denoise_unet; perturb_film(net); x = torch.randn(1, 64, 128)
        with torch.no_grad():
            self.assertFalse(torch.allclose(net(x, torch.tensor([5])), net(x, torch.tensor([900])), atol=1e-5))

    def test_bidirectional_both_ways(self):
        net = build().denoise_unet; perturb_film(net); t = torch.tensor([300])
        x = torch.randn(1, 64, 128); x2 = x.clone(); x2[:, 40:] += torch.randn(1, 24, 128)
        x3 = x.clone(); x3[:, :10] += torch.randn(1, 10, 128)
        with torch.no_grad():
            y, y2, y3 = net(x, t), net(x2, t), net(x3, t)
        self.assertGreater((y2[:, :10] - y[:, :10]).abs().max().item(), 1e-6)   # later inputs affect earlier outputs
        self.assertGreater((y3[:, 30:] - y[:, 30:]).abs().max().item(), 1e-6)   # earlier inputs affect later outputs

    def test_selfcond_zero_input(self):
        m = build(True); gd = m.gaussian_diffusion; e = m.cfg.model.byte_embed_dim; perturb_film(m.denoise_unet)
        self.assertEqual(m.denoise_unet.inp.in_features, 2 * e)
        x = torch.randn(2, 32, e); t = torch.tensor([5, 9])
        with torch.no_grad():
            a = gd.denoise(x, t); b = gd.denoise(x, t, x0_prev=torch.zeros_like(x)); c = gd.denoise(x, t, x0_prev=torch.ones_like(x))
        self.assertTrue(torch.equal(a, b)); self.assertFalse(torch.allclose(a, c))
        self.assertEqual(build(False).denoise_unet.inp.in_features, e)

    def test_hashes_unchanged(self):
        for name in ("compact", "birnn", "hybrid"):
            c = get_preset(name)
            self.assertEqual(Config.from_dict(c.to_dict()).architecture_hash(), c.architecture_hash())
        h = get_preset("hier")
        self.assertNotEqual(h.architecture_hash(), get_preset("hybrid").architecture_hash())
        h2 = replace(h, model=replace(h.model, hybrid_layers=3, hybrid_branches="conv"))  # unused fields do not move the hier hash
        self.assertEqual(h.architecture_hash(), h2.architecture_hash())
        # a finished hybrid run still loads and hashes consistently (cross-version equality is checked in the commit notes)
        import json
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "rung_hyb_selfcond", "config.json")
        if os.path.exists(p):
            c = Config.from_dict(json.load(open(p)))
            self.assertEqual(c.model.denoiser, "hybrid"); self.assertEqual(c.architecture_hash(), Config.from_dict(c.to_dict()).architecture_hash())

    def test_backward_and_train_step(self):
        for sc in (False, True):
            m = build(sc).train(); loss, _ = m(torch.randint(0, 256, (2, 64)))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            bad = [n for n, p in m.denoise_unet.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
            self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
