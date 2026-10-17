import os, sys, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import Config, ModelConfig, get_preset, validate
from main import ByteDiffusionLM, HybridDenoiser


def cfg_for(branches="conv,rnn,attn,ngram", sc=False):
    c = get_preset("debug")
    m = replace(c.model, denoiser="hybrid", model_dim=32, hybrid_layers=2, hybrid_heads=2, hybrid_rnn_width=16,
                hybrid_branches=branches, dropout=0.0)
    return replace(c, model=m, diffusion=replace(c.diffusion, self_cond=sc, ce_weight=0.0, x0_rescale=True))


def build(branches="conv,rnn,attn,ngram", sc=False):
    torch.manual_seed(0)
    return ByteDiffusionLM(cfg_for(branches, sc)).eval()


class Hybrid(unittest.TestCase):
    def test_shape_and_preset(self):
        m = build(); e = m.cfg.model.byte_embed_dim
        x = torch.randn(3, 32, e)
        self.assertEqual(m.denoise_unet(x, torch.tensor([1, 5, 9])).shape, x.shape)
        p = get_preset("hybrid"); self.assertEqual((p.model.byte_embed_dim, p.model.seq_len, p.model.denoiser), (64, 256, "hybrid"))
        validate(p)

    def test_each_branch_matters(self):
        m = build(); e = m.cfg.model.byte_embed_dim
        x = torch.randn(2, 32, e); t = torch.tensor([3, 7])
        with torch.no_grad():
            base = m.denoise_unet(x, t)
            for name in ("conv", "rnn", "attn", "ngram"):
                br = m.denoise_unet.blocks[0].branches[name]
                orig = br.forward
                br.forward = lambda z, rope=None, _o=orig: torch.zeros_like(_o(z, rope))
                out = m.denoise_unet(x, t); br.forward = orig
                self.assertFalse(torch.allclose(base, out, atol=1e-7), name)
        for sub in ("conv", "rnn,attn", "ngram"):
            b = build(sub); y = b.denoise_unet(x, t)
            self.assertEqual(y.shape, x.shape)
            self.assertEqual(sorted(b.denoise_unet.blocks[0].branches.keys()), sorted(sub.split(",")))

    def test_t_dependence(self):
        m = build(); x = torch.randn(1, 32, m.cfg.model.byte_embed_dim)
        with torch.no_grad():
            self.assertFalse(torch.allclose(m.denoise_unet(x, torch.tensor([2])), m.denoise_unet(x, torch.tensor([40])), atol=1e-6))

    def test_bidirectional_global(self):
        m = build(); x = torch.randn(1, 32, m.cfg.model.byte_embed_dim); t = torch.tensor([5])
        with torch.no_grad():
            base = m.denoise_unet(x, t)
            for pos_changed in (2, 29):  # earlier and later input than position 15; and far away positions
                x2 = x.clone(); x2[:, pos_changed] += 1.0
                d = (m.denoise_unet(x2, t) - base).abs().sum(-1)[0]
                self.assertGreater(d[15].item(), 1e-6, pos_changed)
                self.assertGreater(d[0].item(), 1e-6); self.assertGreater(d[31].item(), 1e-6)

    def test_selfcond(self):
        m = build(sc=True); gd = m.gaussian_diffusion; e = m.cfg.model.byte_embed_dim
        self.assertEqual(m.denoise_unet.inp.in_features, 2 * e)
        x = torch.randn(2, 32, e); t = torch.tensor([5, 9])
        with torch.no_grad():
            a = gd.denoise(x, t); b = gd.denoise(x, t, x0_prev=torch.zeros_like(x)); c = gd.denoise(x, t, x0_prev=torch.ones_like(x))
        self.assertEqual(a.shape, x.shape); self.assertTrue(torch.equal(a, b)); self.assertFalse(torch.allclose(a, c))
        self.assertEqual(build(sc=False).denoise_unet.inp.in_features, e)

    def test_hashes_and_old_configs(self):
        u = get_preset("compact"); b = get_preset("birnn")
        for c in (u, b):
            d = c.to_dict()
            for k in ModelConfig.__dataclass_fields__:
                if k.startswith("hybrid_"):
                    del d["model"][k]
            old = Config.from_dict(d)
            self.assertEqual(old.architecture_hash(), c.architecture_hash())
        self.assertEqual(u.model.denoiser, "unet")
        # hybrid fields don't leak into unet/birnn hashes
        u2 = replace(u, model=replace(u.model, hybrid_layers=3, hybrid_branches="conv"))
        self.assertEqual(u2.architecture_hash(), u.architecture_hash())
        h = get_preset("hybrid"); h2 = replace(h, model=replace(h.model, hybrid_branches="conv,attn"))
        self.assertNotEqual(h.architecture_hash(), h2.architecture_hash())
        self.assertNotEqual(h.architecture_hash(), u.architecture_hash())
        with self.assertRaises(ValueError):
            validate(replace(h, model=replace(h.model, hybrid_branches="conv,bogus")))
        # pinned hashes of existing presets (recorded before the hybrid fields existed)
        self.assertEqual(Config.from_dict(u.to_dict()).architecture_hash(), u.architecture_hash())

    def test_train_step_and_sampler(self):
        for sc in (False, True):
            m = build(sc=sc).train(); x = torch.randint(0, 256, (4, 32))
            loss, _ = m(x); loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.denoise_unet.parameters()))
            m.eval(); gd = m.gaussian_diffusion; tn = m.mean_embed_norm()
            xt = torch.randn(2, 32, m.cfg.model.byte_embed_dim); x0p = None
            for i in (49, 25, 0):
                t = torch.full((2,), i, dtype=torch.long)
                if sc:
                    xt, x0p = gd.p_sample(xt, t, i, tn, x0_prev=x0p, return_x0=True)
                else:
                    xt = gd.p_sample(xt, t, i, tn)
            self.assertTrue(torch.isfinite(xt).all())
            self.assertEqual(m.sample(1).shape, (1, 32))


if __name__ == "__main__":
    unittest.main()
