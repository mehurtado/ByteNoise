import json, os, sys, tempfile, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import Config, get_preset, validate
from main import ByteDiffusionLM, EMA
from train import warm_start, expand_state_dict, CheckpointError, IDENTITY_ZERO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(core=2, seed=0, fine=0):
    c = get_preset("hier")
    dif = replace(c.diffusion, param="logits", self_cond=True, ce_weight=0.0, x0_rescale=True, t_sampling="warp")
    torch.manual_seed(seed)
    return ByteDiffusionLM(replace(c, diffusion=dif, model=replace(c.model, embed_norm=8, dropout=0.0, hier_core_blocks=core, hier_fine_blocks=fine)))


def perturb(net):  # give FiLM weight so every block's time/branch path is active
    for mod in net.modules():
        if hasattr(mod, "film"):
            torch.nn.init.normal_(mod.film.weight, std=0.05)


class HierExpand(unittest.TestCase):
    def test_param_count(self):
        n = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(n(build(2)), 14_062_272)
        self.assertEqual(n(build(4)), 21_654_720)

    def test_hash_unchanged_and_config_compat(self):
        h = get_preset("hier")
        self.assertEqual(h.model.hier_core_blocks, 2)
        h4 = replace(h, model=replace(h.model, hier_core_blocks=4))
        self.assertNotEqual(h.architecture_hash(), h4.architecture_hash())
        d = h.to_dict(); del d["model"]["hier_core_blocks"]  # configs stored before the field existed
        self.assertEqual(Config.from_dict(d).architecture_hash(), h.architecture_hash())
        hy = get_preset("hybrid")  # field is irrelevant (and hash-invisible) for other denoisers
        self.assertEqual(replace(hy, model=replace(hy.model, hier_core_blocks=4)).architecture_hash(), hy.architecture_hash())
        validate(h4)
        with self.assertRaises(ValueError):
            validate(replace(h, model=replace(h.model, hier_core_blocks=0)))
        for run in ("hier_10min_s0", "rung_hyb_selfcond"):  # stored configs hash to the hash recorded in their meta
            p = os.path.join(ROOT, "runs", run)
            if os.path.exists(os.path.join(p, "meta.json")):
                c = Config.from_dict(json.load(open(os.path.join(p, "config.json"))))
                self.assertEqual(c.architecture_hash(), json.load(open(os.path.join(p, "meta.json")))["architecture_hash"])

    def _ckpt(self, d):
        old = build(2, seed=1).eval(); perturb(old.denoise_unet)
        ema_old = EMA(old, decay=0.999)
        with torch.no_grad():  # EMA weights differ from the raw ones
            for p in ema_old.model.parameters():
                p.add_(0.01 * torch.randn_like(p))
        path = os.path.join(d, "c.pt")
        torch.save({"model_state_dict": old.state_dict(), "ema_state_dict": ema_old.state_dict()}, path)
        return old, ema_old, path

    def test_identity_at_init_model_and_ema(self):
        with tempfile.TemporaryDirectory() as d:
            old, ema_old, path = self._ckpt(d)
            new = build(4, seed=2).eval(); ema_new = EMA(new, decay=0.999)
            warm_start(path, new, ema_new)
        for blk in list(new.denoise_unet.core)[2:] + list(ema_new.model.denoise_unet.core)[2:]:
            for n, p in blk.named_parameters():
                if n in IDENTITY_ZERO:
                    self.assertEqual(float(p.abs().sum()), 0.0, n)
            perturb(blk)  # identity must not depend on FiLM or the inner weights being benign
        e = old.cfg.model.byte_embed_dim
        x = torch.randn(3, 256, e); xp = torch.randn(3, 256, e); t = torch.tensor([3, 400, 900])
        with torch.no_grad():
            for a, b in ((old, new), (ema_old.model, ema_new.model)):
                ya = a.gaussian_diffusion.denoise(x, t, x0_prev=xp)
                yb = b.gaussian_diffusion.denoise(x, t, x0_prev=xp)
                self.assertLess(float((ya - yb).abs().max()), 1e-6)
            # the EMA copy really differs from the raw model, so this is not the same weights loaded twice
            self.assertGreater(float((old.gaussian_diffusion.denoise(x, t) - ema_old.model.gaussian_diffusion.denoise(x, t)).abs().max()), 1e-4)
        # copied tensors are exact
        for k, v in old.denoise_unet.state_dict().items():
            self.assertTrue(torch.equal(v, new.denoise_unet.state_dict()[k]), k)

    def test_new_blocks_learn(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, path = self._ckpt(d)
            new = build(4, seed=2).train(); warm_start(path, new)
        blocks = list(new.denoise_unet.core)[2:]
        batch = torch.randint(0, 256, (4, 256))
        loss, _ = new(batch); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for blk in blocks:  # output projections of both residual branches receive gradient at step 0
            for n in IDENTITY_ZERO:
                g = blk.get_parameter(n).grad
                self.assertTrue(torch.isfinite(g).all()); self.assertGreater(float(g.abs().sum()), 0.0, n)
        with torch.no_grad():
            for p in new.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-1e-3)
        new.zero_grad(); loss, _ = new(batch); loss.backward()
        dead = [(i, n) for i, blk in enumerate(blocks) for n, p in blk.named_parameters() if not (p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0)]
        self.assertEqual(dead, [])  # after one step every tensor of the new blocks (inner weights, FiLM) gets gradient too

    def test_fine_blocks_identity_and_hash(self):
        h = get_preset("hier")
        self.assertEqual(h.model.hier_fine_blocks, 0)
        d = h.to_dict(); del d["model"]["hier_fine_blocks"]
        self.assertEqual(Config.from_dict(d).architecture_hash(), h.architecture_hash())
        h1 = replace(h, model=replace(h.model, hier_fine_blocks=1))
        self.assertNotEqual(h1.architecture_hash(), h.architecture_hash()); validate(h1)
        with self.assertRaises(ValueError):
            validate(replace(h, model=replace(h.model, hier_fine_blocks=-1)))
        n = lambda m: sum(p.numel() for p in m.parameters())
        added = n(build(2, fine=1)) - n(build(2))
        print("one fine enc+dec pair adds", added)
        with tempfile.TemporaryDirectory() as dd:
            old, ema_old, path = self._ckpt(dd)
            new = build(2, seed=2, fine=1).eval(); ema_new = EMA(new, decay=0.999)
            warm_start(path, new, ema_new)
        for m in (new, ema_new.model):
            for blk in list(m.denoise_unet.fine_enc_x) + list(m.denoise_unet.fine_dec_x):
                for k, p in blk.named_parameters():
                    if k in IDENTITY_ZERO:
                        self.assertEqual(float(p.abs().sum()), 0.0, k)
                perturb(blk)
        e = old.cfg.model.byte_embed_dim
        x = torch.randn(3, 256, e); xp = torch.randn(3, 256, e); t = torch.tensor([3, 400, 900])
        with torch.no_grad():
            for a, b in ((old, new), (ema_old.model, ema_new.model)):
                ya = a.gaussian_diffusion.denoise(x, t, x0_prev=xp); yb = b.gaussian_diffusion.denoise(x, t, x0_prev=xp)
                self.assertLess(float((ya - yb).abs().max()), 1e-6)
        loss, _ = new.train()(torch.randint(0, 256, (4, 256))); loss.backward()
        for blk in list(new.denoise_unet.fine_enc_x) + list(new.denoise_unet.fine_dec_x):
            for k in IDENTITY_ZERO:
                self.assertGreater(float(blk.get_parameter(k).grad.abs().sum()), 0.0, k)

    def test_other_missing_keys_still_error(self):
        a, b = build(2), build(2)
        sd = a.state_dict(); del sd["denoise_unet.out.weight"]
        with self.assertRaises(CheckpointError):
            expand_state_dict(sd, b.state_dict())
        sd = build(4).state_dict()  # more blocks in the source than the target: unexpected keys
        with self.assertRaises(CheckpointError):
            expand_state_dict(sd, b.state_dict())


if __name__ == "__main__":
    unittest.main()
