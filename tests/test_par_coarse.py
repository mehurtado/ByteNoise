import json, os, sys, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import Config, validate
from main import ByteDiffusionLM, EMA
from train import warm_start

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "runs", "diff_nat_v2_hybrid_selfcond_warm_s0")


def cfg(par):
    c = Config.from_dict(json.load(open(os.path.join(RUN, "config.json"))))
    return replace(c, model=replace(c.model, hybrid_par_coarse=par, dropout=0.0))


@unittest.skipUnless(os.path.exists(os.path.join(RUN, "checkpoint.pt")), "warm-start checkpoint missing")
class ParCoarse(unittest.TestCase):
    def test_hash_params_identity_gradients(self):
        base, par = cfg(0), cfg(1)
        stored = Config.from_dict(json.load(open(os.path.join(RUN, "config.json"))))
        self.assertEqual(stored.architecture_hash(), json.load(open(os.path.join(RUN, "meta.json")))["architecture_hash"])
        self.assertEqual(replace(stored, model=replace(stored.model, hybrid_par_coarse=0)).architecture_hash(), stored.architecture_hash())
        self.assertNotEqual(par.architecture_hash(), stored.architecture_hash()); validate(par)
        n = lambda m: sum(p.numel() for p in m.parameters())
        old, new = ByteDiffusionLM(base).eval(), ByteDiffusionLM(par).eval()
        print("params flat", n(old), "with branch", n(new), "added", n(new) - n(old))
        ema_old, ema_new = EMA(old, decay=0.999), EMA(new, decay=0.999)
        path = os.path.join(RUN, "checkpoint.pt")
        warm_start(path, old, ema_old); warm_start(path, new, ema_new)  # both from the real checkpoint (model + EMA)
        e = base.model.byte_embed_dim
        torch.manual_seed(0)
        x = torch.randn(3, 256, e); xp = torch.randn(3, 256, e); t = torch.tensor([3, 400, 900])
        with torch.no_grad():
            for a, b in ((old, new), (ema_old.model, ema_new.model)):
                ya = a.gaussian_diffusion.denoise(x, t, x0_prev=xp); yb = b.gaussian_diffusion.denoise(x, t, x0_prev=xp)
                self.assertLess(float((ya - yb).abs().max()), 1e-6)
            self.assertGreater(float((old.gaussian_diffusion.denoise(x, t) - ema_old.model.gaussian_diffusion.denoise(x, t)).abs().max()), 1e-5)
        pc = new.denoise_unet.par
        batch = torch.randint(0, 256, (4, 256))
        new.train(); loss, _ = new(batch); loss.backward()
        self.assertGreater(float(pc.gate.grad.abs().sum()), 0.0)  # the gate learns from step 0
        self.assertEqual(float(pc.block.fuse.weight.grad.abs().sum()), 0.0)  # branch is closed while gate == 0
        with torch.no_grad():
            for p in new.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-1e-2)
        new.zero_grad(); loss, _ = new(batch); loss.backward()
        self.assertGreater(float(pc.gate.abs().sum()), 0.0)
        for name, q in (("pin", pc.pin.weight), ("block.fuse", pc.block.fuse.weight), ("pout", pc.pout.weight)):
            self.assertTrue(torch.isfinite(q.grad).all()); self.assertGreater(float(q.grad.abs().sum()), 0.0, name)


if __name__ == "__main__":
    unittest.main()
