import os, sys, tempfile, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import Config, get_preset, validate
from main import ByteDiffusionLM, _rope_fp32
from models_lm import apply_rotary

EXPECTED = {"input projection": 16_384, "time MLP": 131_072, "byte conditioner": 131_584, "workspace conditioner": 197_376,
            "local blocks": 662_528, "pack norm+projection": 98_816, "read attention+norms": 123_200, "workspace blocks": 1_476_096,
            "write attention+norms": 82_240, "final norm+output": 32_896}
TOTAL, EMBED = 2_968_576, 16_384


def build(seed=0, seq_len=256):
    c = get_preset("rpw")
    dif = replace(c.diffusion, param="logits", self_cond=True, ce_weight=0.0, x0_rescale=True, t_sampling="warp")
    torch.manual_seed(seed)
    return ByteDiffusionLM(replace(c, diffusion=dif, model=replace(c.model, seq_len=seq_len)))


def rep(net):
    return torch.randint(0, 256, (4, net.cfg.model.seq_len))


class RPW(unittest.TestCase):
    def test_counts(self):
        net = build()
        n = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(n(net.byte_embedder), EMBED)
        self.assertEqual(net.denoise_unet.group_counts(), EXPECTED)
        self.assertEqual(n(net), TOTAL)
        self.assertEqual(n(net.denoise_unet) + EMBED, TOTAL)  # no second embedding hiding in the denoiser
        self.assertFalse([k for k, _ in net.denoise_unet.named_parameters() if "embed" in k])
        print("params total", n(net), EXPECTED)

    def test_hash_and_validation(self):
        c = get_preset("rpw"); validate(c)
        for k, v in (("model_dim", 100), ("rpw_workspace_dim", 100), ("seq_len", 250)):
            with self.assertRaises(ValueError):
                validate(replace(c, model=replace(c.model, **{k: v})))
        hy = get_preset("hybrid")  # field invisible for other denoisers
        self.assertEqual(replace(hy, model=replace(hy.model, rpw_workspace_dim=96)).architecture_hash(), hy.architecture_hash())
        self.assertNotEqual(replace(c, model=replace(c.model, rpw_workspace_dim=96)).architecture_hash(), c.architecture_hash())

    def test_lengths_finite_and_invalid(self):
        for n in (256, 512):
            net = build(seq_len=n)
            loss, stats = net(rep(net))
            self.assertTrue(torch.isfinite(loss))
            e = net.cfg.model.byte_embed_dim
            out = net.denoise_unet(torch.randn(2, n, 2 * e), torch.tensor([5, 700]))
            self.assertEqual(tuple(out.shape), (2, n, 256)); self.assertTrue(torch.isfinite(out).all())
        with self.assertRaisesRegex(ValueError, "divisible by four"):
            build().denoise_unet(torch.randn(1, 250, 128), torch.tensor([3]))

    def test_fractional_rope(self):
        pos = 4 * torch.arange(64) + 1.5
        cos, sin = _rope_fp32(pos, 32)
        self.assertEqual(tuple(cos.shape), (64, 16)); self.assertEqual(cos.dtype, torch.float32)
        with torch.autocast("cpu", dtype=torch.bfloat16):  # tables stay fp32 under autocast; rotated q keeps its own dtype
            cos2, _ = _rope_fp32(pos, 32)
        self.assertEqual(cos2.dtype, torch.float32)
        q = torch.randn(2, 6, 64, 32, dtype=torch.bfloat16)
        self.assertEqual(apply_rotary(q, cos, sin).dtype, torch.bfloat16)
        self.assertAlmostEqual(float(pos[0]), 1.5); self.assertAlmostEqual(float(pos[1]), 5.5)
        # relative offsets are honoured: shifting both q and k coordinates by a constant leaves q.k unchanged
        qq, kk = torch.randn(1, 1, 3, 32), torch.randn(1, 1, 5, 32)
        s = lambda off: (apply_rotary(qq, *_rope_fp32(torch.tensor([1.5, 5.5, 9.5]) + off, 32)) @ apply_rotary(kk, *_rope_fp32(torch.arange(5.0) + off, 32)).transpose(-1, -2))
        self.assertLess(float((s(0.0) - s(37.0)).abs().max()), 1e-3)

    def test_write_back_changes_logits(self):
        net = build().eval(); d = net.denoise_unet
        for p in d.parameters():  # open conditioners so the check does not depend on a zero-init path
            p.data.add_(0.01 * torch.randn_like(p))
        x, t = torch.randn(2, 256, 128), torch.tensor([10, 600])
        with torch.no_grad():
            a = d(x, t); d.skip_write = True; b = d(x, t); d.skip_write = False
            c = d(x, t)
        self.assertGreater(float((a - b).abs().max()), 1e-4)
        self.assertTrue(torch.equal(a, c))

    def test_gradients_and_embedding(self):
        net = build().train()
        batch = rep(net)
        torch.manual_seed(1); loss, _ = net(batch); loss.backward()
        d = net.denoise_unet
        groups = {"input": [d.inp], "encoder": list(d.enc), "pack": [d.pack_norm, d.pack], "read": [d.read_nq, d.read_nk, d.read], "workspace": list(d.work),
                  "write": [d.write_nq, d.write_nk, d.write], "decoder": list(d.dec), "final": [d.final_norm, d.out], "byte_cond": [d.byte_cond], "work_cond": [d.work_cond]}
        for name, mods in groups.items():
            for m in mods:
                for k, p in m.named_parameters():
                    self.assertIsNotNone(p.grad, f"{name}.{k}")
                    self.assertTrue(torch.isfinite(p.grad).all(), f"{name}.{k}")
                    self.assertGreater(float(p.grad.abs().sum()), 0.0, f"{name}.{k}")
        # zero-initialised conditioners block the upstream time MLP on the first backward only
        self.assertEqual(sum(float(p.grad.abs().sum()) for p in d.time_mlp.parameters() if p.grad is not None), 0.0)
        g = net.byte_embedder.embedding.weight.grad
        self.assertTrue(torch.isfinite(g).all()); self.assertGreater(float(g.abs().sum()), 0.0)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(p.grad, alpha=-1e-2)
        net.zero_grad(); torch.manual_seed(1); loss, _ = net(batch); loss.backward()  # fresh forward after the update opened the conditioners
        for p in d.time_mlp.parameters():
            self.assertTrue(torch.isfinite(p.grad).all()); self.assertGreater(float(p.grad.abs().sum()), 0.0)

    def test_save_load_roundtrip(self):
        a = build(seed=1).eval()
        for p in a.parameters():
            p.data.add_(0.01 * torch.randn_like(p))
        with tempfile.TemporaryDirectory() as dd:
            path = os.path.join(dd, "m.pt"); torch.save(a.state_dict(), path)
            b = build(seed=2).eval(); b.load_state_dict(torch.load(path))
        x, t = torch.randn(2, 256, 128), torch.tensor([10, 600])
        with torch.no_grad():
            self.assertTrue(torch.equal(a.denoise_unet(x, t), b.denoise_unet(x, t)))


if __name__ == "__main__":
    unittest.main()
