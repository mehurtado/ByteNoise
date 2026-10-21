import os, sys, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import get_preset
from main import ByteDiffusionLM, _rope_fp32

OLD_TOTAL = 3_066_496  # rpw + rpw_enc_attn


def build(rep, seed=0, enc_attn=True):
    c = get_preset("rpw")
    dif = replace(c.diffusion, param="logits", self_cond=True, ce_weight=0.0, x0_rescale=True, t_sampling="warp")
    torch.manual_seed(seed)
    return ByteDiffusionLM(replace(c, diffusion=dif, model=replace(c.model, seq_len=256, rpw_enc_attn=enc_attn, rpw_local_repeats=rep)))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def reference_forward(d, x_embed, time):
    """Verbatim copy of RPWDenoiser.forward before rpw_local_repeats existed."""
    b, n, _ = x_embed.shape
    t = d.time_mlp(time)
    cb = [c[:, None] for c in d.byte_cond(t).chunk(4, dim=-1)]
    cw = [c[:, None] for c in d.work_cond(t).chunk(4, dim=-1)]
    rb = _rope_fp32(torch.arange(n, device=x_embed.device), 32)
    rs = _rope_fp32(4 * torch.arange(n // 4, device=x_embed.device) + 1.5, 32)
    x = d.inp(x_embed)
    for blk in d.enc:
        x = blk(x, cb, t, rb)
    z = d.pack(d.pack_norm(x.reshape(b, n // 4, -1)))
    z = z + d.read(d.read_nq(z), d.read_nk(x), rs, rb)
    for blk in d.work:
        z = blk(z, cw, rs)
    x = x + d.write(d.write_nq(x), d.write_nk(z), rb, rs)
    for blk in d.dec:
        x = blk(x, cb, t, rb)
    return d.out(d.final_norm(x))


def randomize(m, seed=5):
    g = torch.Generator().manual_seed(seed)  # zero-init layers would hide differences; give every param signal
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() > 1:
                p.add_(torch.randn(p.shape, generator=g) * 0.02)


class Repeats(unittest.TestCase):
    def test_hash_and_default(self):
        c = get_preset("rpw")
        self.assertEqual(c.model.rpw_local_repeats, 1)
        self.assertEqual(replace(c, model=replace(c.model, rpw_local_repeats=1)).architecture_hash(), c.architecture_hash())
        self.assertNotEqual(replace(c, model=replace(c.model, rpw_local_repeats=2)).architecture_hash(), c.architecture_hash())

    def test_old_config_dict_loads(self):
        c = get_preset("rpw"); d = c.to_dict(); d["model"].pop("rpw_local_repeats")
        self.assertEqual(type(c).from_dict(d).model.rpw_local_repeats, 1)

    def test_repeats1_bit_identical(self):
        m = build(1); randomize(m); m.eval()
        den = m.denoise_unet
        g = torch.Generator().manual_seed(11)
        x = torch.randn(3, 256, 128, generator=g); t = torch.tensor([10.0, 500.0, 900.0])
        with torch.no_grad():
            self.assertTrue(torch.equal(den(x, t), reference_forward(den, x, t)))

    def test_repeats2_shape_params_grads(self):
        m1, m2 = build(1), build(2)
        self.assertEqual(n_params(m1), OLD_TOTAL)
        self.assertEqual(n_params(m2), OLD_TOTAL)
        self.assertEqual(list(m1.state_dict().keys()), list(m2.state_dict().keys()))
        randomize(m2); m2.load_state_dict(m2.state_dict()); m1.load_state_dict(m2.state_dict())
        for mm in (m1, m2): mm.eval()
        x = torch.randn(2, 256, 128); t = torch.tensor([100.0, 800.0])
        with torch.no_grad():
            o1, o2 = m1.denoise_unet(x, t), m2.denoise_unet(x, t)
        self.assertEqual(o1.shape, o2.shape)
        self.assertFalse(torch.equal(o1, o2))  # repeats actually change the computation
        m2.train(); X = torch.randint(0, 256, (4, 256))
        loss, _ = m2(X); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for n_, p in m2.named_parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all(), n_)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in m2.denoise_unet.enc.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in m2.denoise_unet.dec.parameters()))

    def test_repeats2_without_enc_attn(self):
        m = build(2, enc_attn=False); self.assertEqual(n_params(m), n_params(build(1, enc_attn=False)))


if __name__ == "__main__":
    unittest.main()
