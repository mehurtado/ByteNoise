import os, sys, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import get_preset
from main import ByteDiffusionLM, EMA, window_starts, _rope_fp32
from train import warm_start, load_model_from_run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, "runs", "rpw_s0")
OLD_TOTAL = 2_968_576


def build(attn, seed=0, seq_len=256):
    c = get_preset("rpw")
    dif = replace(c.diffusion, param="logits", self_cond=True, ce_weight=0.0, x0_rescale=True, t_sampling="warp")
    torch.manual_seed(seed)
    return ByteDiffusionLM(replace(c, diffusion=dif, model=replace(c.model, seq_len=seq_len, rpw_enc_attn=attn)))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class EncAttn(unittest.TestCase):
    def test_hash_unchanged_when_off(self):
        c = get_preset("rpw")
        self.assertEqual(replace(c, model=replace(c.model, rpw_enc_attn=False)).architecture_hash(), c.architecture_hash())
        self.assertNotEqual(replace(c, model=replace(c.model, rpw_enc_attn=True)).architecture_hash(), c.architecture_hash())

    def test_param_count(self):
        old, new = build(False), build(True)
        self.assertEqual(n_params(old), OLD_TOTAL)
        d = new.denoise_unet
        grp = d.group_counts()
        a = grp["encoder local attention (+norm, conditioner)"]
        self.assertEqual(a, 128 + 256 * 256 + 256 + 4 * 128 * 128)  # RMSNorm + FiLM Linear(256,256) + q,k,v,o
        # conv bank 4->2 dilations: -2*128*3 conv, -2*128*128 fusion columns
        self.assertEqual(n_params(new), OLD_TOTAL - 2 * 128 * 3 - 2 * 128 * 128 + a)
        print("new total", n_params(new), "delta", n_params(new) - OLD_TOTAL, grp)

    def test_window_coverage(self):
        s = window_starts(256)
        self.assertEqual(len(s), 15)
        cnt = torch.zeros(256)
        for a in s:
            cnt[a:a + 32] += 1
        self.assertTrue(torch.equal(cnt[:16], torch.ones(16)) and torch.equal(cnt[240:], torch.ones(16)))
        self.assertTrue(torch.equal(cnt[16:240], torch.full((224,), 2.0)))
        self.assertEqual(len(window_starts(512)), 31)
        with self.assertRaises(ValueError):
            window_starts(16)

    def test_shapes_lengths(self):
        for n in (256, 512):
            net = build(True, seq_len=n)
            loss, _ = net(torch.randint(0, 256, (2, n)))
            self.assertTrue(torch.isfinite(loss))
            out = net.denoise_unet(torch.randn(2, n, 128), torch.tensor([5, 700]))
            self.assertEqual(tuple(out.shape), (2, n, 256))

    def test_zero_output_attention_and_locality(self):
        net = build(True).eval(); d = net.denoise_unet; blk = d.enc[1]
        self.assertEqual(float(blk.attn.o.weight.abs().sum()), 0.0)
        for p in blk.parameters():
            p.data.add_(0.0)  # no-op; keep init
        # open the attention and conditioner, then check strict locality: bytes >32 away cannot affect the attention output at p
        blk.attn.o.weight.data.normal_(0, 0.1)
        h = torch.randn(1, 256, 128)
        cos, sin = _rope_fp32(torch.arange(256), 32)
        with torch.no_grad():
            a = blk.attn(h, (cos, sin))
            h2 = h.clone(); h2[:, 200:] += torch.randn(1, 56, 128)
            b = blk.attn(h2, (cos, sin))
        self.assertLess(float((a[:, :150] - b[:, :150]).abs().max()), 1e-5)   # untouched far from the change
        self.assertGreater(float((a[:, 200:] - b[:, 200:]).abs().max()), 1e-3)

    def test_gradients(self):
        net = build(True).train(); batch = torch.randint(0, 256, (4, 256))
        torch.manual_seed(1); loss, _ = net(batch); loss.backward()
        blk = net.denoise_unet.enc[1]
        self.assertGreater(float(blk.attn.o.weight.grad.abs().sum()), 0.0)
        # o is zero at init: q/k/v, the attention norm and its conditioner get zero gradient on the first backward only
        for m in (blk.attn.q, blk.attn.k, blk.attn.v, blk.acond):
            self.assertEqual(sum(float(p.grad.abs().sum()) for p in m.parameters()), 0.0)
        with torch.no_grad():
            for p in net.parameters():
                p.add_(p.grad, alpha=-1e-2)
        net.zero_grad(); torch.manual_seed(1); loss, _ = net(batch); loss.backward()
        for name, m in (("q", blk.attn.q), ("k", blk.attn.k), ("v", blk.attn.v), ("o", blk.attn.o), ("na", blk.na), ("acond", blk.acond)):
            for p in m.parameters():
                self.assertTrue(torch.isfinite(p.grad).all(), name); self.assertGreater(float(p.grad.abs().sum()), 0.0, name)
        for k, p in blk.named_parameters():
            self.assertIsNotNone(p.grad, k)

    @unittest.skipUnless(os.path.exists(os.path.join(RUN, "checkpoint.pt")), "needs runs/rpw_s0")
    def test_warm_start_model_and_ema(self):
        ck = torch.load(os.path.join(RUN, "checkpoint.pt"), map_location="cpu", weights_only=False)
        new = build(True); ema = EMA(new, decay=0.999, warmup=True)
        warm_start(os.path.join(RUN, "checkpoint.pt"), new, ema)
        pre = "denoise_unet."
        for name, sd_new, sd_old in (("model", new.state_dict(), ck["model_state_dict"]),
                                     ("ema", ema.model.state_dict(), {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in ck["ema_state_dict"].items()})):
            dropped = [k for k in sd_old if k.startswith(pre) and k not in sd_new]
            self.assertEqual(sorted(dropped), sorted(f"{pre}enc.1.convs.{i}.weight" for i in (2, 3)), name)
            added = [k for k in sd_new if k not in sd_old]
            self.assertTrue(added and all(("enc.1." in k) and k.startswith(("denoise_unet.", "gaussian_diffusion.denoise_model.")) and any(t in k for t in (".na.", ".acond.", ".attn.")) for k in added), (name, added))
            for k, v in sd_new.items():
                if k in sd_old and not k.endswith("enc.1.mix.weight"):
                    self.assertTrue(torch.equal(v, sd_old[k]), (name, k))
            for mk in (pre + "enc.1.mix.weight", "gaussian_diffusion.denoise_model.enc.1.mix.weight"):
                self.assertTrue(torch.equal(sd_new[mk], sd_old[mk][:, :256]), (name, mk))
            self.assertEqual(float(sd_new[pre + "enc.1.attn.o.weight"].abs().sum()), 0.0, name)
            self.assertEqual(float(sd_new[pre + "enc.1.acond.weight"].abs().sum() + sd_new[pre + "enc.1.acond.bias"].abs().sum()), 0.0, name)
        self.assertEqual([c.dilation[0] for c in new.denoise_unet.enc[1].convs], [1, 2])

    @unittest.skipUnless(os.path.exists(os.path.join(RUN, "checkpoint.pt")), "needs runs/rpw_s0")
    def test_old_checkpoint_still_loads(self):
        m, cfg = load_model_from_run(RUN, "ema", "cpu")
        self.assertEqual(n_params(m), OLD_TOTAL)
        self.assertFalse(cfg.model.rpw_enc_attn)


if __name__ == "__main__":
    unittest.main()
