import copy, os, sys, tempfile, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import Config, get_preset, validate
from main import ByteDiffusionLM, EMA, word_segment_ids
import train


def build(tied=False, span=0.0, denoiser="unet", embed_norm=4.0, **dkw):
    cfg = get_preset("debug")
    m = replace(cfg.model, embed_norm=embed_norm, tied_head=tied, denoiser=denoiser,
                **({"hybrid_layers": 1, "hybrid_heads": 2} if denoiser == "hybrid" else {}))
    return ByteDiffusionLM(replace(cfg, model=m, diffusion=replace(cfg.diffusion, param="logits", span_noise_prob=span, **dkw)))


class Hash(unittest.TestCase):
    def test_defaults_hash_stable_and_options_change_it(self):
        cfg = replace(get_preset("debug"))
        d = cfg.to_dict()
        del d["model"]["tied_head"], d["diffusion"]["span_noise_prob"]
        self.assertEqual(Config.from_dict(d).architecture_hash(), cfg.architecture_hash())  # old configs load, hash unchanged
        self.assertNotEqual(replace(cfg, model=replace(cfg.model, tied_head=True)).architecture_hash(), cfg.architecture_hash())
        self.assertNotEqual(replace(cfg, diffusion=replace(cfg.diffusion, span_noise_prob=0.3)).architecture_hash(), cfg.architecture_hash())

    def test_validate(self):
        cfg = get_preset("debug")
        with self.assertRaises(ValueError):
            validate(replace(cfg, model=replace(cfg.model, tied_head=True)))  # needs param=logits
        with self.assertRaises(ValueError):
            validate(replace(cfg, diffusion=replace(cfg.diffusion, span_noise_prob=0.5)))
        with self.assertRaises(ValueError):
            validate(replace(cfg, diffusion=replace(cfg.diffusion, param="logits", span_noise_prob=1.5)))

    def test_set_items(self):
        cfg, _ = train.apply_set_items(get_preset("compact"), ["model.tied_head=true", "diffusion.span_noise_prob=0.3"])
        self.assertEqual((cfg.model.tied_head, cfg.diffusion.span_noise_prob), (True, 0.3))


class Tied(unittest.TestCase):
    def test_shapes_params_and_init_scale(self):
        for den in ("unet", "hybrid"):
            m, u = build(tied=True, denoiser=den), build(tied=False, denoiser=den)
            self.assertEqual(m.gaussian_diffusion.denoise(torch.randn(2, 32, 16), torch.tensor([3, 4])).shape, (2, 32, 256))
            self.assertLess(sum(p.numel() for p in m.parameters()), sum(p.numel() for p in u.parameters()))
            self.assertAlmostEqual(float(m.gaussian_diffusion.head.log_scale.exp()), 1 / 4.0, places=5)
            self.assertEqual(sum(k.endswith("embedding.weight") for k in m.state_dict()), 1)  # table registered once
        lg = m.gaussian_diffusion.denoise(torch.randn(4, 32, 16), torch.tensor([3, 4, 5, 6]))
        self.assertLess(float(lg.std()), 3.0)  # initial logits are O(1)

    def test_gradient_paths(self):
        m = build(tied=True)
        x = torch.randint(0, 256, (4, 32))
        loss, _ = m(x, t=torch.tensor([0, 10, 25, 49])); loss.backward()
        self.assertTrue(m.gaussian_diffusion.head.log_scale.grad.abs() > 0)
        self.assertTrue(m.byte_embedder.embedding.weight.grad.abs().sum() > 0)
        # head path alone reaches E (not through the input embedding) ...
        m.zero_grad()
        z = torch.randn(2, 32, 16)
        m.gaussian_diffusion.head(z).sum().backward()
        self.assertTrue(m.byte_embedder.embedding.weight.grad.abs().sum() > 0)
        # ... while x0_hat = softmax(logits) @ E does not send gradient into E
        m.zero_grad()
        lg = m.gaussian_diffusion.head(z).detach().requires_grad_(True)
        m.gaussian_diffusion.x0_from_logits(lg, m.embedding_table()).sum().backward()
        self.assertIsNone(m.byte_embedder.embedding.weight.grad)

    def test_embed_norm_zero_and_ema_and_roundtrip(self):
        m = build(tied=True, embed_norm=0.0)
        self.assertAlmostEqual(float(m.gaussian_diffusion.head.log_scale.exp()), 1 / 4.0, places=5)  # 1/sqrt(16)
        ema = EMA(m)
        self.assertIs(ema.model.gaussian_diffusion.head._emb[0], ema.model.byte_embedder)  # deepcopy points at the copied embedder
        with tempfile.TemporaryDirectory() as d:
            torch.save(m.state_dict(), os.path.join(d, "a.pt"))
            m2 = build(tied=True, embed_norm=0.0)
            m2.load_state_dict(torch.load(os.path.join(d, "a.pt")))
        x = torch.randint(0, 256, (2, 32)); t = torch.tensor([5, 30]); noise = torch.randn(2, 32, 16)
        m.eval(); m2.eval()
        self.assertTrue(torch.allclose(m(x, t=t, noise=noise)[0], m2(x, t=t, noise=noise)[0]))

    def test_sampler_runs(self):
        m = build(tied=True).eval()
        self.assertEqual(m.sample(2).shape, (2, 32))


class Segments(unittest.TestCase):
    def test_word_segments(self):
        x = torch.tensor([list(b"ab, cd e\n\xc3\xa9x. ")])
        # "ab" | ", cd" | " e" | "\n\xc3\xa9x" (multibyte stays inside the word) | ". " trailing
        self.assertEqual(word_segment_ids(x)[0].tolist(), [0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4])
        self.assertEqual(word_segment_ids(torch.tensor([[32, 33, 34]]))[0].tolist(), [0, 0, 0])
        self.assertEqual(word_segment_ids(torch.tensor([[65, 66]]))[0].tolist(), [0, 0])


class Span(unittest.TestCase):
    def test_p0_identical_to_before(self):
        torch.manual_seed(0); a = build(span=0.0)
        torch.manual_seed(0); b = build(span=0.0)
        x = torch.randint(0, 256, (4, 32)); t = torch.tensor([0, 10, 25, 49]); noise = torch.randn(4, 32, 16)
        self.assertEqual(float(a(x, t=t, noise=noise)[0]), float(b(x, t=t, noise=noise)[0]))
        self.assertEqual(a.denoise_unet.init_conv.in_channels, 16)  # no extra channel
        # p>0 in eval mode (val CE) is the ordinary single-t noising, with a constant sqrt(ab) channel
        c = build(span=0.5).eval()
        _, st = c(x, t=t, noise=noise)
        self.assertEqual(st["t"].tolist(), t.tolist())
        self.assertEqual(c.denoise_unet.init_conv.in_channels, 17)

    def test_p1_distinct_t_within_example(self):
        m = build(span=1.0).train()
        gd = m.gaussian_diffusion
        txt = torch.tensor([list(b"the quick brown fox jumps over a lazy dog now")[:32] + [32] * 0]).repeat(6, 1)
        txt = torch.cat([txt, torch.full((6, 32 - txt.size(1)), 32)], 1) if txt.size(1) < 32 else txt
        t = torch.full((6,), 25)
        te, tp = gd.span_timesteps(txt, t)
        self.assertEqual(tp.shape, (6, 32))
        self.assertTrue(all(len(set(r)) > 1 for r in tp.tolist()))
        seg = word_segment_ids(txt)
        for r in range(6):  # constant inside a segment
            for s in seg[r].unique():
                self.assertEqual(len(set(tp[r][seg[r] == s].tolist())), 1)
        self.assertTrue((te != 25).any())
        gd.span_noise_prob = 0.0 + 1e-9  # p ~ 0: nobody switches, t_pos == t everywhere
        te, tp = gd.span_timesteps(txt, t)
        self.assertTrue((tp == 25).all() and (te == 25).all())

    def test_mixed_batch_fraction_and_warp(self):
        m = build(span=0.5, t_sampling="warp").train()
        gd = m.gaussian_diffusion
        x = torch.randint(0, 256, (200, 32))
        te, tp = gd.span_timesteps(x, torch.full((200,), 7))
        varied = (tp != tp[:, :1]).any(1).float().mean()
        self.assertTrue(0.3 < float(varied) < 0.7)

    def test_train_step_shapes_and_selfcond_hybrid(self):
        for den in ("unet", "hybrid"):
            for sc in (False, True):
                m = build(span=0.6, tied=True, denoiser=den, self_cond=sc).train()
                x = torch.randint(0, 256, (4, 32))
                loss, st = m(x)
                self.assertTrue(torch.isfinite(loss)); loss.backward()
                self.assertEqual(st["t"].shape, (4,))
                self.assertTrue(torch.isfinite(st["noise_loss"]))
                self.assertTrue(all(p.grad is not None for p in m.denoise_unet.parameters()))
                m.eval()
                self.assertEqual(m.gaussian_diffusion.denoise(torch.randn(2, 32, 16), torch.tensor([3, 4])).shape, (2, 32, 256))
                self.assertEqual(m.sample(2).shape, (2, 32))

    def test_channel_value_fed_to_network(self):
        m = build(span=1.0).eval()
        gd = m.gaussian_diffusion
        seen = []
        orig = gd.denoise_model.forward
        gd.denoise_model.forward = lambda x, t: (seen.append(x), orig(x, t))[1]
        t = torch.tensor([5, 40])
        gd.denoise(torch.randn(2, 32, 16), t)
        ch = seen[0][..., -1]
        self.assertTrue(torch.allclose(ch, gd.sqrt_alphas_cumprod[t][:, None].expand(2, 32)))

    def test_checkpoint_roundtrip(self):
        cfg = replace(get_preset("debug"), model=replace(get_preset("debug").model, embed_norm=4.0, tied_head=True),
                      diffusion=replace(get_preset("debug").diffusion, param="logits", span_noise_prob=0.5, t_sampling="warp"),
                      train=replace(get_preset("debug").train, max_updates=4, save_interval=2, batch_size=2, grad_accum_steps=1))
        data = (bytes(range(65, 123)) * 40)
        with tempfile.TemporaryDirectory() as d:
            r = train.run_training(cfg, d, "fresh", device="cpu", data=data, val=data, quiet=True, stop_after=2)
            self.assertEqual(r["progress"]["updates"], 2)
            r2 = train.run_training(None, d, "resume", device="cpu", data=data, val=data, quiet=True)
            self.assertEqual(r2["progress"]["updates"], 4)
            model, c = train.load_model_from_run(d, "ema", "cpu")
            self.assertTrue(c.model.tied_head and c.diffusion.span_noise_prob == 0.5)
            self.assertEqual(model.sample(1).shape, (1, 32))


if __name__ == "__main__":
    unittest.main()
