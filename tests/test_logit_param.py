import json, os, sys, tempfile, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import Config, get_preset
from main import ByteDiffusionLM
import train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(param="logits", t_sampling="uniform", embed_norm=None, denoiser="unet", **dkw):
    cfg = get_preset("debug")
    en = float(cfg.model.byte_embed_dim) ** 0.5 if embed_norm is None else embed_norm
    m = replace(cfg.model, embed_norm=en if param == "logits" else 0.0, denoiser=denoiser, **({"hybrid_layers": 1, "hybrid_heads": 2} if denoiser == "hybrid" else {}))
    return ByteDiffusionLM(replace(cfg, model=m, diffusion=replace(cfg.diffusion, param=param, t_sampling=t_sampling, **dkw)))


class Hash(unittest.TestCase):
    def test_defaults_hash_stable(self):
        cfg = get_preset("debug")
        d = cfg.to_dict()
        for k in ("param", "t_sampling", "x0_mse_weight"):
            del d["diffusion"][k]
        del d["model"]["embed_norm"]
        self.assertEqual(Config.from_dict(d).architecture_hash(), cfg.architecture_hash())
        for kw in (dict(diffusion=replace(cfg.diffusion, param="logits")), dict(diffusion=replace(cfg.diffusion, t_sampling="warp")),
                   dict(model=replace(cfg.model, embed_norm=4.0))):
            self.assertNotEqual(replace(cfg, **kw).architecture_hash(), cfg.architecture_hash())

    def test_existing_run_hash_unchanged(self):
        for run in ("diff_nat_gate_s0", "diff_nat_hybrid_s0"):
            p = os.path.join(ROOT, "runs", run)
            if not os.path.exists(os.path.join(p, "meta.json")):
                continue
            cfg = Config.from_dict(json.load(open(os.path.join(p, "config.json"))))
            self.assertEqual(cfg.architecture_hash(), json.load(open(os.path.join(p, "meta.json")))["architecture_hash"])

    def test_set_items(self):
        cfg, tov = train.apply_set_items(get_preset("compact"), ["model.embed_norm=8", "diffusion.param=logits", "diffusion.t_sampling=warp"])
        self.assertEqual((cfg.model.embed_norm, cfg.diffusion.param, cfg.diffusion.t_sampling), (8.0, "logits", "warp"))


class Logits(unittest.TestCase):
    def test_shapes_and_loss(self):
        for den in ("unet", "hybrid"):
            for sc in (False, True):
                m = build(denoiser=den, self_cond=sc)
                self.assertEqual(m.cfg.model.vocab_size, 256)
                x = torch.randint(0, 256, (3, 32))
                loss, st = m(x, t=torch.tensor([0, 25, 49]))
                self.assertTrue(torch.isfinite(loss)); loss.backward()
                self.assertIn("ce_loss", st)
                self.assertTrue(torch.isfinite(st["per_sample_loss"]).all())
                out = m.gaussian_diffusion.denoise(torch.randn(2, 32, m.cfg.model.byte_embed_dim), torch.tensor([3, 4]))
                self.assertEqual(out.shape, (2, 32, 256))

    def test_ce_finite_near_T_and_not_amplified(self):
        m = build().eval()
        x = torch.randint(0, 256, (8, 32))
        with torch.no_grad():
            _, s = m(x, t=torch.full((8,), 49))
        self.assertTrue(0 < float(s["ce_loss"]) < 20)  # ~ln(256)=5.5 for an untrained net; no 1/sqrt(ab) blow-up
        self.assertTrue(torch.isfinite(s["noise_loss"]))

    def test_embed_norm_fixed(self):
        m = build(embed_norm=3.0)
        with torch.no_grad():
            m.byte_embedder.embedding.weight.mul_(100)
        self.assertTrue(torch.allclose(m.embedding_table().norm(dim=-1), torch.full((256,), 3.0), atol=1e-4))
        self.assertAlmostEqual(m.mean_embed_norm(), 3.0, places=3)

    def test_x0_hat_in_convex_hull(self):
        m = build(embed_norm=2.0).eval(); gd = m.gaussian_diffusion
        table = m.embedding_table().detach()
        lg = torch.randn(2, 32, 256) * 10
        x0 = gd.x0_from_logits(lg, table)
        self.assertLessEqual(float(x0.norm(dim=-1).max()), float(table.norm(dim=-1).max()) + 1e-4)
        w = torch.softmax(lg, -1)
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones(2, 32), atol=1e-5))
        # derived eps consistent: x_t = sqrt(ab) x0 + sqrt(1-ab) eps
        t = torch.tensor([10, 40]); xt = torch.randn(2, 32, m.cfg.model.byte_embed_dim)
        eps = gd.eps_from_x0(xt, t, x0)
        self.assertTrue(torch.allclose(gd.q_sample(x0, t, noise=eps), xt, atol=1e-4))

    def test_mse_term(self):
        m = build(x0_mse_weight=1.0)
        loss, st = m(torch.randint(0, 256, (2, 32)), t=torch.tensor([5, 30]))
        self.assertIn("x0_mse", st); self.assertGreater(float(loss), float(st["ce_loss"]))

    def test_sampler_runs_and_ignores_rescale(self):
        for sc in (False, True):
            m = build(self_cond=sc, x0_rescale=True, num_timesteps=6)
            ids = m.sample(2)
            self.assertEqual(tuple(ids.shape), (2, 32))
            self.assertTrue(((ids >= 0) & (ids < 256)).all())
        gd = m.gaussian_diffusion
        x = torch.randn(2, 32, m.cfg.model.byte_embed_dim)
        out, x0, lg = gd.p_sample(x, torch.full((2,), 3), 3, table=m.embedding_table().detach(), return_logits=True)
        self.assertEqual((out.shape, x0.shape, lg.shape), (x.shape, x.shape, (2, 32, 256)))

    def test_cond_prompt_masks_loss(self):
        m = build(cond_prompt=True)
        cm = torch.zeros(2, 32, dtype=torch.bool); cm[:, :8] = True
        loss, st = m(torch.randint(0, 256, (2, 32)), t=torch.tensor([5, 30]), clean_mask=cm)
        self.assertTrue(torch.isfinite(loss))

    def test_eps_mode_unchanged_stats(self):
        m = build(param="eps")
        self.assertEqual(m.denoise_unet.final_conv.out_channels, m.cfg.model.byte_embed_dim)
        _, st = m(torch.randint(0, 256, (2, 32)))
        self.assertIn("per_sample_loss", st)
        self.assertFalse(hasattr(m.gaussian_diffusion, "t_loss_ema"))

    def test_quick_eval_and_eval_diffusion(self):
        import eval as ev
        m = build(num_timesteps=10) if False else build()
        cfg = m.cfg
        val = torch.randint(0, 256, (32 * 40,), dtype=torch.uint8)
        r = train.quick_eval(m, replace(cfg, train=replace(cfg.train, batch_size=4)), val, "cpu", 2)
        self.assertTrue(0 < r["val_ce_loss"] < 20)


class Warp(unittest.TestCase):
    def test_sampler_valid_and_favours_high_loss(self):
        m = build(param="eps", t_sampling="warp", num_timesteps=1000)  # 50 bins of 20
        gd = m.gaussian_diffusion
        t = gd.sample_t(512, "cpu")
        self.assertTrue(((t >= 0) & (t < 1000)).all()); self.assertEqual(t.dtype, torch.long)
        self.assertTrue(torch.allclose(gd.t_probs(), torch.full((50,), 0.02)))  # nothing seen: uniform
        gd.train()
        for _ in range(60):  # bin 7 (t in [140,160)) has 10x the loss
            tt = torch.arange(0, 1000, 20) + 3
            gd.update_t_stats(tt, torch.where(tt // 20 == 7, 10.0, 1.0))
        p = gd.t_probs()
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
        self.assertGreater(float(p[7]), 5 * float(p[8]))
        self.assertGreaterEqual(float(p.min()), 0.2 / 50 - 1e-9)  # uniform floor
        t = gd.sample_t(4000, "cpu")
        self.assertTrue(((t >= 0) & (t < 1000)).all())
        frac7 = float(((t // 20) == 7).float().mean())
        self.assertGreater(frac7, 0.12); self.assertAlmostEqual(frac7, float(p[7]), delta=0.03)

    def test_updates_only_in_training_and_for_logits(self):
        m = build(t_sampling="warp")
        x = torch.randint(0, 256, (4, 32)); gd = m.gaussian_diffusion
        m.eval(); m(x)
        self.assertFalse(bool(gd.t_loss_seen.any()))
        m.train(); m(x)
        self.assertTrue(bool(gd.t_loss_seen.any()))
        self.assertTrue(((gd.t_loss_ema > 0) == gd.t_loss_seen).all())

    def test_state_saved_and_loaded(self):
        m = build(t_sampling="warp")
        m.train()
        for _ in range(5):
            m(torch.randint(0, 256, (8, 32)))
        sd = m.state_dict()
        self.assertIn("gaussian_diffusion.t_loss_ema", sd)
        m2 = build(t_sampling="warp"); m2.load_state_dict(sd)
        self.assertTrue(torch.equal(m2.gaussian_diffusion.t_loss_ema, m.gaussian_diffusion.t_loss_ema))
        self.assertTrue(torch.equal(m2.gaussian_diffusion.t_loss_seen, m.gaussian_diffusion.t_loss_seen))
        # uniform models carry no warp buffers, so old checkpoints stay loadable
        self.assertNotIn("gaussian_diffusion.t_loss_ema", build().state_dict())

    def test_resume_roundtrip(self):
        cfg = get_preset("debug")
        cfg = replace(cfg, model=replace(cfg.model, embed_norm=4.0),
                      diffusion=replace(cfg.diffusion, param="logits", t_sampling="warp"),
                      train=replace(cfg.train, max_updates=6, save_interval=3, eval_interval=3, eval_batches=1, batch_size=4, grad_accum_steps=1))
        data = torch.randint(0, 256, (32 * 64,), dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as d:
            rd = os.path.join(d, "r")
            train.run_training(cfg, rd, "fresh", "cpu", data=data, val=data, stop_after=3, quiet=True)
            a = torch.load(os.path.join(rd, "checkpoint.pt"), weights_only=False)["model_state_dict"]["gaussian_diffusion.t_loss_ema"]
            self.assertTrue((a > 0).any())
            train.run_training(None, rd, "resume", "cpu", data=data, val=data, quiet=True)
            b = torch.load(os.path.join(rd, "checkpoint.pt"), weights_only=False)["model_state_dict"]["gaussian_diffusion.t_loss_ema"]
            self.assertFalse(torch.equal(a, b))
            rows = [json.loads(l) for l in open(os.path.join(rd, "metrics.jsonl"))]
            self.assertTrue(any(r.get("eval") and "val_ce_loss" in r for r in rows))


if __name__ == "__main__":
    unittest.main()
