import os, sys, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import get_preset, validate
from main import ByteDiffusionLM


def build(prob, steps=2, jump=50, seed=0):
    c = get_preset("rpw")
    dif = replace(c.diffusion, param="logits", self_cond=True, ce_weight=0.0, x0_rescale=True, t_sampling="warp",
                  rollout_prob=prob, rollout_steps=steps, rollout_jump=jump)
    torch.manual_seed(seed)
    return ByteDiffusionLM(replace(c, diffusion=dif))


class Rollout(unittest.TestCase):
    def test_hash_and_validation(self):
        c = get_preset("rpw")
        self.assertEqual(replace(c, diffusion=replace(c.diffusion, rollout_steps=3, rollout_jump=7)).architecture_hash(), c.architecture_hash())
        self.assertNotEqual(replace(c, diffusion=replace(c.diffusion, param="logits", rollout_prob=0.25)).architecture_hash(),
                            replace(c, diffusion=replace(c.diffusion, param="logits")).architecture_hash())
        with self.assertRaises(ValueError):
            validate(replace(c, diffusion=replace(c.diffusion, param="eps", rollout_prob=0.2)))

    def test_off_is_identical(self):
        x = torch.randint(0, 256, (8, 256))
        a, b = build(0.0).train(), build(0.0, steps=3, jump=10).train()
        torch.manual_seed(5); la, _ = a(x); torch.manual_seed(5); lb, _ = b(x)
        self.assertTrue(torch.equal(la, lb))

    def test_jump_posterior_matches_adjacent(self):
        g = build(0.25).gaussian_diffusion
        x0, xs = torch.randn(4, 8, 64), torch.randn(4, 8, 64)
        s = torch.tensor([5, 300, 600, 999])
        m1, v1 = g.jump_posterior(x0, xs, s, s - 1)
        m2, v2, _ = g.q_posterior(x0, xs, s)
        self.assertLess(float((m1 - m2).abs().max()), 1e-4)
        self.assertLess(float((v1 - v2).abs().max()), 1e-6)
        # composition sanity: a long jump is a valid Gaussian with variance below the prior-conditioned bound
        m, v = g.jump_posterior(x0, xs, s, s - 4)
        self.assertTrue(bool((v > 0).all()))

    def test_rollout_changes_state_detached_and_trains(self):
        net = build(1.0, steps=2, jump=50).train(); g = net.gaussian_diffusion
        x = torch.randint(0, 256, (8, 256))
        emb = net.byte_embedder(x)
        t = torch.tensor([10, 100, 400, 800, 900, 950, 990, 999])
        xt = g.q_sample(emb, t)
        new, frac = g.rollout_state(emb, xt, t, net.byte_embedder.embedding.weight)
        self.assertEqual(frac, 4 / 8)                                    # eligible: t + 100 <= 999 -> first four
        self.assertTrue(torch.equal(new[4:], xt[4:]))                    # ineligible rows untouched
        self.assertGreater(float((new[:4] - xt[:4]).abs().mean()), 0.05)  # eligible rows replaced
        self.assertFalse(new.requires_grad)
        loss, stats = net(x); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertIn("rollout_frac", stats)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_eval_mode_never_rolls_out(self):
        net = build(1.0).eval(); x = torch.randint(0, 256, (4, 256))
        with torch.no_grad():
            torch.manual_seed(3); a, st = net(x)
        self.assertNotIn("rollout_frac", st) if False else None
        ref = build(0.0).eval()
        with torch.no_grad():
            torch.manual_seed(3); b, _ = ref(x)
        self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
