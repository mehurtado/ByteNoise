import itertools, os, sys, unittest
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import get_preset
from main import ByteDiffusionLM, JointByteHead, EMA
from train import warm_start

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "runs", "rpw_encattn_s0")
J = JointByteHead


def build(joint, seed=0, enc=True):
    c = get_preset("rpw")
    dif = replace(c.diffusion, param="logits", self_cond=True, ce_weight=0.0, x0_rescale=True, t_sampling="warp")
    torch.manual_seed(seed)
    return ByteDiffusionLM(replace(c, diffusion=dif, model=replace(c.model, rpw_enc_attn=enc, rpw_joint_head=joint)))


class Joint(unittest.TestCase):
    def test_hash_unchanged_when_off(self):
        c = get_preset("rpw")
        self.assertEqual(replace(c, model=replace(c.model, rpw_joint_head=False)).architecture_hash(), c.architecture_hash())
        self.assertNotEqual(replace(c, model=replace(c.model, rpw_joint_head=True)).architecture_hash(), c.architecture_hash())

    def test_param_count(self):
        a, b = build(False), build(True)
        n = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(n(b) - n(a), 512 * 128 + 128 + 128 * 4 + 4 + 128 * 4096)
        self.assertEqual(b.denoise_unet.group_counts()["joint head"], 590_468)
        print("joint head params", n(b) - n(a), "total", n(b))

    def test_normalisation_small_vocab(self):
        torch.manual_seed(0)
        K, V = 4, 4
        logpi = torch.log_softmax(torch.randn(1, 1, K), -1)
        logp = torch.log_softmax(torch.randn(1, 1, K, 4, V), -1)
        total = 0.0
        for combo in itertools.product(range(V), repeat=4):
            total += float(J.joint_logprob(logpi, logp, torch.tensor(combo).view(1, 1, 4)).exp())
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_dependence_example(self):
        # two deterministic alternatives, "abcd" and "wxyz" (as ids 0-3 / 4-7), equal weight, V=8
        V = 8
        logp = torch.full((1, 1, 2, 4, V), -30.0)
        for j in range(4):
            logp[0, 0, 0, j, j] = 0.0; logp[0, 0, 1, j, 4 + j] = 0.0
        logpi = torch.log(torch.tensor([[[0.5, 0.5]]]))
        jp = lambda t: float(J.joint_logprob(logpi, logp, torch.tensor(t).view(1, 1, 4)).exp())
        marg = (logpi[..., None, None] + logp).logsumexp(2).exp()[0, 0]           # (4, V)
        prod = lambda t: float(torch.prod(torch.stack([marg[j, t[j]] for j in range(4)])))
        self.assertAlmostEqual(jp([0, 1, 2, 3]), 0.5, places=4)                  # coherent string
        self.assertAlmostEqual(prod([0, 1, 2, 3]), 0.0625, places=4)             # product of marginals underrates it 8x
        self.assertLess(jp([0, 1, 6, 7]), 1e-6)                                  # incoherent mix: joint ~0
        self.assertAlmostEqual(prod([0, 1, 6, 7]), 0.0625, places=4)             # ... product of marginals gives the same 1/16
        best, cand, score = J.candidates(logpi, logp)
        self.assertTrue(torch.equal(cand[0, 0, 0], torch.tensor([0, 1, 2, 3])) and torch.equal(cand[0, 0, 1], torch.tensor([4, 5, 6, 7])))
        self.assertIn(best[0, 0].tolist(), ([0, 1, 2, 3], [4, 5, 6, 7]))

    def test_candidate_beats_marginal_argmax(self):
        # weights 0.55/0.45 on "abcd"/"wxyz" with slightly leaky alternatives: marginal argmax can mix strings, the selected candidate cannot
        V = 8
        logp = torch.log_softmax(torch.full((1, 1, 2, 4, V), -6.0).scatter_(-1, torch.tensor([[[[[0], [1], [2], [3]], [[4], [5], [6], [7]]]]]), 0.0), -1)
        logpi = torch.log(torch.tensor([[[0.5, 0.5]]]))
        marg = (logpi[..., None, None] + logp).logsumexp(2)[0, 0]
        mix_arg = marg.argmax(-1).tolist()                                        # ties between the two strings: picks one id per position independently
        best, _, _ = J.candidates(logpi, logp)
        jl = lambda t: float(J.joint_logprob(logpi, logp, torch.tensor(t).view(1, 1, 4)))
        self.assertGreaterEqual(jl(best[0, 0].tolist()), jl(mix_arg) - 1e-6)

    def test_forward_loss_grads_and_collapse_stats(self):
        net = build(True).train(); x = torch.randint(0, 256, (4, 256))
        torch.manual_seed(1); loss, stats = net(x); loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for k in ("joint_pi_entropy", "joint_alt_agree", "marginal_ce"):
            self.assertIn(k, stats)
        self.assertAlmostEqual(float(stats["joint_pi_entropy"]), 1.3863, places=3)   # uniform pi at init
        h = net.denoise_unet.joint
        for name, m in (("hid", h.hid), ("pi", h.pi), ("delta", h.delta)):
            for p in m.parameters():
                self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(float(h.delta.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(net.denoise_unet.out.weight.grad.abs().sum()), 0.0)  # base byte head still trains
        # joint NLL/4 <= marginal CE is not guaranteed in general, but both must be finite and close at init
        self.assertLess(abs(float(stats["ce_loss"]) - float(stats["marginal_ce"])), 0.5)

    def test_marginals_normalised_and_marginal_mode(self):
        net = build(True).eval(); h = net.denoise_unet.joint
        x = torch.randn(2, 256, 128 * 0 + 128)
        with torch.no_grad():
            lg = net.denoise_unet(torch.randn(2, 256, 128), torch.tensor([10, 700]))
        self.assertLess(float((lg.exp().sum(-1) - 1).abs().max()), 1e-4)
        self.assertEqual(tuple(h.joint_decode().shape), (2, 256))
        xb = torch.randint(0, 256, (2, 256))
        a = h.joint_nll(xb)
        self.assertEqual(tuple(a.shape), (2, 64))

    @unittest.skipUnless(os.path.exists(os.path.join(SRC, "checkpoint.pt")), "needs runs/rpw_encattn_s0")
    def test_transfer_from_source_checkpoint(self):
        ck = torch.load(os.path.join(SRC, "checkpoint.pt"), map_location="cpu", weights_only=False)
        new = build(True); ema = EMA(new, decay=0.999, warmup=True)
        warm_start(os.path.join(SRC, "checkpoint.pt"), new, ema)
        old = build(False); old.load_state_dict(ck["model_state_dict"])
        for k, v in old.state_dict().items():
            self.assertTrue(torch.equal(new.state_dict()[k], v), k)
        x, t = torch.randn(4, 256, 128), torch.tensor([10, 300, 600, 900])
        new.eval(); old.eval()
        with torch.no_grad():
            po = torch.softmax(old.denoise_unet(x, t), -1); pn = new.denoise_unet(x, t).exp()
        kl = float((po * (po.clamp_min(1e-12).log() - pn.clamp_min(1e-12).log())).sum(-1).mean())
        agree = float((po.argmax(-1) == pn.argmax(-1)).float().mean())
        print("transfer: KL(old||new marginal) %.5f nats, argmax agreement %.4f" % (kl, agree))
        self.assertLess(kl, 0.05); self.assertGreater(agree, 0.95)


if __name__ == "__main__":
    unittest.main()
