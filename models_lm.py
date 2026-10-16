import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class GatedFFN(nn.Module):
    def __init__(self, d_model, mult, dropout):
        super().__init__()
        hidden = int(round(d_model * mult / 8)) * 8 or 8
        self.w_in = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.w_in(x).chunk(2, dim=-1)
        return self.drop(self.w_out(F.silu(a) * b))


def rotary_cache(seq_len, head_dim, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    ang = torch.outer(torch.arange(seq_len, device=device).float(), inv)
    return ang.cos(), ang.sin()


def apply_rotary(x, cos, sin):  # x: (b, h, n, d)
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = cos[None, None, : x.size(2)], sin[None, None, : x.size(2)]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, rope):
        b, n, c = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.h, c // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rotary(q, *rope), apply_rotary(k, *rope)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(b, n, c))


# --- Gated linear recurrence (minGRU form): h_t = (1 - z_t) h_{t-1} + z_t * g(c_t), h_0 = 0 ---

def positive_activation(x):
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_positive_activation(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


def scan_sequential(z, cand):
    """Reference: z in (0,1), cand > 0; both (b, n, d). Loops over time."""
    h = torch.zeros_like(z[:, 0])
    out = []
    for t in range(z.size(1)):
        h = (1 - z[:, t]) * h + z[:, t] * cand[:, t]
        out.append(h)
    return torch.stack(out, dim=1)


def scan_parallel(z_logit, log_cand):
    """Same recurrence in log space via logcumsumexp (Heinsen scan), fp32. z = sigmoid(z_logit). Returns h."""
    z_logit, log_cand = z_logit.float(), log_cand.float()
    log_z = -F.softplus(-z_logit)
    log_1mz = -F.softplus(z_logit)
    a_star = torch.cumsum(log_1mz, dim=1)
    return (a_star + torch.logcumsumexp(log_z + log_cand - a_star, dim=1)).exp()


class RecurrentMixer(nn.Module):
    """Causal short conv -> gated linear recurrence -> output gate. Uses earlier positions only."""

    def __init__(self, d_model, expansion, conv_kernel):
        super().__init__()
        d_in = d_model * expansion
        self.k = conv_kernel
        self.in_proj = nn.Linear(d_model, 2 * d_in, bias=False)  # value path + output gate
        self.conv = nn.Conv1d(d_in, d_in, conv_kernel, groups=d_in)
        self.gates = nn.Linear(d_in, 2 * d_in)  # candidate + forget/update gate logits
        self.out_proj = nn.Linear(d_in, d_model, bias=False)

    def forward(self, x, parallel=True):
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv(F.pad(u.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2))
        cand_logit, z_logit = self.gates(u).chunk(2, dim=-1)
        if parallel:
            h = scan_parallel(z_logit, log_positive_activation(cand_logit.float()))
        else:
            h = scan_sequential(torch.sigmoid(z_logit.float()), positive_activation(cand_logit.float()))
        return self.out_proj(h.to(x.dtype) * F.silu(gate))


class Block(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.norm1 = RMSNorm(lm.d_model)
        self.norm2 = RMSNorm(lm.d_model)
        if lm.arch == "transformer":
            self.mixer = CausalSelfAttention(lm.d_model, lm.n_heads, lm.dropout)
        else:
            self.mixer = RecurrentMixer(lm.d_model, lm.expansion, lm.conv_kernel)
        self.ffn = GatedFFN(lm.d_model, lm.ffn_mult, lm.dropout)
        self.arch = lm.arch

    def forward(self, x, rope=None, parallel=True):
        h = self.norm1(x)
        x = x + (self.mixer(h, rope) if self.arch == "transformer" else self.mixer(h, parallel))
        return x + self.ffn(self.norm2(x))


class ByteLM(nn.Module):
    """Decoder-only byte LM. Transformer or recurrent mixer; the block stack can be looped with shared weights."""

    def __init__(self, cfg: Config):
        super().__init__()
        lm = cfg.lm
        self.cfg = cfg
        self.lm = lm
        self.embed = nn.Embedding(lm.vocab_size, lm.d_model)
        self.blocks = nn.ModuleList([Block(lm) for _ in range(lm.n_layers)])
        self.norm = RMSNorm(lm.d_model)
        self.iter_emb = nn.Parameter(torch.zeros(lm.loops, lm.d_model)) if lm.iter_embedding and lm.loops > 1 else None
        self.apply(self._init)
        for b in self.blocks:  # scaled residual init
            nn.init.normal_(b.ffn.w_out.weight, std=0.02 / math.sqrt(2 * lm.n_layers * lm.loops))
        self._rope = None
        self.parallel = True

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def _rope_for(self, n, device):
        if self.lm.arch != "transformer":
            return None
        if self._rope is None or self._rope[0].size(0) < n or self._rope[0].device != device:
            self._rope = rotary_cache(max(n, self.lm.seq_len), self.lm.d_model // self.lm.n_heads, device)
        return self._rope

    def logits(self, idx, return_norms=False):
        x = self.embed(idx)
        rope = self._rope_for(idx.size(1), idx.device)
        norms = []
        for loop in range(self.lm.loops):
            if self.iter_emb is not None:
                x = x + self.iter_emb[loop]
            for blk in self.blocks:
                x = blk(x, rope, self.parallel) if self.lm.arch == "transformer" else blk(x, None, self.parallel)
            if return_norms:
                norms.append(x.detach().float().norm(dim=-1).mean().item())
        out = F.linear(self.norm(x), self.embed.weight)
        return (out, norms) if return_norms else out

    def forward(self, batch, **_):
        """batch: (b, n+1) bytes. Returns (loss, stats); loss is mean next-byte NLL in nats."""
        inputs, targets = batch[:, :-1], batch[:, 1:]
        logits = self.logits(inputs)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss, {"nll": loss.detach(), "noise_loss": loss.detach()}

    @torch.no_grad()
    def generate(self, prompt, max_new, temperature=1.0, top_k=0, generator=None):
        self.eval()
        idx = prompt
        for _ in range(max_new):
            ctx = idx[:, -self.lm.seq_len:]
            logits = self.logits(ctx)[:, -1].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = logits.topk(top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
