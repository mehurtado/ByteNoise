import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import os
import glob
from functools import partial
import copy
from einops import rearrange, reduce # Using einops for clearer tensor manipulations

from config import Config, ModelConfig, DiffusionConfig, validate
from models_lm import RMSNorm, GatedFFN, scan_parallel, log_positive_activation, rotary_cache, apply_rotary


# --- Helper Functions & Modules ---

class EMA(nn.Module):
    def __init__(self, model, decay=0.999, warmup=True):
        super().__init__()
        self.decay = decay
        self.warmup = warmup
        self.model = copy.deepcopy(model).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.register_buffer('num_updates', torch.zeros((), dtype=torch.long))

    def current_decay(self):
        if not self.warmup:
            return self.decay
        n = int(self.num_updates.item())
        return min(self.decay, (1 + n) / (10 + n))

    @torch.no_grad()
    def update(self, model):
        decay = self.current_decay()
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
        self.num_updates += 1


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """Extracts values from a for given timesteps t, reshaping to x_shape."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps, beta_start, beta_end):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672 by Nichol and Dhariwal.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Positional Embedding for Timesteps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time): # time: (batch_size,)
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1: # zero pad if dim is odd
            embeddings = F.pad(embeddings, (0,1))
        return embeddings

class ByteEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_norm=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_norm = embed_norm

    def table(self):
        """(vocab, dim) embedding table; with embed_norm > 0 every row is rescaled to exactly that L2 norm."""
        w = self.embedding.weight
        return F.normalize(w, dim=-1) * self.embed_norm if self.embed_norm > 0 else w

    def forward(self, x):
        # x: (batch_size, seq_len)
        return F.embedding(x, self.table()) # Output: (batch_size, seq_len, embed_dim)

class TiedHead(nn.Module):
    """Output head tied to the embedding table: logits = exp(log_scale) * z @ E^T, z = (b, n, byte_embed_dim) features.

    E is the table as constrained by embed_norm (rows of norm embed_norm) and is NOT detached here: the CE gradient also trains
    E (standard weight tying; with embed_norm it can only move rows along the sphere, and the input-embedding path already trains
    E through CE). The learnable scalar log_scale starts at -log(row norm) so initial logits are ~ the std of z (O(1)). x0_hat =
    softmax(logits) @ E still detaches E (GaussianDiffusion.x0_from_logits). The embedder is kept in a plain list so it is not
    registered a second time (state_dict has it once, under byte_embedder) while copy.deepcopy (EMA) still points the copy at the
    copied embedder."""
    def __init__(self, embedder, init_norm):
        super().__init__()
        self._emb = [embedder]
        self.log_scale = nn.Parameter(torch.tensor(-math.log(init_norm)))

    def forward(self, z):
        return self.log_scale.exp() * (z.float() @ self._emb[0].table().float().t())


def word_segment_ids(x_bytes):
    """(b, n) int segment id per position for word-structured noise.

    Letters are ASCII A-Za-z and bytes >= 128 (so UTF-8 multibyte characters stay inside a word). A segment is a maximal letter run
    together with the non-letter bytes (whitespace, punctuation, newlines, digits) that PRECEDE it, so separators take the noise
    level of the following word; non-letter bytes after the last word of the window form a final segment of their own."""
    letter = (((x_bytes | 32) >= 97) & ((x_bytes | 32) <= 122)) | (x_bytes >= 128)
    end = letter.clone()
    end[:, :-1] &= ~letter[:, 1:]  # last letter of each run
    end = end.long()
    return end.cumsum(1) - end


# --- U-Net Components ---

class ResnetBlock1D(nn.Module):
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None, groups, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size=3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        self.dropout = nn.Dropout(dropout)
        self.block2 = nn.Sequential(
            nn.Conv1d(dim_out, dim_out, kernel_size=3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.SiLU()
        )
        self.res_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None): # x: (batch, channels, seq_len)
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1') # Reshape for broadcasting
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            h = h * (scale + 1) + shift

        h = self.dropout(h)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention1D(nn.Module):
    def __init__(self, dim, heads, dim_head, groups):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = nn.GroupNorm(groups, dim)
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x): # x: (batch, channels, seq_len)
        b, c, n = x.shape
        x_norm = self.norm(x)
        qkv = self.to_qkv(x_norm).chunk(3, dim=1) # (b, h*d, n), (b, h*d, n), (b, h*d, n)

        # Rearrange for multi-head attention: b (h d) n -> b h n d
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h=self.heads).contiguous(), qkv)

        # Use efficient scaled dot product attention
        out = F.scaled_dot_product_attention(q, k, v)

        # Rearrange back: b h n d -> b (h d) n
        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out) + x # Add residual connection

class Downsample1D(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        self.conv = nn.Conv1d(dim_in, default(dim_out, dim_in), kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(dim_in, default(dim_out, dim_in), kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(
        self,
        model_dim,
        in_embed_dim,
        out_dim=None,
        dim_mults=(1, 2, 3, 4),
        time_emb_dim_mult=4,
        resnet_groups=8,
        attn_heads=4,
        attn_dim_head=None,
        dropout=0.1
    ):
        super().__init__()
        attn_dim_head = default(attn_dim_head, model_dim // attn_heads)
        self.in_embed_dim = in_embed_dim
        self.out_dim = default(out_dim, in_embed_dim) # Predict noise of same dim as input embeddings

        # --- Time Embedding ---
        time_dim = model_dim * time_emb_dim_mult
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_dim), # Initial time embedding uses model_dim
            nn.Linear(model_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # --- Initial Convolution ---
        # Input x will be (batch, seq_len, in_embed_dim), permute to (batch, in_embed_dim, seq_len)
        self.init_conv = nn.Conv1d(in_embed_dim, model_dim, kernel_size=7, padding=3)

        dims = [model_dim, *map(lambda m: model_dim * m, dim_mults)]
        in_out_dims = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out_dims)

        block_klass = partial(ResnetBlock1D, groups=resnet_groups, time_emb_dim=time_dim, dropout=dropout)
        attn_klass = partial(Attention1D, heads=attn_heads, dim_head=attn_dim_head, groups=resnet_groups)

        # --- Downsampling Path ---
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out_dims):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in), # Two ResNet blocks per level
                block_klass(dim_in, dim_in),
                attn_klass(dim_in),
                Downsample1D(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            ]))

        # --- Bottleneck ---
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = attn_klass(mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        # --- Upsampling Path ---
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out_dims)):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out), # Skip connection doubles channels
                block_klass(dim_out + dim_in, dim_out), # (dim_out from prev upsample + dim_in from skip)
                attn_klass(dim_out),
                Upsample1D(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)
            ]))

        # --- Final Convolution ---
        self.final_res_block = block_klass(model_dim * 2, model_dim) # model_dim from init_conv skip + model_dim from last upsample
        self.final_conv = nn.Conv1d(model_dim, self.out_dim, 1)


    def forward(self, x_embed, time): # x_embed: (batch, seq_len, in_embed_dim), time: (batch,)
        # Permute input for Conv1d: (batch, in_embed_dim, seq_len)
        x = rearrange(x_embed, 'b n c -> b c n')

        t_emb = self.time_mlp(time)

        h = self.init_conv(x)
        # Bolt optimization: removed unnecessary .clone() since PyTorch out-of-place ops return new tensors
        skip_connections = [h] # Store initial conv output for final skip

        # Downsampling
        for resnet_block1, resnet_block2, attention, downsample in self.downs:
            h = resnet_block1(h, t_emb)
            skip_connections.append(h)
            h = resnet_block2(h, t_emb)
            h = attention(h)
            skip_connections.append(h)
            h = downsample(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Upsampling
        for resnet_block1, resnet_block2, attention, upsample in self.ups:
            h = torch.cat((h, skip_connections.pop()), dim=1) # Add skip from attention
            h = resnet_block1(h, t_emb)
            h = torch.cat((h, skip_connections.pop()), dim=1) # Add skip from resnet_block1
            h = resnet_block2(h, t_emb)
            h = attention(h)
            h = upsample(h)

        h = torch.cat((h, skip_connections.pop()), dim=1) # Final skip from init_conv
        h = self.final_res_block(h, t_emb)
        out = self.final_conv(h) # (batch, out_dim, seq_len)

        # Permute back to (batch, seq_len, out_dim)
        return rearrange(out, 'b c n -> b n c')


class BiMixer(nn.Module):
    """Bidirectional gated linear recurrence: centered depthwise conv, forward scan + scan over the flipped sequence
    (separate gates per direction), both gated by one output gate, concatenated and projected. Non-causal."""
    def __init__(self, d, conv_kernel):
        super().__init__()
        self.in_proj = nn.Linear(d, 2 * d, bias=False)  # value path + output gate
        self.conv = nn.Conv1d(d, d, conv_kernel, groups=d, padding=conv_kernel // 2)  # centered: sees past and future
        self.gates_f = nn.Linear(d, 2 * d)
        self.gates_b = nn.Linear(d, 2 * d)
        self.out_proj = nn.Linear(2 * d, d, bias=False)

    @staticmethod
    def _scan(gates, u):
        cand_logit, z_logit = gates(u).chunk(2, dim=-1)
        return scan_parallel(z_logit, log_positive_activation(cand_logit.float()))

    def forward(self, x):
        u, gate = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2)).transpose(1, 2))
        hf = self._scan(self.gates_f, u)
        hb = self._scan(self.gates_b, u.flip(1)).flip(1)
        g = F.silu(gate)
        return self.out_proj(torch.cat((hf.to(x.dtype), hb.to(x.dtype)), dim=-1) * g.repeat(1, 1, 2))


class BiRNNBlock(nn.Module):
    def __init__(self, d, conv_kernel, ffn_mult, dropout):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.mixer = BiMixer(d, conv_kernel)
        self.ffn = GatedFFN(d, ffn_mult, dropout)
        self.film = nn.Linear(d, 4 * d)  # per-block timestep conditioning: scale/shift before mixer and before FFN

    def forward(self, x, t_emb):
        s1, b1, s2, b2 = self.film(t_emb)[:, None].chunk(4, dim=-1)
        x = x + self.mixer(self.norm1(x) * (1 + s1) + b1)
        return x + self.ffn(self.norm2(x) * (1 + s2) + b2)


class BiRNNDenoiser(nn.Module):
    """Bidirectional recurrent noise predictor; same interface as UNet1D: (b, n, embed), (b,) -> (b, n, embed)."""
    def __init__(self, model_dim, in_embed_dim, n_layers, conv_kernel, ffn_mult, dropout, out_dim=None):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(model_dim), nn.Linear(model_dim, model_dim), nn.GELU(),
                                      nn.Linear(model_dim, model_dim))
        self.inp = nn.Linear(in_embed_dim, model_dim)
        self.blocks = nn.ModuleList([BiRNNBlock(model_dim, conv_kernel, ffn_mult, dropout) for _ in range(n_layers)])
        self.final_norm = RMSNorm(model_dim)
        self.out = nn.Linear(model_dim, default(out_dim, in_embed_dim))

    def forward(self, x_embed, time):
        t_emb = self.time_mlp(time)
        h = self.inp(x_embed) + t_emb[:, None]
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out(self.final_norm(h))


class _DWConvBranch(nn.Module):
    """Multi-scale local branch: depthwise k=3 convs at dilations 1/2/4/8 (receptive field ~31), concat, pointwise mix."""
    DILATIONS = (1, 2, 4, 8)

    def __init__(self, d):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(d, d, 3, groups=d, dilation=r, padding=r) for r in self.DILATIONS])
        self.mix = nn.Linear(d * len(self.DILATIONS), d)

    def forward(self, x, rope=None):
        xt = x.transpose(1, 2)
        return self.mix(F.silu(torch.cat([c(xt) for c in self.convs], dim=1)).transpose(1, 2))


class _NGramBranch(nn.Module):
    """Local n-gram evidence: depthwise conv (kernel k, centered) + pointwise."""
    def __init__(self, d, kernel):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel, groups=d, padding=kernel // 2)
        self.mix = nn.Linear(d, d)

    def forward(self, x, rope=None):
        return self.mix(F.silu(self.conv(x.transpose(1, 2)).transpose(1, 2)))


class _RNNBranch(nn.Module):
    """Bidirectional recurrent branch: down-project to `width`, BiMixer (fwd + flipped scans, fp32 inside), project back."""
    def __init__(self, d, width, conv_kernel):
        super().__init__()
        self.down = nn.Linear(d, width, bias=False)
        self.mixer = BiMixer(width, conv_kernel)
        self.up = nn.Linear(width, d, bias=False)

    def forward(self, x, rope=None):
        return self.up(self.mixer(self.down(x)))


class _GlobalAttnBranch(nn.Module):
    """Full-resolution non-causal multi-head self-attention with rotary positions."""
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x, rope):
        b, n, c = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.h, c // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rotary(q, *rope), apply_rotary(k, *rope)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.out(y.transpose(1, 2).reshape(b, n, c))


class HybridBlock(nn.Module):
    """RMSNorm -> FiLM -> parallel branches on the same input -> concat -> linear mix -> GatedFFN (its w_out is the
    output projection) -> residual."""
    def __init__(self, m: ModelConfig):
        super().__init__()
        d = m.model_dim
        self.names = [b for b in m.hybrid_branches.split(",") if b]
        makers = {"conv": lambda: _DWConvBranch(d), "rnn": lambda: _RNNBranch(d, m.hybrid_rnn_width, m.rnn_conv),
                  "attn": lambda: _GlobalAttnBranch(d, m.hybrid_heads), "ngram": lambda: _NGramBranch(d, m.hybrid_ngram_kernel)}
        self.branches = nn.ModuleDict({n: makers[n]() for n in self.names})
        self.norm = RMSNorm(d)
        self.film = nn.Linear(d, 2 * d)
        self.mix = nn.Linear(len(self.names) * d, d)
        self.ffn = GatedFFN(d, m.hybrid_ffn_mult, m.dropout)

    def forward(self, h, t_emb, rope):
        s, b = self.film(t_emb)[:, None].chunk(2, dim=-1)
        z = self.norm(h) * (1 + s) + b
        y = torch.cat([self.branches[n](z, rope) for n in self.names], dim=-1)
        return h + self.ffn(self.mix(y))


class _ParCoarse(nn.Module):
    """Parallel coarse branch for the flat hybrid: pack four adjacent positions -> width 384 -> one trimmed hybrid block -> project and unpack,
    added back through a zero-initialised per-channel gate (so the model starts as the unchanged flat one)."""
    def __init__(self, m: ModelConfig):
        super().__init__()
        d, w = m.model_dim, 384
        self.pnorm, self.pin = RMSNorm(4 * d), nn.Linear(4 * d, w)
        self.block = HierBlock(w, w // 2, 8, m.rnn_conv, 3.0, m.dropout)
        self.pout, self.gate = nn.Linear(w, 4 * d), nn.Parameter(torch.zeros(d))
        self.hd = w // 8

    def forward(self, h, t_emb):
        b, n, d = h.shape
        if n % 4:
            raise ValueError("parallel coarse branch needs a sequence length divisible by four")
        z = self.pin(self.pnorm(h.reshape(b, n // 4, 4 * d)))
        z = self.block(z, t_emb, rotary_cache(n // 4, self.hd, h.device))
        return h + self.gate * self.pout(z).reshape(b, n, d)


class HybridDenoiser(nn.Module):
    """Parallel multi-branch (conv / bidirectional recurrent / global attention / n-gram) noise predictor.
    Same interface as UNet1D: (b, n, in_embed), (b,) -> (b, n, out_dim)."""
    def __init__(self, m: ModelConfig, in_embed_dim, out_dim):
        super().__init__()
        d = m.model_dim
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.inp = nn.Linear(in_embed_dim, d)
        self.blocks = nn.ModuleList([HybridBlock(m) for _ in range(m.hybrid_layers)])
        for blk in self.blocks:  # scaled residual init
            nn.init.normal_(blk.ffn.w_out.weight, std=0.02 / math.sqrt(2 * m.hybrid_layers))
        self.par_after = m.hybrid_par_after
        self.par = _ParCoarse(m) if m.hybrid_par_coarse else None
        self.final_norm = RMSNorm(d)
        self.out = nn.Linear(d, out_dim)
        self.head_dim = d // m.hybrid_heads
        self._rope = None

    def forward(self, x_embed, time):
        n = x_embed.size(1)
        if self._rope is None or self._rope[0].size(0) < n or self._rope[0].device != x_embed.device:
            self._rope = rotary_cache(n, self.head_dim, x_embed.device)
        t_emb = self.time_mlp(time)
        h = self.inp(x_embed) + t_emb[:, None]
        for i, blk in enumerate(self.blocks):
            h = blk(h, t_emb, self._rope)
            if self.par is not None and i + 1 == self.par_after:
                h = self.par(h, t_emb)
        return self.out(self.final_norm(h))


class HierBlock(nn.Module):
    """Trimmed hybrid block (conv / bidirectional recurrent / rotary attention, no n-gram branch): RMSNorm+FiLM -> branches ->
    concat -> fuse -> residual; then RMSNorm+FiLM -> GatedFFN -> residual. FiLM starts at zero; fuse and FFN-out use a small init."""
    def __init__(self, d, rnn_width, heads, rnn_conv, ffn_mult, dropout):
        super().__init__()
        self.norm_mix, self.norm_ffn = RMSNorm(d), RMSNorm(d)
        self.film = nn.Linear(256, 4 * d)
        self.branches = nn.ModuleList([_DWConvBranch(d), _RNNBranch(d, rnn_width, rnn_conv), _GlobalAttnBranch(d, heads)])
        self.fuse = nn.Linear(3 * d, d)
        self.ffn = GatedFFN(d, ffn_mult, dropout)
        for w in (self.fuse.weight, self.ffn.w_out.weight):
            nn.init.normal_(w, std=0.02 / math.sqrt(12))
        nn.init.zeros_(self.fuse.bias); nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def forward(self, x, t_emb, rope):
        a, b, c, e = self.film(t_emb)[:, None].chunk(4, dim=-1)
        z = self.norm_mix(x) * (1 + a) + b
        x = x + self.fuse(torch.cat([br(z, rope) for br in self.branches], dim=-1))
        return x + self.ffn(self.norm_ffn(x) * (1 + c) + e)


class _Pack(nn.Module):  # adjacent pairs (b, n, d) -> (b, n/2, d_out), order preserved
    def __init__(self, d_in, d_out):
        super().__init__()
        self.norm, self.proj = RMSNorm(2 * d_in), nn.Linear(2 * d_in, d_out)

    def forward(self, x):
        b, n, d = x.shape
        return self.proj(self.norm(x.reshape(b, n // 2, 2 * d)))


class _Unpack(nn.Module):  # (b, n, d_in) -> (b, 2n, d_out), fused into the skip with a learned per-channel gate (init sigmoid 0.5)
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj, self.gate = nn.Linear(d_in, 2 * d_out), nn.Parameter(torch.zeros(d_out))

    def forward(self, x, skip):
        b, n, _ = x.shape
        return skip + self.gate.sigmoid() * self.proj(x).reshape(b, 2 * n, -1)


class HierDenoiser(nn.Module):
    """Three-resolution trimmed hybrid: fine (n, d) -> medium (n/2, 4d/3) -> coarse (n/4, 2d, hier_core_blocks blocks, default two) and back with gated skips.
    Six blocks by default. Same interface as UNet1D: (b, n, in_embed), (b,) -> (b, n, out_dim); n must be divisible by 4."""
    def __init__(self, m: ModelConfig, in_embed_dim, out_dim):
        super().__init__()
        d = m.model_dim; d1, d2 = d * 4 // 3, d * 2  # 192 / 256 / 384
        self.hd = (d // 6, d1 // 8, d2 // 8)  # head dims 32 / 32 / 48
        blk = lambda w, h: HierBlock(w, w // 2, h, m.rnn_conv, 3.0, m.dropout)
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(256), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256))
        self.inp = nn.Linear(in_embed_dim, d)
        self.fine_enc, self.mid_enc = blk(d, 6), blk(d1, 8)
        self.core = nn.ModuleList([blk(d2, 8) for _ in range(m.hier_core_blocks)])
        self.mid_dec, self.fine_dec = blk(d1, 8), blk(d, 6)
        self.fine_enc_x = nn.ModuleList([blk(d, 6) for _ in range(m.hier_fine_blocks)])  # extra full-resolution blocks after fine_enc / after fine_dec
        self.fine_dec_x = nn.ModuleList([blk(d, 6) for _ in range(m.hier_fine_blocks)])
        self.down1, self.down2, self.up1, self.up2 = _Pack(d, d1), _Pack(d1, d2), _Unpack(d2, d1), _Unpack(d1, d)
        self.final_norm = RMSNorm(d)
        self.out = nn.Linear(d, out_dim)

    def forward(self, x_embed, time):
        n = x_embed.size(1)
        if n % 4:
            raise ValueError("hier denoiser needs a sequence length divisible by four")
        dev = x_embed.device
        r0, r1, r2 = (rotary_cache(n // s, h, dev) for s, h in zip((1, 2, 4), self.hd))
        t = self.time_mlp(time)
        fine = self.fine_enc(self.inp(x_embed), t, r0)
        for blk in self.fine_enc_x:
            fine = blk(fine, t, r0)
        mid = self.mid_enc(self.down1(fine), t, r1)
        z = self.down2(mid)
        for blk in self.core:
            z = blk(z, t, r2)
        z = self.mid_dec(self.up1(z, mid), t, r1)
        z = self.fine_dec(self.up2(z, fine), t, r0)
        for blk in self.fine_dec_x:
            z = blk(z, t, r0)
        return self.out(self.final_norm(z))


def _rope_fp32(pos, head_dim):
    """cos/sin tables for arbitrary float coordinates `pos` (n,), always computed in fp32."""
    inv = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=pos.device).float() / head_dim))
    with torch.autocast(pos.device.type, enabled=False):
        ang = torch.outer(pos.float(), inv)
    return ang.cos(), ang.sin()


class _RPWAttn(nn.Module):
    """Bias-free multi-head attention, head dim 32, rotary on queries and keys at the given float coordinates."""
    def __init__(self, dq, dkv, heads, out_std):
        super().__init__()
        self.h, inner = heads, heads * 32
        self.q, self.k, self.v = nn.Linear(dq, inner, bias=False), nn.Linear(dkv, inner, bias=False), nn.Linear(dkv, inner, bias=False)
        self.o = nn.Linear(inner, dq, bias=False)
        nn.init.normal_(self.o.weight, std=out_std)

    def forward(self, xq, xkv, rq, rk):
        b, nq, _ = xq.shape; nk = xkv.size(1)
        q = self.q(xq).view(b, nq, self.h, 32).transpose(1, 2)
        k = self.k(xkv).view(b, nk, self.h, 32).transpose(1, 2)
        v = self.v(xkv).view(b, nk, self.h, 32).transpose(1, 2)
        y = F.scaled_dot_product_attention(apply_rotary(q, *rq), apply_rotary(k, *rk), v)
        return self.o(y.transpose(1, 2).reshape(b, nq, -1))


def _film(x, norm, scale, shift):
    return norm(x) * (1 + scale) + shift


class _RPWLocalBlock(nn.Module):
    """RMSNorm+FiLM -> four parallel depthwise convs (k=3, dilations 1,2,4,8, SiLU) -> concat -> project -> residual; RMSNorm+FiLM -> gated FFN -> residual."""
    def __init__(self, d, out_std):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.convs = nn.ModuleList([nn.Conv1d(d, d, 3, padding=r, dilation=r, groups=d, bias=False) for r in (1, 2, 4, 8)])
        self.mix = nn.Linear(4 * d, d, bias=False)
        self.ffn = GatedFFN(d, 2.0, 0.0)
        for w in (self.mix.weight, self.ffn.w_out.weight):
            nn.init.normal_(w, std=out_std)

    def forward(self, x, c):
        s1, b1, s2, b2 = c
        z = _film(x, self.n1, s1, b1).transpose(1, 2)
        x = x + self.mix(torch.cat([F.silu(cv(z)) for cv in self.convs], dim=1).transpose(1, 2))
        return x + self.ffn(_film(x, self.n2, s2, b2))


class _RPWWorkBlock(nn.Module):
    """Workspace block: RMSNorm+FiLM -> rotary self-attention (width/32 heads) -> residual; RMSNorm+FiLM -> gated FFN (inner 2x width) -> residual."""
    def __init__(self, d, out_std):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = _RPWAttn(d, d, d // 32, out_std)
        self.ffn = GatedFFN(d, 2.0, 0.0)
        nn.init.normal_(self.ffn.w_out.weight, std=out_std)

    def forward(self, z, c, rope):
        s1, b1, s2, b2 = c
        h = _film(z, self.n1, s1, b1)
        z = z + self.attn(h, h, rope, rope)
        return z + self.ffn(_film(z, self.n2, s2, b2))


class RPWDenoiser(nn.Module):
    """Read-process-write byte denoiser: 2 local blocks at byte resolution, read cross-attention into packed (4-byte) summaries, 4 workspace
    blocks over the summaries, write cross-attention back into the bytes, 2 local blocks, RMSNorm, byte logits. Same interface as UNet1D.
    The byte embedding table lives in the outer ByteDiffusionLM, not here."""
    def __init__(self, m: ModelConfig, in_embed_dim, out_dim, local_blocks=2, work_blocks=4):
        super().__init__()
        d, w = m.model_dim, m.rpw_workspace_dim
        std = 0.02 / math.sqrt(2 * (2 * local_blocks + work_blocks))
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(256), nn.Linear(256, 256, bias=False), nn.SiLU(), nn.Linear(256, 256, bias=False))
        self.byte_cond, self.work_cond = nn.Linear(256, 4 * d), nn.Linear(256, 4 * w)  # two (scale, shift) pairs, shared by every block of the group
        for lin in (self.byte_cond, self.work_cond):
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self.inp = nn.Linear(in_embed_dim, d, bias=False)
        self.enc = nn.ModuleList([_RPWLocalBlock(d, std) for _ in range(local_blocks)])
        self.pack_norm, self.pack = RMSNorm(4 * d), nn.Linear(4 * d, w, bias=False)
        self.read_nq, self.read_nk, self.read = RMSNorm(w), RMSNorm(d), _RPWAttn(w, d, w // 32, std)
        self.work = nn.ModuleList([_RPWWorkBlock(w, std) for _ in range(work_blocks)])
        self.write_nq, self.write_nk, self.write = RMSNorm(d), RMSNorm(w), _RPWAttn(d, w, d // 32, std)
        self.dec = nn.ModuleList([_RPWLocalBlock(d, std) for _ in range(local_blocks)])
        self.final_norm, self.out = RMSNorm(d), nn.Linear(d, out_dim, bias=False)
        self.skip_write = False  # test hook only: drop the write-back

    def group_counts(self):
        n = lambda *mods: sum(p.numel() for mod in mods for p in mod.parameters())
        return {"input projection": n(self.inp), "time MLP": n(self.time_mlp), "byte conditioner": n(self.byte_cond), "workspace conditioner": n(self.work_cond),
                "local blocks": n(self.enc, self.dec), "pack norm+projection": n(self.pack_norm, self.pack), "read attention+norms": n(self.read_nq, self.read_nk, self.read),
                "workspace blocks": n(self.work), "write attention+norms": n(self.write_nq, self.write_nk, self.write), "final norm+output": n(self.final_norm, self.out)}

    def forward(self, x_embed, time):
        b, n, _ = x_embed.shape
        if n % 4:
            raise ValueError("rpw denoiser needs a sequence length divisible by four")
        dev = x_embed.device
        t = self.time_mlp(time)
        cb = [c[:, None] for c in self.byte_cond(t).chunk(4, dim=-1)]
        cw = [c[:, None] for c in self.work_cond(t).chunk(4, dim=-1)]
        rb = _rope_fp32(torch.arange(n, device=dev), 32)                    # byte i at coordinate i
        rs = _rope_fp32(4 * torch.arange(n // 4, device=dev) + 1.5, 32)     # summary j covers bytes 4j..4j+3: centre 4j+1.5
        x = self.inp(x_embed)
        for blk in self.enc:
            x = blk(x, cb)
        z = self.pack(self.pack_norm(x.reshape(b, n // 4, -1)))
        z = z + self.read(self.read_nq(z), self.read_nk(x), rs, rb)
        for blk in self.work:
            z = blk(z, cw, rs)
        if not self.skip_write:
            x = x + self.write(self.write_nq(x), self.write_nk(z), rb, rs)
        for blk in self.dec:
            x = blk(x, cb)
        return self.out(self.final_norm(x))


# --- Gaussian Diffusion Process ---
T_BINS = 50          # time-warp bins
WARP_EMA = 0.05      # per-observation EMA rate of the per-bin loss
WARP_MIX = 0.8       # p(bin) = WARP_MIX * normalised EMA loss + (1 - WARP_MIX) * uniform

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_model, # The U-Net model
        *,
        seq_len,
        num_timesteps,
        beta_schedule_fn,
        beta_start=0.0001,
        beta_end=0.02,
        loss_type='l2',
        clip_denoised=True,
        ce_weight=0.1,
        x0_rescale=False,
        cond_prompt=False,
        max_clean_prefix=128,
        self_cond=False,
        ce_schedule='const',
        param='eps',
        t_sampling='uniform',
        x0_mse_weight=0.0,
        head=None,
        span_noise_prob=0.0
    ):
        super().__init__()
        self.head = head  # optional TiedHead applied to the denoiser output (param=logits)
        self.span_noise_prob = span_noise_prob
        self.param = param
        self.t_sampling = t_sampling
        self.x0_mse_weight = x0_mse_weight
        self.t_bins = min(T_BINS, num_timesteps)
        if t_sampling == 'warp':  # state lives in buffers so it is checkpointed/resumed; only registered when used (old checkpoints stay loadable)
            self.register_buffer('t_loss_ema', torch.zeros(self.t_bins))
            self.register_buffer('t_loss_seen', torch.zeros(self.t_bins, dtype=torch.bool))
        self.self_cond = self_cond
        self.ce_schedule = ce_schedule
        self.cond_prompt = cond_prompt
        self.max_clean_prefix = max_clean_prefix
        self.denoise_model = denoise_model
        self.seq_len = seq_len
        self.num_timesteps = num_timesteps
        self.clip_denoised = clip_denoised
        self.loss_type = loss_type
        self.ce_weight = ce_weight
        self.x0_rescale = x0_rescale

        if beta_schedule_fn == 'linear':
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif beta_schedule_fn == 'cosine':
            betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule_fn}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # Add 1.0 at the beginning

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        # Clamp log to avoid log(0)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def denoise(self, x_t, t, clean_mask=None, x0_prev=None, sqrt_ab=None):
        """Call the denoiser; with cond_prompt a 1-channel clean-position mask (b, n) is appended to the input.

        With span_noise_prob > 0 a 1-channel per-position noise level sqrt(alpha_bar) (b, n) follows (sqrt_ab, or the constant
        sqrt(alpha_bar_t) when None, as in sampling/eval). With self_cond the previous x0 estimate (b, n, d) is appended last
        (zeros when x0_prev is None). With a tied head the output is the byte logits."""
        if self.cond_prompt:
            m = torch.zeros(x_t.shape[:2], device=x_t.device, dtype=x_t.dtype) if clean_mask is None else clean_mask.to(x_t.dtype)
            x_t = torch.cat((x_t, m.unsqueeze(-1)), dim=-1)
        if self.span_noise_prob > 0:
            sa = self.sqrt_alphas_cumprod[t][:, None].expand(x_t.shape[:2]) if sqrt_ab is None else sqrt_ab
            x_t = torch.cat((x_t, sa.to(x_t.dtype).unsqueeze(-1)), dim=-1)
        if self.self_cond:
            xp = torch.zeros_like(x_t[..., :self.denoise_model_embed_dim(x_t)]) if x0_prev is None else x0_prev.to(x_t.dtype)
            x_t = torch.cat((x_t, xp), dim=-1)
        out = self.denoise_model(x_t, t)
        return out if self.head is None else self.head(out)

    def denoise_model_embed_dim(self, x_t):
        return x_t.shape[-1] - int(self.cond_prompt) - int(self.span_noise_prob > 0)

    def x0_from_logits(self, logits, table):
        """x0_hat = softmax(logits) @ E: a per-position convex combination of embeddings (E is detached here)."""
        return F.softmax(logits.float(), dim=-1) @ table.detach().float()

    def eps_from_x0(self, x_t, t, x0):
        """eps implied by an x0 estimate: (x_t - sqrt(ab) x0) / sqrt(1 - ab). t may be (b,) or per-position (b, n)."""
        if t.dim() == 2:
            return (x_t - self.sqrt_alphas_cumprod[t].unsqueeze(-1) * x0) / self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return (x_t - extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x0) / extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

    def x0_estimate(self, x_t, t, eps, target_norm=None, table=None):
        """Detached clean-embedding estimate used as the self-conditioning input (per-position norm rescale).

        For param=logits `eps` is the network output (logits) and the estimate is softmax(logits) @ E (no rescale)."""
        if self.param == 'logits':
            return self.x0_from_logits(eps, table).detach()
        x0 = self.predict_start_from_noise(x_t, t, eps)
        if target_norm is not None:
            x0 = x0 * (target_norm / x0.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        return x0.detach()

    def sample_clean_mask(self, batch_size, device):
        """Random clean prefix per window, length uniform in [0, max_clean_prefix]."""
        n = torch.randint(0, self.max_clean_prefix + 1, (batch_size, 1), device=device)
        return torch.arange(self.seq_len, device=device)[None] < n

    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x0 from x_t and predicted noise (epsilon)."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start_embed, x_t_embed, t):
        """Compute the mean and variance of q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t_embed.shape) * x_start_embed +
            extract(self.posterior_mean_coef2, t, x_t_embed.shape) * x_t_embed
        )
        posterior_variance = extract(self.posterior_variance, t, x_t_embed.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t_embed.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample(self, x_t_embed, t, t_index, target_norm=None, x0_prev=None, return_x0=False, table=None, return_logits=False):
        """Denoise one step from x_t to x_{t-1}.

        param=logits: x0_hat = softmax(logits) @ table, DDPM posterior q(x_{t-1}|x_t, x0_hat); x0_rescale/target_norm are ignored.
        Returns out, or (out, x0_hat) with return_x0, or (out, x0_hat, logits) with return_logits.

        With x0_rescale: predict x0, rescale every position vector to target_norm (mean real embedding norm),
        then sample from the q(x_{t-1} | x_t, x0_hat) posterior. target_norm is required in that mode.
        """
        if self.param == 'logits':
            if table is None:
                raise ValueError('param=logits sampling needs the embedding table')
            logits = self.denoise(x_t_embed, t, x0_prev=x0_prev).float()
            x0_hat = self.x0_from_logits(logits, table)
            mean, _, log_var = self.q_posterior(x0_hat, x_t_embed, t)
            out = mean if t_index == 0 else mean + (0.5 * log_var).exp() * torch.randn_like(x_t_embed)
            if return_logits:
                return out, x0_hat, logits
            return (out, x0_hat) if return_x0 else out

        if self.x0_rescale:
            if target_norm is None:
                raise ValueError('x0_rescale sampling needs target_norm (mean embedding norm)')
            x0_hat = self.predict_start_from_noise(x_t_embed, t, self.denoise(x_t_embed, t, x0_prev=x0_prev))
            x0_hat = x0_hat * (target_norm / x0_hat.norm(dim=-1, keepdim=True).clamp(min=1e-8))
            mean, _, log_var = self.q_posterior(x0_hat, x_t_embed, t)
            out = mean if t_index == 0 else mean + (0.5 * log_var).exp() * torch.randn_like(x_t_embed)
            return (out, x0_hat) if return_x0 else out

        betas_t = extract(self.betas, t, x_t_embed.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t_embed.shape)
        # Corrected: alpha_t = 1 - beta_t
        sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / (1.0 - self.betas)), t, x_t_embed.shape)


        # Equation 11 in DDPM: model predicts noise
        predicted_noise = self.denoise(x_t_embed, t, x0_prev=x0_prev)
        model_mean = sqrt_recip_alphas_t * (x_t_embed - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            out = model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x_t_embed.shape)
            noise = torch.randn_like(x_t_embed)
            out = model_mean + torch.sqrt(posterior_variance_t) * noise
        if return_x0:
            return out, self.x0_estimate(x_t_embed, t, predicted_noise, target_norm)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape, target_norm=None, table=None, return_logits=False):
        """Generate samples by denoising from T to 0. param=logits with return_logits returns the last step's logits (b, n, 256)."""
        batch_size = shape[0]
        # Start with random noise
        device = self.betas.device
        x_t_embed = torch.randn(shape, device=device) # shape: (batch, seq_len, embed_dim)

        x0_prev = None  # zeros at the first step when self_cond
        for i in reversed(range(0, self.num_timesteps)):
            current_t_tensor = torch.full((batch_size,), i, device=device, dtype=torch.long)
            if self.param == 'logits':
                x_t_embed, x0_prev, last_logits = self.p_sample(x_t_embed, current_t_tensor, i, x0_prev=x0_prev if self.self_cond else None,
                                                                 table=table, return_logits=True)
            elif self.self_cond:
                x_t_embed, x0_prev = self.p_sample(x_t_embed, current_t_tensor, i, target_norm, x0_prev=x0_prev, return_x0=True)
            else:
                x_t_embed = self.p_sample(x_t_embed, current_t_tensor, i, target_norm)

        if self.param == 'logits' and return_logits:
            return last_logits
        # x_t_embed is now approximately x_0_embed
        if self.clip_denoised and not self.x0_rescale and self.param != 'logits': # Optional: clip to [-1, 1] if embeddings are normalized
             x_t_embed = torch.clamp(x_t_embed, -1., 1.)
        return x_t_embed


    def q_sample(self, x_start_embed, t, noise=None):
        """Sample x_t from x_start using the reparameterization trick (forward diffusion)."""
        noise = default(noise, lambda: torch.randn_like(x_start_embed))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start_embed.shape) * x_start_embed +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_embed.shape) * noise
        )

    # --- time warping (t_sampling="warp") ---
    def t_probs(self):
        """p(bin): WARP_MIX * normalised EMA loss + (1 - WARP_MIX) * uniform. Unseen bins take the mean seen loss."""
        nb = self.t_bins
        if self.t_sampling != 'warp' or not bool(self.t_loss_seen.any()):
            return torch.full((nb,), 1.0 / nb, device=self.betas.device)
        e = torch.where(self.t_loss_seen, self.t_loss_ema, self.t_loss_ema[self.t_loss_seen].mean()).clamp(min=0)
        return WARP_MIX * e / e.sum().clamp(min=1e-12) + (1 - WARP_MIX) / nb

    def sample_t(self, batch_size, device):
        """Training timesteps. warp: draw a bin from t_probs(), then uniform inside the bin. Deliberately NOT importance-weighted
        (the loss is reweighted toward hard noise levels on purpose)."""
        if self.t_sampling != 'warp':
            return torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        idx = torch.multinomial(self.t_probs().to(device), batch_size, replacement=True)
        lo, hi = idx * self.num_timesteps // self.t_bins, (idx + 1) * self.num_timesteps // self.t_bins
        t = lo + (torch.rand(batch_size, device=device) * (hi - lo)).long()
        return torch.minimum(t, hi - 1).long()

    def span_timesteps(self, x_bytes, t):
        """Per-example Bernoulli(span_noise_prob) switch to word-structured noise: such an example draws an independent timestep per
        word segment (word_segment_ids) from the same t-sampling distribution (uniform / warp) as sample_t. Other examples keep their
        single t at every position. Returns (t_eff (b,), t_pos (b, n)); t_eff is the example's t, or for span examples the rounded
        mean of its per-position timesteps (time embedding, loss buckets, warp statistics)."""
        b, n = x_bytes.shape
        t_seg = self.sample_t(b * n, x_bytes.device).view(b, n).gather(1, word_segment_ids(x_bytes))
        use = torch.rand(b, device=x_bytes.device) < self.span_noise_prob
        t_pos = torch.where(use[:, None], t_seg, t[:, None])
        t_eff = torch.where(use, t_seg.float().mean(1).round().long(), t)
        return t_eff, t_pos

    @torch.no_grad()
    def update_t_stats(self, t, per_sample_loss):
        v = per_sample_loss.detach().float()
        ok = torch.isfinite(v)
        b = (t * self.t_bins // self.num_timesteps).clamp(max=self.t_bins - 1)
        sums = torch.zeros_like(self.t_loss_ema).index_add_(0, b[ok], v[ok])
        cnt = torch.zeros_like(self.t_loss_ema).index_add_(0, b[ok], torch.ones_like(v[ok]))
        present = cnt > 0
        mean = sums / cnt.clamp(min=1)
        new = torch.where(self.t_loss_seen, (1 - WARP_EMA) * self.t_loss_ema + WARP_EMA * mean, mean)
        self.t_loss_ema.copy_(torch.where(present, new, self.t_loss_ema))
        self.t_loss_seen.copy_(self.t_loss_seen | present)

    def _logits_losses(self, x_start_embed, x_t_embed, t, noise, logits, x_bytes, w, table, t_noise=None):
        """param=logits: CE(logits, byte) is the loss at every t (no 1/sqrt(ab) factor anywhere)."""
        logits = logits.float()
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x_bytes.reshape(-1), reduction='none').view(logits.shape[:2])
        wsum = w.sum(1).clamp(min=1)
        per_sample = (ce * w).sum(1) / wsum
        loss = per_sample.mean()
        x0_hat = self.x0_from_logits(logits, table)
        stats = {'ce_loss': loss.detach(), 'per_sample_loss': per_sample.detach(), 't': t.detach()}
        if self.x0_mse_weight > 0:  # E (target and mixture table) detached so the term trains the logits only and cannot collapse E
            mse = (x0_hat - x_start_embed.detach().float()).pow(2).mean(dim=-1)
            loss = loss + self.x0_mse_weight * ((mse * w).sum(1) / wsum).mean()
            stats['x0_mse'] = ((mse * w).sum(1) / wsum).mean().detach()
        # diagnostics comparable with the eps parametrisation: noise-prediction error of the implied eps
        err = (noise - self.eps_from_x0(x_t_embed.float(), t if t_noise is None else t_noise, x0_hat.detach())).pow(2).mean(dim=-1)
        ps_noise = ((err * w).sum(1) / wsum).detach()
        stats.update({'noise_loss': ps_noise.mean(), 'per_sample_noise_loss': ps_noise, 'x0_pred_norm': x0_hat.detach().norm(dim=-1).mean(),
                      'x_start_norm': x_start_embed.detach().norm(dim=-1).mean(),
                      'x0_acc': ((logits.argmax(-1) == x_bytes).float() * w).sum().detach() / w.sum().clamp(min=1)})
        return loss, stats

    def p_losses(self, x_start_embed, t, noise=None, x_bytes=None, to_logits_fn=None, clean_mask=None, table=None):
        """Returns (loss, stats). stats holds detached per-component diagnostics.

        clean_mask (b, n) bool (cond_prompt only): True positions are fed clean, are not noised and carry no loss.
        param=logits needs x_bytes and the embedding table; stats['per_sample_loss'] is the per-example training loss."""
        noise = default(noise, lambda: torch.randn_like(x_start_embed))
        if clean_mask is None and self.cond_prompt and self.training:
            clean_mask = self.sample_clean_mask(x_start_embed.shape[0], x_start_embed.device)
        if not self.cond_prompt:
            clean_mask = None

        t_pos = sqrt_ab = None
        if self.span_noise_prob > 0 and self.training and x_bytes is not None:  # word-structured noise only while training
            t, t_pos = self.span_timesteps(x_bytes, t)
            sqrt_ab = self.sqrt_alphas_cumprod[t_pos]
            x_t_embed = sqrt_ab.unsqueeze(-1) * x_start_embed + self.sqrt_one_minus_alphas_cumprod[t_pos].unsqueeze(-1) * noise
        else:
            x_t_embed = self.q_sample(x_start_embed, t, noise=noise)
        if clean_mask is not None:
            x_t_embed = torch.where(clean_mask.unsqueeze(-1), x_start_embed, x_t_embed)
        logits_mode = self.param == 'logits'
        if logits_mode and (x_bytes is None or table is None):
            raise ValueError('param=logits training needs x_bytes and the embedding table')
        x0_prev = None
        if self.self_cond:
            with torch.no_grad():
                eps0 = self.denoise(x_t_embed, t, clean_mask, x0_prev=None, sqrt_ab=sqrt_ab)
                tn = x_start_embed.detach().norm(dim=-1).mean()
                x0_prev = self.x0_estimate(x_t_embed, t, eps0, tn, table=table)
                keep = (torch.rand(x_start_embed.shape[0], device=x_start_embed.device) >= 0.5).to(x0_prev.dtype)
                x0_prev = x0_prev * keep[:, None, None]  # prob 0.5 per example: zeros
        predicted_noise = self.denoise(x_t_embed, t, clean_mask, x0_prev=x0_prev, sqrt_ab=sqrt_ab)

        if logits_mode:
            w = torch.ones(x_bytes.shape, device=x_bytes.device) if clean_mask is None else (~clean_mask).float()
            loss, stats = self._logits_losses(x_start_embed, x_t_embed, t, noise, predicted_noise, x_bytes, w, table, t_noise=t_pos)
            if self.training and self.t_sampling == 'warp':
                self.update_t_stats(t, stats['per_sample_loss'])
            return loss, stats

        if self.loss_type == 'l1':
            err = (noise - predicted_noise).abs().mean(dim=-1)
        elif self.loss_type == 'l2':
            err = (noise - predicted_noise).pow(2).mean(dim=-1)
        else:
            raise ValueError(f'unknown loss type {self.loss_type}')
        if clean_mask is None:
            w = torch.ones_like(err)
        else:
            w = (~clean_mask).to(err.dtype)
        per_sample = (err * w).sum(1) / w.sum(1).clamp(min=1)
        noise_loss = per_sample.mean()
        loss = noise_loss

        stats = {'noise_loss': noise_loss.detach(), 't': t.detach(), 'per_sample_noise_loss': per_sample.detach(),
                 'per_sample_loss': per_sample.detach()}

        if x_bytes is not None and to_logits_fn is not None:
            x0_pred = self.predict_start_from_noise(x_t_embed, t, predicted_noise)
            logits = to_logits_fn(x0_pred)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x_bytes.reshape(-1), reduction='none').view(err.shape)
            if self.ce_schedule == 'alpha_bar':  # per-example weight alpha_bar_t, applied before the batch mean
                ce_ps = (ce * w).sum(1) / w.sum(1).clamp(min=1)
                ce_loss = (ce_ps * extract(self.alphas_cumprod, t, ce_ps.shape).reshape(-1)).mean()
            else:
                ce_loss = (ce * w).sum() / w.sum().clamp(min=1)
            loss = loss + self.ce_weight * ce_loss
            stats['ce_loss'] = ce_loss.detach()
            stats['x0_pred_norm'] = x0_pred.detach().norm(dim=-1).mean()
        stats['x_start_norm'] = x_start_embed.detach().norm(dim=-1).mean()
        if self.training and self.t_sampling == 'warp':
            self.update_t_stats(t, per_sample)

        return loss, stats

    def forward(self, x_start_embed, x_bytes=None, to_logits_fn=None, t=None, noise=None, clean_mask=None, table=None):
        batch_size = x_start_embed.shape[0]
        if t is None:
            t = self.sample_t(batch_size, x_start_embed.device)
        return self.p_losses(x_start_embed, t, noise=noise, x_bytes=x_bytes, to_logits_fn=to_logits_fn, clean_mask=clean_mask, table=table)


# --- Data Loading ---
def load_text_data(data_dir):
    text = ""
    txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in directory: {data_dir}")
    for filepath in txt_files:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text += f.read()
    return text


class ByteTokenizer:
    vocab_size = 256

    def encode(self, string: str) -> list[int]:
        return list(string.encode('utf-8', errors='replace'))

    def decode(self, byte_list: list[int]) -> str:
        return bytes([b if 0 <= b <= 255 else ord('?') for b in byte_list]).decode('utf-8', errors='replace')


class TextDataset(Dataset):
    """Non-overlapping windows of seq_len bytes; trailing partial window is dropped, not padded."""
    def __init__(self, tokenized_text_bytes, seq_len: int):
        self.seq_len = seq_len
        self.data = torch.tensor(tokenized_text_bytes, dtype=torch.long)
        self.num_sequences = self.data.size(0) // seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return self.data[start:start + self.seq_len]


# --- Main Model Combining Embedder, U-Net, Diffusion, and Logits Head ---
class ByteDiffusionLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        validate(cfg)
        m, d = cfg.model, cfg.diffusion
        self.cfg = cfg
        self.byte_embedder = ByteEmbedding(m.vocab_size, m.byte_embed_dim, m.embed_norm)
        out_dim = m.vocab_size if d.param == 'logits' and not m.tied_head else m.byte_embed_dim  # logits head: one logit per byte (tied head: a byte_embed_dim projection)
        in_extra = int(d.cond_prompt) + int(d.span_noise_prob > 0)  # extra input channels: clean mask, sqrt(alpha_bar)
        head = TiedHead(self.byte_embedder, m.embed_norm if m.embed_norm > 0 else math.sqrt(m.byte_embed_dim)) if m.tied_head else None

        if m.denoiser == "birnn":  # attribute keeps the name denoise_unet so state_dict keys stay uniform
            self.denoise_unet = BiRNNDenoiser(m.model_dim, m.byte_embed_dim + in_extra, m.rnn_layers, m.rnn_conv, m.rnn_ffn_mult,
                                              m.dropout, out_dim=out_dim)
        elif m.denoiser == "hybrid":
            self.denoise_unet = HybridDenoiser(m, m.byte_embed_dim * (1 + int(d.self_cond)) + in_extra, out_dim)
        elif m.denoiser == "rpw":
            self.denoise_unet = RPWDenoiser(m, m.byte_embed_dim * (1 + int(d.self_cond)) + in_extra, out_dim)
        elif m.denoiser == "hier":
            self.denoise_unet = HierDenoiser(m, m.byte_embed_dim * (1 + int(d.self_cond)) + in_extra, out_dim)
        else:
            self.denoise_unet = UNet1D(
                model_dim=m.model_dim,
                in_embed_dim=m.byte_embed_dim * (1 + int(d.self_cond)) + in_extra,
                out_dim=out_dim,
                dim_mults=m.dim_mults,
                time_emb_dim_mult=m.time_emb_mult,
                resnet_groups=m.resnet_groups,
                attn_heads=m.attn_heads,
                attn_dim_head=m.attn_dim_head,
                dropout=m.dropout,
            )

        self.gaussian_diffusion = GaussianDiffusion(
            denoise_model=self.denoise_unet,
            seq_len=m.seq_len,
            num_timesteps=d.num_timesteps,
            beta_schedule_fn=d.beta_schedule,
            beta_start=d.beta_start,
            beta_end=d.beta_end,
            loss_type=d.loss_type,
            clip_denoised=d.clip_denoised,
            ce_weight=d.ce_weight,
            x0_rescale=d.x0_rescale,
            cond_prompt=d.cond_prompt,
            self_cond=d.self_cond,
            ce_schedule=d.ce_schedule,
            param=d.param,
            t_sampling=d.t_sampling,
            x0_mse_weight=d.x0_mse_weight,
            head=head,
            span_noise_prob=d.span_noise_prob,
        )

    def embedding_table(self):
        return self.byte_embedder.table()

    def get_logits(self, x):
        return F.linear(x, self.embedding_table())

    def forward(self, x_bytes, t=None, noise=None, clean_mask=None):
        """Training forward pass: returns (loss, stats)."""
        x_start_embed = self.byte_embedder(x_bytes)
        return self.gaussian_diffusion(x_start_embed, x_bytes=x_bytes, to_logits_fn=self.get_logits, t=t, noise=noise,
                                       clean_mask=clean_mask, table=self.embedding_table())

    def mean_embed_norm(self):
        return self.embedding_table().detach().norm(dim=-1).mean().item()

    @torch.no_grad()
    def sample(self, batch_size=1):
        self.eval()
        shape = (batch_size, self.gaussian_diffusion.seq_len, self.byte_embedder.embedding.embedding_dim)
        if self.gaussian_diffusion.param == 'logits':  # final decode: argmax of the last step's logits
            return torch.argmax(self.gaussian_diffusion.p_sample_loop(shape, table=self.embedding_table().detach(), return_logits=True), dim=-1)
        x0_embed_predicted = self.gaussian_diffusion.p_sample_loop(shape, target_norm=self.mean_embed_norm())
        logits = self.get_logits(x0_embed_predicted)
        return torch.argmax(logits, dim=-1)


if __name__ == '__main__':
    from train import main
    main()
