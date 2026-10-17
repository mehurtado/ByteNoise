import copy
import hashlib
import json
from dataclasses import dataclass, field, asdict, replace


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    byte_embed_dim: int = 128
    model_dim: int = 192
    dim_mults: tuple = (1, 2, 3, 4)
    time_emb_mult: int = 4
    resnet_groups: int = 8
    attn_heads: int = 6
    dropout: float = 0.1
    seq_len: int = 256
    denoiser: str = "unet"           # "unet" | "birnn" | "hybrid"
    rnn_layers: int = 8               # birnn only: block count (width = model_dim)
    rnn_conv: int = 7                 # birnn only: centered depthwise conv width (odd)
    rnn_ffn_mult: float = 2.667       # birnn only
    # hybrid only (excluded from architecture_hash for other denoisers); width = model_dim, depth = hybrid_layers
    hybrid_layers: int = 6
    hybrid_branches: str = "conv,rnn,attn,ngram"   # comma list, subset of conv,rnn,attn,ngram
    hybrid_heads: int = 8
    hybrid_rnn_width: int = 128       # inner width of the bidirectional recurrent branch
    hybrid_ffn_mult: float = 4.0
    hybrid_ngram_kernel: int = 5      # odd
    embed_norm: float = 0.0           # >0: embedding table rows are L2-normalised to this norm (e.g. sqrt(byte_embed_dim) = unit per-element RMS); 0 = off (excluded from hash)

    @property
    def attn_dim_head(self):
        return self.model_dim // self.attn_heads


@dataclass(frozen=True)
class LMConfig:
    arch: str = "transformer"        # "transformer" | "recurrent"
    vocab_size: int = 256
    d_model: int = 512
    n_layers: int = 8                 # unique blocks
    n_heads: int = 8
    ffn_mult: float = 2.667
    seq_len: int = 512
    dropout: float = 0.0
    loops: int = 1                    # repeat the whole block stack this many times (shared weights)
    conv_kernel: int = 4              # recurrent mixer causal conv width
    expansion: int = 1                # recurrent mixer inner width multiplier
    iter_embedding: bool = False      # learned per-loop offset added before each pass


@dataclass(frozen=True)
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_schedule: str = "cosine"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    loss_type: str = "l2"
    ce_weight: float = 0.1
    clip_denoised: bool = True
    x0_rescale: bool = False          # sampler-only: rescale x0_hat to mean embedding norm each step
    self_cond: bool = False           # feed previous x0 estimate as extra input channels (default-off; excluded from hash when False)
    ce_schedule: str = "const"        # "const" | "alpha_bar": per-example CE weight ce_weight*alphas_cumprod[t] (default-off; excluded from hash when "const")
    cond_prompt: bool = False         # train with a random clean prefix (inpainting conditioning); denoiser gets a mask channel
    param: str = "eps"                # "eps" (predict noise) | "logits" (CDCD-style: 256 byte logits, x0_hat = softmax @ E; CE is the loss). "eps" excluded from hash
    t_sampling: str = "uniform"       # "uniform" | "warp" (loss-EMA-weighted timestep sampling, 50 bins); "uniform" excluded from hash
    x0_mse_weight: float = 0.0        # logits only: weight of an extra MSE(x0_hat, E[x]) term (E detached); 0 = off (excluded from hash)


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "./dataset/USE"
    batch_size: int = 16
    grad_accum_steps: int = 16
    learning_rate: float = 5e-5
    lr_min: float = 1e-6
    max_updates: int = 40
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    ema_warmup: bool = True
    seed: int = 0
    log_interval: int = 10
    save_interval: int = 10
    sample_interval: int = 0
    noise_buckets: int = 4
    weight_decay: float = 0.01
    warmup_updates: int = 0
    precision: str = "fp32"           # fp32 | bf16 | fp16
    max_seconds: float = 0.0          # wall-clock budget, 0 = unlimited
    eval_interval: int = 0            # updates between validation passes, 0 = off
    eval_batches: int = 32
    data: str = ""                    # prepared dataset dir (data/<name>); empty = legacy data_dir
    task_mix: str = ""                # comma list of controlled tasks, used when data == "tasks"


@dataclass(frozen=True)
class Config:
    name: str = "original"
    arch: str = "diffusion"           # "diffusion" | "transformer" | "recurrent"
    lm: LMConfig = field(default_factory=LMConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        m = dict(d["model"])
        m["dim_mults"] = tuple(m["dim_mults"])
        return Config(
            name=d["name"],
            arch=d.get("arch", "diffusion"),
            lm=LMConfig(**d["lm"]) if "lm" in d else LMConfig(),
            model=ModelConfig(**m),
            diffusion=DiffusionConfig(**d["diffusion"]),
            train=TrainConfig(**d["train"]),
        )

    def architecture_hash(self):
        full = self.to_dict()
        if self.arch == "diffusion":
            diff = {k: v for k, v in full["diffusion"].items() if k != "x0_rescale" and not (k in ("cond_prompt", "self_cond") and not v) and not (k == "ce_schedule" and v == "const") and not (k == "param" and v == "eps") and not (k == "t_sampling" and v == "uniform") and not (k == "x0_mse_weight" and v == 0.0)}  # sampler-only / default-off; keeps old hashes
            mod = {k: v for k, v in full["model"].items() if not (k == "embed_norm" and v == 0.0)}
            hyb = [k for k in mod if k.startswith("hybrid_")]
            if self.model.denoiser == "unet":  # birnn/hybrid-only fields don't exist for U-Net configs; keeps old hashes
                mod = {k: v for k, v in mod.items() if k not in ("denoiser", "rnn_layers", "rnn_conv", "rnn_ffn_mult", *hyb)}
            elif self.model.denoiser == "birnn":  # keeps birnn hashes from before the hybrid fields existed
                mod = {k: v for k, v in mod.items() if k not in hyb}
            else:  # hybrid reuses rnn_conv only
                mod = {k: v for k, v in mod.items() if k not in ("rnn_layers", "rnn_ffn_mult")}
            parts = {"model": mod, "diffusion": diff}
        else:
            parts = {"arch": self.arch, "lm": full["lm"]}
        blob = json.dumps(parts, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


PRESETS = {
    "original": Config(name="original"),
    "compact": Config(
        name="compact",
        model=ModelConfig(byte_embed_dim=64, model_dim=96, attn_heads=4),
    ),
    "birnn": Config(
        name="birnn",
        model=ModelConfig(byte_embed_dim=64, model_dim=256, denoiser="birnn", rnn_layers=8),
    ),
    "hybrid": Config(
        name="hybrid",
        model=ModelConfig(byte_embed_dim=64, model_dim=256, seq_len=256, denoiser="hybrid", hybrid_layers=7),
    ),
    "debug": Config(
        name="debug",
        model=ModelConfig(byte_embed_dim=16, model_dim=32, dim_mults=(1, 2), attn_heads=2, dropout=0.0, seq_len=32),
        diffusion=DiffusionConfig(num_timesteps=50),
        train=TrainConfig(batch_size=4, grad_accum_steps=2, max_updates=20, log_interval=5, save_interval=5, learning_rate=1e-3),
    ),
}


def _lm(name, arch, **kw):
    lm_kw = {k: kw.pop(k) for k in list(kw) if k in LMConfig.__dataclass_fields__}
    tr = dict(batch_size=32, grad_accum_steps=2, learning_rate=1e-3, lr_min=1e-4, max_updates=2000, weight_decay=0.1,
              warmup_updates=100, precision="bf16", ema_warmup=True, ema_decay=0.999, log_interval=50, save_interval=500, eval_interval=250)
    tr.update(kw)
    return Config(name=name, arch=arch, lm=LMConfig(arch=arch, **lm_kw), train=TrainConfig(**tr))


PRESETS.update({
    "tf_debug": _lm("tf_debug", "transformer", d_model=32, n_layers=2, n_heads=2, seq_len=32, batch_size=4, grad_accum_steps=1,
                    max_updates=20, log_interval=5, save_interval=10, eval_interval=10, eval_batches=2, warmup_updates=2),
    "rec_debug": _lm("rec_debug", "recurrent", d_model=32, n_layers=2, n_heads=2, seq_len=32, batch_size=4, grad_accum_steps=1,
                     max_updates=20, log_interval=5, save_interval=10, eval_interval=10, eval_batches=2, warmup_updates=2),
    "tf_task": _lm("tf_task", "transformer", d_model=256, n_layers=4, n_heads=4, seq_len=128, batch_size=64, grad_accum_steps=1,
                   max_updates=2000, data="tasks-v1", ema_decay=0.99, save_interval=500, eval_interval=250),
    "rec_task": _lm("rec_task", "recurrent", d_model=256, n_layers=4, n_heads=4, seq_len=128, batch_size=64, grad_accum_steps=1,
                    max_updates=2000, data="tasks-v1", ema_decay=0.99, save_interval=500, eval_interval=250),
    "tf_nat_a": _lm("tf_nat_a", "transformer", d_model=384, n_layers=12, n_heads=6, seq_len=512, batch_size=32, grad_accum_steps=2,
                    max_updates=100000, max_seconds=900, data="natural-v1", eval_interval=200, save_interval=200),
    "tf_nat_b": _lm("tf_nat_b", "transformer", d_model=512, n_layers=8, n_heads=8, seq_len=512, batch_size=32, grad_accum_steps=2,
                    max_updates=100000, max_seconds=900, data="natural-v1", eval_interval=200, save_interval=200),
    "tf_nat_c": _lm("tf_nat_c", "transformer", d_model=640, n_layers=10, n_heads=10, seq_len=512, batch_size=32, grad_accum_steps=2,
                    learning_rate=6e-4, max_updates=100000, max_seconds=900, data="natural-v1", eval_interval=200, save_interval=200),
    **{name: _lm(name, "recurrent", d_model=d, n_layers=n, n_heads=8, loops=l, seq_len=512, batch_size=16, grad_accum_steps=4,
                 max_updates=100000, max_seconds=900, data="natural-v1", eval_interval=200, save_interval=200)
       for name, d, n, l in [("rec_nat_L8", 512, 8, 1), ("rec_nat_s2x1", 512, 2, 1), ("rec_nat_s2x2", 512, 2, 2),
                             ("rec_nat_s2x4", 512, 2, 4), ("rec_nat_sep4", 512, 4, 1), ("rec_nat_wide", 1024, 2, 4)]},
    "tf_small": _lm("tf_small", "transformer", d_model=512, n_layers=8, n_heads=8, seq_len=512),
})


def get_preset(name):
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return copy.deepcopy(PRESETS[name])


def override(cfg, **train_overrides):
    return replace(cfg, train=replace(cfg.train, **train_overrides))


def validate(cfg):
    m, d, t = cfg.model, cfg.diffusion, cfg.train
    errors = []
    if cfg.arch not in ("diffusion", "transformer", "recurrent"):
        errors.append(f"unknown arch {cfg.arch!r}")
    if cfg.arch != "diffusion":
        lm = cfg.lm
        if lm.arch != cfg.arch:
            errors.append(f"lm.arch ({lm.arch}) must match arch ({cfg.arch})")
        if lm.d_model % lm.n_heads != 0:
            errors.append(f"d_model ({lm.d_model}) must be divisible by n_heads ({lm.n_heads})")
        if cfg.arch == "transformer" and (lm.d_model // lm.n_heads) % 2 != 0:
            errors.append("head dim must be even for rotary embeddings")
        if lm.loops < 1 or lm.n_layers < 1:
            errors.append("loops and n_layers must be >= 1")
    if d.self_cond and m.denoiser not in ("unet", "hybrid"):
        errors.append("diffusion.self_cond is only supported for the unet and hybrid denoisers")
    if d.ce_schedule not in ("const", "alpha_bar"):
        errors.append(f"unknown ce_schedule {d.ce_schedule!r}")
    if d.param not in ("eps", "logits"):
        errors.append(f"unknown diffusion.param {d.param!r}")
    if d.t_sampling not in ("uniform", "warp"):
        errors.append(f"unknown diffusion.t_sampling {d.t_sampling!r}")
    if m.embed_norm < 0 or d.x0_mse_weight < 0:
        errors.append("embed_norm and x0_mse_weight must be >= 0")
    if d.param == "logits" and cfg.arch == "diffusion" and m.vocab_size != 256:
        errors.append("diffusion.param=logits needs vocab_size 256")
    if t.precision not in ("fp32", "bf16", "fp16"):
        errors.append(f"unknown precision {t.precision!r}")
    if cfg.arch != "diffusion":
        if t.grad_accum_steps < 1 or t.batch_size < 1 or t.max_updates < 1:
            errors.append("batch_size, grad_accum_steps and max_updates must be >= 1")
        if errors:
            raise ValueError("invalid config:\n  - " + "\n  - ".join(errors))
        return cfg
    if m.denoiser not in ("unet", "birnn", "hybrid"):
        errors.append(f"unknown denoiser {m.denoiser!r}")
    if m.denoiser == "hybrid":
        br = [b for b in m.hybrid_branches.split(",") if b]
        if not br or any(b not in ("conv", "rnn", "attn", "ngram") for b in br) or len(set(br)) != len(br):
            errors.append(f"hybrid_branches must be a non-empty unique subset of conv,rnn,attn,ngram (got {m.hybrid_branches!r})")
        if m.hybrid_layers < 1 or m.hybrid_heads < 1 or m.model_dim % m.hybrid_heads != 0 or (m.model_dim // max(m.hybrid_heads, 1)) % 2:
            errors.append("hybrid needs hybrid_layers >= 1 and model_dim divisible by hybrid_heads with an even head dim")
        if m.model_dim % 2 != 0 or m.hybrid_rnn_width < 1:
            errors.append("hybrid model_dim must be even and hybrid_rnn_width >= 1")
        if m.hybrid_ngram_kernel % 2 == 0 or m.rnn_conv % 2 == 0:
            errors.append("hybrid_ngram_kernel and rnn_conv must be odd")
    if m.denoiser in ("birnn", "hybrid"):
        if m.denoiser == "birnn":
            if m.rnn_layers < 1 or m.rnn_conv < 1 or m.rnn_conv % 2 == 0:
                errors.append("birnn needs rnn_layers >= 1 and an odd rnn_conv")
            if m.model_dim % 2 != 0:
                errors.append("birnn model_dim must be even (sinusoidal time embedding)")
        if d.beta_schedule not in ("linear", "cosine"):
            errors.append(f"unknown beta_schedule {d.beta_schedule!r}")
        if d.loss_type not in ("l1", "l2"):
            errors.append(f"unknown loss_type {d.loss_type!r}")
        if t.grad_accum_steps < 1 or t.batch_size < 1 or t.max_updates < 1:
            errors.append("batch_size, grad_accum_steps and max_updates must be >= 1")
        if errors:
            raise ValueError("invalid config:\n  - " + "\n  - ".join(errors))
        return cfg
    if m.attn_heads <= 0 or m.model_dim % m.attn_heads != 0:
        errors.append(f"model_dim ({m.model_dim}) must be divisible by attn_heads ({m.attn_heads})")
    for mult in m.dim_mults:
        ch = m.model_dim * mult
        if ch % m.resnet_groups != 0:
            errors.append(f"channels {ch} (model_dim*{mult}) not divisible by resnet_groups ({m.resnet_groups})")
    if m.model_dim % m.resnet_groups != 0:
        errors.append(f"model_dim ({m.model_dim}) not divisible by resnet_groups ({m.resnet_groups})")
    halvings = len(m.dim_mults) - 1
    if halvings < 0 or m.seq_len % (2 ** halvings) != 0:
        errors.append(f"seq_len ({m.seq_len}) must be divisible by 2**{halvings} for the U-Net path")
    if d.beta_schedule not in ("linear", "cosine"):
        errors.append(f"unknown beta_schedule {d.beta_schedule!r}")
    if d.loss_type not in ("l1", "l2"):
        errors.append(f"unknown loss_type {d.loss_type!r}")
    if t.grad_accum_steps < 1 or t.batch_size < 1 or t.max_updates < 1:
        errors.append("batch_size, grad_accum_steps and max_updates must be >= 1")
    if errors:
        raise ValueError("invalid config:\n  - " + "\n  - ".join(errors))
    return cfg
