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
            parts = {"model": full["model"], "diffusion": full["diffusion"]}
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
    if t.precision not in ("fp32", "bf16", "fp16"):
        errors.append(f"unknown precision {t.precision!r}")
    if cfg.arch != "diffusion":
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
