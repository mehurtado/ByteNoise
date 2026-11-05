"""Adapters over the original model code (main.ByteDiffusionLM, models_lm.ByteLM). The wrapped modules are untouched."""
import torch
import torch.nn.functional as F

from ..core import legacy
from ..core.config import deep_merge
from .base import CAP_CONTINUE, CAP_GENERATE, CAP_LOSS, CAP_SCORE, ByteModel, LossOut, ScoreOut, UnsupportedError
from .registry import register_model


def legacy_config(cfg):
    """{preset, set} -> legacy config.Config."""
    legacy.ensure()
    import config as C
    base = C.get_preset(cfg["preset"]).to_dict()
    return C.Config.from_dict(deep_merge(base, cfg.get("set") or {}))


class _LegacyBase(ByteModel):
    arch_kinds = ()

    def __init__(self, module, lcfg):
        self.module = module
        self.lcfg = lcfg
        self.name = lcfg.name
        self.device = "cpu"

    @classmethod
    def from_config(cls, cfg):
        legacy.ensure()
        import train
        lcfg = legacy_config(cfg)
        if lcfg.arch not in cls.arch_kinds:
            raise ValueError(f"preset {cfg['preset']!r} has arch {lcfg.arch!r}, not valid for family {cls.family!r}")
        if "seed" in cfg:
            torch.manual_seed(int(cfg["seed"]))
        return cls(train.build_model(lcfg), lcfg)

    @classmethod
    def from_module(cls, module, lcfg):
        return cls(module, lcfg)

    def load_weights(self, state, which="ema"):
        legacy.ensure()
        import train
        from main import EMA
        if which not in ("ema", "raw"):
            raise ValueError(f"which must be 'ema' or 'raw', got {which!r}")
        if which == "ema" and "ema_state_dict" not in state:
            raise ValueError("checkpoint has no EMA weights; use which='raw'")
        ema = EMA(self.module, decay=self.lcfg.train.ema_decay)
        train.load_checkpoint_into(state, self.module, ema)
        if which == "ema":
            self.module.load_state_dict(ema.model.state_dict())
        self.module.eval()
        return self

    def to(self, device):
        self.module.to(device)
        self.device = device
        return self

    def num_params(self):
        return sum(p.numel() for p in self.module.parameters())


@register_model("ar_lm")
class ArLm(_LegacyBase):
    arch_kinds = ("transformer", "recurrent")
    capabilities = frozenset({CAP_LOSS, CAP_SCORE, CAP_GENERATE, CAP_CONTINUE})

    @property
    def seq_len(self):
        return self.lcfg.lm.seq_len

    @classmethod
    def sampler_schema(cls):
        return {"temperature": {"type": "float", "default": 0.8, "min": 0.0, "max": 5.0, "help": "0 = greedy"},
                "top_k": {"type": "int", "default": 40, "min": 0, "max": 256, "help": "0 = off"}}

    def loss(self, batch):
        loss, stats = self.module(batch.to(self.device).long())
        return LossOut(loss, {k: float(v) for k, v in stats.items()})

    @torch.no_grad()
    def score(self, window):
        self.module.eval()
        w = window.to(self.device).long()
        lg = self.module.logits(w[:, :-1]).float()
        tgt = w[:, 1:]
        nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1), reduction="none").view(tgt.shape)
        return ScoreOut(nll.cpu(), (lg.argmax(-1) == tgt).cpu(), exact=True)

    @torch.no_grad()
    def generate(self, prompt, n_bytes, seed=0, **sampler):
        s = self.check_sampler(sampler)
        self.module.eval()
        ids = torch.tensor(list(prompt or b"\n"), dtype=torch.long, device=self.device)[None]
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        out = self.module.generate(ids, int(n_bytes), temperature=s["temperature"], top_k=s["top_k"], generator=g)
        return bytes(out[0, ids.size(1):].cpu().to(torch.uint8).tolist())


@register_model("diffusion")
class Diffusion(_LegacyBase):
    """Unconditional byte diffusion. Prefix-conditioned continuation exists only in uncommitted work, so `continue` is not declared."""
    arch_kinds = ("diffusion",)
    capabilities = frozenset({CAP_LOSS, CAP_GENERATE})

    @property
    def seq_len(self):
        return self.lcfg.model.seq_len

    def loss(self, batch):
        loss, stats = self.module(batch.to(self.device).long())
        return LossOut(loss, {k: float(v) for k, v in stats.items() if torch.as_tensor(v).numel() == 1})

    @torch.no_grad()
    def generate(self, prompt, n_bytes, seed=0, **sampler):
        self.check_sampler(sampler)
        if prompt:
            raise UnsupportedError("diffusion adapter is unconditional; pass an empty prompt")
        if n_bytes > self.seq_len:
            raise ValueError(f"diffusion samples exactly seq_len={self.seq_len} bytes; asked for {n_bytes}")
        torch.manual_seed(int(seed))
        row = self.module.sample(1)[0].cpu().to(torch.uint8).tolist()
        return bytes(row[:n_bytes])

    def describe(self):
        d = super().describe()
        d["denoiser"] = self.lcfg.model.denoiser
        return d
