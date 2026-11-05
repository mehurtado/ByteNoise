"""The ByteModel contract: bytes in, bytes/scores out. Everything above this layer (eval, UI, agent) uses only this."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

CAP_LOSS = "loss"
CAP_SCORE = "score"
CAP_GENERATE = "generate"
CAP_CONTINUE = "continue"  # generate() honours a non-empty prompt


class UnsupportedError(NotImplementedError):
    """Raised when an endpoint is called on a model that did not declare the capability."""


@dataclass
class LossOut:
    total: torch.Tensor
    parts: dict = field(default_factory=dict)


@dataclass
class ScoreOut:
    nll: torch.Tensor    # (b, n) nats of each target byte given its prefix
    top1: torch.Tensor   # (b, n) bool, argmax prediction equals the target
    exact: bool          # True for exact likelihoods (AR); False for bounds


class ByteModel(ABC):
    family = ""
    capabilities = frozenset()

    name: str
    seq_len: int

    # --- construction ---
    @classmethod
    @abstractmethod
    def from_config(cls, cfg):
        """cfg: a model yaml ({family, preset, set})."""

    @abstractmethod
    def load_weights(self, state, which="ema"):
        """state: a checkpoint payload already read from disk."""

    # --- training endpoint ---
    def loss(self, batch):
        raise UnsupportedError(f"{self.family} does not implement loss")

    # --- inference endpoints (bytes) ---
    def score(self, window):
        raise UnsupportedError(f"{self.family} does not implement score")

    def generate(self, prompt, n_bytes, seed=0, **sampler):
        raise UnsupportedError(f"{self.family} does not implement generate")

    @classmethod
    def sampler_schema(cls):
        """{knob: {type, default, min, max, help}}; drives CLI flags and UI forms."""
        return {}

    # --- housekeeping ---
    @abstractmethod
    def to(self, device):
        ...

    @abstractmethod
    def num_params(self):
        ...

    def describe(self):
        return {"family": self.family, "name": self.name, "seq_len": self.seq_len,
                "capabilities": sorted(self.capabilities), "params": self.num_params(),
                "sampler": self.sampler_schema()}

    def check_sampler(self, sampler):
        schema = self.sampler_schema()
        bad = sorted(set(sampler) - set(schema))
        if bad:
            raise ValueError(f"unknown sampler knobs {bad}; {self.family} accepts {sorted(schema)}")
        out = {}
        for k, spec in schema.items():
            v = {"int": int, "float": float, "bool": bool}[spec["type"]](sampler.get(k, spec["default"]))
            if ("min" in spec and v < spec["min"]) or ("max" in spec and v > spec["max"]):
                raise ValueError(f"sampler knob {k}={v} outside [{spec.get('min')}, {spec.get('max')}]")
            out[k] = v
        return out
