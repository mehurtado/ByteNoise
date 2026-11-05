"""An eval protocol is a yaml file; its hash is stamped on every result so comparisons can be validated."""
from dataclasses import dataclass

from ..core.config import load_yaml, resolve_ref, stable_hash

KNOWN_KEYS = {"name", "data", "split", "seed", "modes"}


@dataclass(frozen=True)
class Protocol:
    name: str
    spec: dict
    hash: str

    @property
    def modes(self):
        return self.spec["modes"]


def make_protocol(spec, data=None):
    spec = dict(spec)
    bad = sorted(set(spec) - KNOWN_KEYS)
    if bad:
        raise ValueError(f"unknown protocol keys {bad}")
    if not spec.get("modes"):
        raise ValueError("protocol needs a non-empty `modes` mapping")
    if data:
        spec["data"] = data
    if not spec.get("data"):
        raise ValueError("protocol needs `data` (a dataset name), or pass one explicitly")
    spec.setdefault("name", "adhoc")
    spec.setdefault("split", "val")
    spec.setdefault("seed", 1234)
    return Protocol(spec["name"], spec, stable_hash(spec))


def load_protocol(ref, data=None, allow_path=False):
    return make_protocol(load_yaml(resolve_ref("protocols", ref, allow_path)), data)
