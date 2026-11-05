"""Load a trained model (legacy checkpoint.pt run dir) as a ByteModel."""
from pathlib import Path

import torch

from ..core import legacy
from ..runstore.reader import payload_dir
from .registry import get_model_class


def load_run_model(run_dir, which="ema", device="cpu", ckpt_name=None):
    legacy.ensure()
    import train
    from config import Config
    path = payload_dir(run_dir) / (ckpt_name or train.CHECKPOINT_NAME)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    lcfg = Config.from_dict(state["config"])
    family = "diffusion" if lcfg.arch == "diffusion" else "ar_lm"
    model = get_model_class(family).from_module(train.build_model(lcfg), lcfg)
    model.load_weights(state, which)
    return model.to(device)
