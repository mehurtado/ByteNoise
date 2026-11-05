"""Model families register here; configs refer to them by name."""
_MODELS = {}


def register_model(name):
    def deco(cls):
        if name in _MODELS and _MODELS[name] is not cls:
            raise ValueError(f"model family {name!r} already registered")
        cls.family = name
        _MODELS[name] = cls
        return cls
    return deco


def _load_builtin():
    from . import legacy_adapters  # noqa: F401  (registers diffusion and ar_lm)


def list_models():
    _load_builtin()
    return sorted(_MODELS)


def get_model_class(name):
    _load_builtin()
    if name not in _MODELS:
        raise KeyError(f"unknown model family {name!r}; registered: {sorted(_MODELS)}")
    return _MODELS[name]


def build_model(cfg):
    """cfg = {family, preset, set}: instantiate a fresh model."""
    return get_model_class(cfg["family"]).from_config(cfg)
