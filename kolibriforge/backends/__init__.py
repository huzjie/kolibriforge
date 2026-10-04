"""Backend registry.

CRITICAL: submodules are imported explicitly below. The `@register_backend`
decorator runs at import time, so without these imports `list_backends()` would
return an empty dict and `get_backend()` would raise "unknown backend".
"""
_BACKENDS = {}


def register_backend(name):
    def deco(cls):
        _BACKENDS[name] = cls
        return cls
    return deco


def list_backends():
    return sorted(_BACKENDS.keys())


def get_backend(name):
    if name not in _BACKENDS:
        raise KeyError(f"unknown backend '{name}'; available: {list_backends()}")
    return _BACKENDS[name]


from . import mock, cpu, openai, vllm, transformers  # noqa: E402,F401  (register side effects)

__all__ = ["register_backend", "list_backends", "get_backend"]
