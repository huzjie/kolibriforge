"""Benchmark registry.

Submodules are imported explicitly so the `@register_benchmark` decorator runs
and `list_benchmarks()` is populated. (Same gotcha as the backends registry.)
"""
_BENCHMARKS = {}


def register_benchmark(name):
    def deco(fn):
        _BENCHMARKS[name] = fn
        return fn
    return deco


def list_benchmarks():
    return sorted(_BENCHMARKS.keys())


def get_benchmark(name):
    return _BENCHMARKS[name]


from . import honesty, abstention, math, longcontext, tokenization  # noqa: E402,F401

__all__ = ["register_benchmark", "list_benchmarks", "get_benchmark"]
