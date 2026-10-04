"""Benchmark runner: run all registered benchmarks and print a table."""
from ..utils.logging import get_logger

log = get_logger("kolibriforge.bench")


def run_all(cfg, backend=None):
    from . import list_benchmarks, get_benchmark
    from ..backends import get_backend
    backend = backend or get_backend("mock")(cfg)
    results = {}
    for name in list_benchmarks():
        try:
            results[name] = get_benchmark(name)(cfg, backend=backend)
            log.info("bench %-14s %s", name, results[name])
        except Exception as e:
            results[name] = {"error": str(e)}
            log.error("bench %-14s FAILED: %s", name, e)
    return results
