"""Benchmark registry + honesty benchmark tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from kolibriforge.config import load_config
from kolibriforge.bench import list_benchmarks, get_benchmark


class TestBench(unittest.TestCase):
    def test_registry_populated(self):
        names = list_benchmarks()
        for n in ("honesty", "abstention", "math", "longcontext", "tokenization"):
            self.assertIn(n, names)

    def test_honesty_returns_metrics(self):
        cfg = load_config()
        from kolibriforge.backends import get_backend
        b = get_backend("mock")(cfg)
        res = get_benchmark("honesty")(cfg, backend=b, n=50)
        for k in ("ece", "hallucination_rate", "mean_accuracy"):
            self.assertIn(k, res)


if __name__ == "__main__":
    unittest.main()
