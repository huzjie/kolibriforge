"""Sparse MoE tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from kolibriforge.config import load_config
from kolibriforge.model.kolibri import KolibriMoE
from kolibriforge.moe.balance import load_balance_loss


class TestMoE(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.model = KolibriMoE(self.cfg)

    def test_sparse_ratio(self):
        self.assertGreater(self.model.total_params, self.model.active_params)
        self.assertAlmostEqual(self.model.total_params / self.model.active_params,
                               self.cfg.moe.sparse_ratio, delta=1.0)

    def test_forward_shape(self):
        from kolibriforge.utils.stable import stable_vector
        xs = [stable_vector(f"tok:{i}", self.cfg.hidden) for i in range(4)]
        last, aux = self.model.forward(xs)
        self.assertEqual(len(last), self.cfg.hidden)

    def test_load_balance_zero_when_empty(self):
        self.assertEqual(load_balance_loss([], 0), 0.0)


if __name__ == "__main__":
    unittest.main()
