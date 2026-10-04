"""Merlin-Arthur abstention tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from kolibriforge.config import load_config
from kolibriforge.merlin.gate import AbstentionGate, GATE_ACCEPT, GATE_ABSTAIN, GATE_REJECT


class TestGate(unittest.TestCase):
    def setUp(self):
        self.gate = AbstentionGate(abstain_threshold=0.5, reject_threshold=0.2)

    def test_accept(self):
        self.assertEqual(self.gate.decide(0.9), GATE_ACCEPT)

    def test_abstain(self):
        self.assertEqual(self.gate.decide(0.35), GATE_ABSTAIN)

    def test_reject(self):
        self.assertEqual(self.gate.decide(0.1), GATE_REJECT)


class TestMerlinArthur(unittest.TestCase):
    def test_decision_fields(self):
        from kolibriforge.merlin.protocol import MerlinArthur
        cfg = load_config()
        ma = MerlinArthur(cfg)
        v = ma.decide("What is 2+2?", "4", skill=1.0)
        for k in ("question", "answer", "confidence", "action"):
            self.assertIn(k, v)


if __name__ == "__main__":
    unittest.main()
