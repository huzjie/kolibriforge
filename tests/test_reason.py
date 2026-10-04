"""Reasoning budget tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from kolibriforge.config import load_config
from kolibriforge.reason.budget import ReasoningBudget, LEVELS


class TestBudget(unittest.TestCase):
    def setUp(self):
        self.rb = ReasoningBudget(load_config())

    def test_levels(self):
        self.assertEqual(len(LEVELS), 4)
        self.assertEqual(LEVELS[0], "none")

    def test_plan_fields(self):
        p = self.rb.plan("Why does the sky look blue?")
        self.assertIn(p["level"], LEVELS)
        self.assertIn("extra_tokens", p)

    def test_hard_question_gets_more_budget(self):
        easy = self.rb.plan("hi")
        hard = self.rb.plan("Prove the sum of two even numbers is even in full detail")
        self.assertGreaterEqual(hard["extra_tokens"], easy["extra_tokens"])


if __name__ == "__main__":
    unittest.main()
