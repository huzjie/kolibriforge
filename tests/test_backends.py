"""Backend registry + trainable mock tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from kolibriforge.config import load_config
from kolibriforge.backends import list_backends, get_backend


class TestBackends(unittest.TestCase):
    def test_registry_populated(self):
        names = list_backends()
        self.assertIn("mock", names)
        self.assertIn("cpu", names)
        self.assertIn("openai", names)
        self.assertIn("vllm", names)
        self.assertIn("transformers", names)

    def test_mock_trainable(self):
        cfg = load_config()
        b = get_backend("mock")(cfg)
        s0 = b.skill
        for _ in range(50):
            b.train_step("q", "a", True, lr=0.1)
        self.assertGreater(b.skill, s0)
        self.assertLessEqual(b.skill, 1.0)


if __name__ == "__main__":
    unittest.main()
