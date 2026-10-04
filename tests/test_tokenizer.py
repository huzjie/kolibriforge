"""UniBPE tokenizer tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from kolibriforge.config import load_config
from kolibriforge.tokenizer import build_unibpe


class TestUniBPE(unittest.TestCase):
    def setUp(self):
        self.tok = build_unibpe(load_config())

    def test_compound_annotation(self):
        info = self.tok.tokenize_with_info("Donaudampfschifffahrtsgesellschaft")
        self.assertLess(info["num_tokens"], info["num_chars"])

    def test_reversible(self):
        text = "Kraftfahrzeughaftpflichtversicherung"
        ids = self.tok.encode(text)
        decoded = self.tok.decode(ids)
        self.assertEqual(decoded, text)

    def test_info_shape(self):
        info = self.tok.tokenize_with_info("hello")
        for k in ("raw", "tokens", "num_chars", "num_tokens", "compression_ratio"):
            self.assertIn(k, info)


if __name__ == "__main__":
    unittest.main()
