"""Synthetic data pipeline tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from kolibriforge.data.synth import honesty_triplets, math_pairs, SynthCorpus
from kolibriforge.data.pipeline import DataPipeline


class TestData(unittest.TestCase):
    def test_honesty_triplets_shape(self):
        ts = honesty_triplets(n=50)
        self.assertEqual(len(ts), 50)
        self.assertIn("correct", ts[0])
        self.assertIsInstance(ts[0]["correct"], bool)

    def test_math_pairs(self):
        pairs = math_pairs(n=10)
        self.assertEqual(len(pairs), 10)

    def test_pipeline_split(self):
        pipe = DataPipeline(honesty_triplets(n=100))
        self.assertEqual(len(pipe.train) + len(pipe.eval), 100)
        self.assertGreater(len(pipe.train), len(pipe.eval))

    def test_corpus_sequence(self):
        c = SynthCorpus(vocab=256)
        seq = c.sequence(0, length=8)
        self.assertEqual(len(seq), 8)


if __name__ == "__main__":
    unittest.main()
