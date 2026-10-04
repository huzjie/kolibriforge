"""Deterministic synthetic data generators.

Three datasets feed the three training paths:

- `SynthCorpus`   -- token sequences for MoE pretraining (next-token + load-balance)
- `honesty_triplets` -- (question, answer, correct) triples for Merlin-Arthur RL/SFT
- `math_pairs`    -- arithmetic Q/A for the AIME-style math benchmark

All generators are seeded via `utils.stable` so the datasets are reproducible.
"""
from ..utils.stable import stable_ints, stable_float, stable_choice


class SynthCorpus:
    def __init__(self, vocab, seed=0):
        self.vocab = vocab
        self.seed = seed

    def sequence(self, idx, length=16):
        return stable_ints(f"corpus:{self.seed}:{idx}", length, 0, self.vocab - 1)

    def batch(self, n, length=16):
        return [self.sequence(i, length) for i in range(n)]


def honesty_triplets(n=200, seed=0):
    """Yield (question, answer, correct) triples where correctness is the
    deterministic `knowable` signal shared with the mock world model."""
    out = []
    for i in range(n):
        q = f"honesty-question-{seed}-{i}"
        correct = stable_float(f"knowable:{q}") > 0.5
        answer = "Yes, with high confidence." if correct else "I'm not certain."
        out.append({"question": q, "answer": answer, "correct": correct})
    return out


def math_pairs(n=60, seed=0):
    """Deterministic arithmetic pairs `a+b` with the known answer."""
    out = []
    for i in range(n):
        a = (i * 7) % 50
        b = (i * 13) % 50
        out.append({"question": f"{a}+{b}", "answer": str(a + b)})
    return out
