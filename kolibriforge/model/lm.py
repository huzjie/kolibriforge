"""Language-model head (vocabulary projection)."""
import math


def softmax_head(logits, temperature=1.0):
    scaled = [v / temperature for v in logits]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    s = sum(exps) or 1e-12
    return [e / s for e in exps]


class LMHead:
    """Project hidden states to vocabulary logits (deterministic seed init)."""

    def __init__(self, dim, vocab, seed=0):
        self.dim = dim
        self.vocab = vocab
        self.w = _mat(seed, vocab, dim)

    def logits(self, h):
        return [sum(hi * self.w[i][k] for k, hi in enumerate(h)) for i in range(self.vocab)]

    def next_token(self, h, temperature=1.0):
        probs = softmax_head(self.logits(h), temperature)
        r = (hash(tuple(round(x, 6) for x in h)) % 100000) / 100000.0
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r < cum:
                return i
        return max(range(self.vocab), key=lambda i: probs[i])


def _mat(seed, rows, cols):
    out = []
    for i in range(rows):
        row = []
        for j in range(cols):
            v = ((seed * 1103515245 + i * 137 + j * 97) % 2147483647) / 2147483647 - 0.5
            row.append(v * math.sqrt(2.0 / max(cols, 1)))
        out.append(row)
    return out
