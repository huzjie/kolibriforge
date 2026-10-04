"""Difficulty router: a lightweight classifier over the 4 depth levels.

This is the trainable counterpart to the heuristic `ReasoningBudget`: a tiny
logistic router that predicts the cheapest depth level that still solves the
task. Training it trades off a small accuracy penalty against token savings --
exactly the "reasoning budget" dial Kolibri-1 exposes at inference time.
"""
import math


class DifficultyRouter:
    def __init__(self, dim=8, seed=0):
        self.dim = dim
        self.w = [((seed * 1103515245 + i * 137) % 2147483647) / 2147483647 - 0.5
                  for i in range(dim)]
        self.b = 0.0

    def _feats(self, query):
        q = str(query).lower()
        f = [0.0] * self.dim
        f[0] = min(len(q) / 120.0, 1.0)
        f[1] = min(sum(q.count(w) for w in ("why", "how", "prove", "explain")) / 4.0, 1.0)
        f[2] = 1.0 if any(ch in q for ch in "0123456789+-*/=") else 0.0
        f[3] = 1.0 if q.endswith("?") else 0.0
        for i in range(4, self.dim):
            f[i] = (hash((q, i)) % 1000) / 1000.0
        return f

    def score(self, query):
        f = self._feats(query)
        return 1.0 / (1.0 + math.exp(-(sum(fi * wi for fi, wi in zip(f, self.w)) + self.b)))
