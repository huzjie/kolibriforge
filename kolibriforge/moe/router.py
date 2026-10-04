"""Top-K router for sparse expert selection.

For each token embedding, compute a softmax over expert logits and keep the top-K
experts. Kolibri-1 activates only ~3.46B of 78.1B parameters per token -- the
router is the piece that decides *which* ~4% runs.
"""


def _softmax_row(vals):
    m = max(vals)
    exps = [__import__("math").exp(v - m) for v in vals]
    s = sum(exps) or 1e-12
    return [e / s for e in exps]


class TopKRouter:
    def __init__(self, n_experts, top_k, dim, seed=0):
        self.n_experts = n_experts
        self.top_k = top_k
        self.dim = dim
        self.w = [[((seed * 1103515245 + i * 137 + j * 97) % 2147483647) / 2147483647 - 0.5
                   for j in range(dim)] for i in range(n_experts)]

    def logits(self, x):
        return [sum(xi * wj for xi, wj in zip(x, self.w[i])) for i in range(self.n_experts)]

    def route(self, x, top_k=None):
        """Return (selected_indices, gating_weights) for a single token vector x."""
        top_k = top_k or self.top_k
        logits = self.logits(x)
        probs = _softmax_row(logits)
        order = sorted(range(self.n_experts), key=lambda i: logits[i], reverse=True)
        selected = order[:top_k]
        weights = [probs[i] for i in selected]
        s = sum(weights) or 1e-12
        weights = [w / s for w in weights]
        return selected, weights

    def route_batch(self, xs, top_k=None):
        """Batch routing: list of token vectors -> per-token (selected, weights)."""
        return [self.route(x, top_k=top_k) for x in xs]
