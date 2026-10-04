"""SwiGLU experts and the expert bank (the '3.46B active' part)."""
import math


class SwiGLUExpert:
    """A single expert: a SwiGLU feed-forward block with a fixed seed init.

    swish(x) = x * sigmoid(x); the gated FFN is W2 @ (swish(W1 @ x) * (W3 @ x)).
    """

    def __init__(self, dim, hidden, seed=0):
        self.dim = dim
        self.hidden = hidden
        self.w1 = _mat(seed, hidden, dim)
        self.w3 = _mat(seed + 1, hidden, dim)
        self.w2 = _mat(seed + 2, dim, hidden)

    def __call__(self, x):
        # W1/W3 are (hidden x dim): h1 = W1 @ x, h3 = W3 @ x
        h1 = [sum(x[k] * self.w1[j][k] for k in range(self.dim)) for j in range(self.hidden)]
        h3 = [sum(x[k] * self.w3[j][k] for k in range(self.dim)) for j in range(self.hidden)]
        h = [h1[j] * _swish(h1[j]) * h3[j] for j in range(self.hidden)]
        # W2 is (dim x hidden): out = W2 @ h
        out = [sum(h[k] * self.w2[i][k] for k in range(self.hidden)) for i in range(self.dim)]
        return out


class ExpertBank:
    """Holds all experts; only the top-K routed experts run per token."""

    def __init__(self, n_experts, dim, hidden, seed=0):
        self.n_experts = n_experts
        self.dim = dim
        self.hidden = hidden
        self.experts = [SwiGLUExpert(dim, hidden, seed=seed + i) for i in range(n_experts)]
        # per-expert activation counters (for load-balance telemetry)
        self.activation_counts = [0] * n_experts

    def dispatch(self, xs, selections):
        """Run selected experts per token. xs: list of token vectors,
        selections: list of (indices, weights). Returns weighted outputs."""
        outs = []
        for x, (sel, weights) in zip(xs, selections):
            mixed = [0.0] * self.dim
            for idx, w in zip(sel, weights):
                self.activation_counts[idx] += 1
                e_out = self.experts[idx](x)
                for d in range(self.dim):
                    mixed[d] += w * e_out[d]
            outs.append(mixed)
        return outs

    def reset_counts(self):
        self.activation_counts = [0] * self.n_experts


def _mat(seed, rows, cols):
    out = []
    for i in range(rows):
        row = []
        for j in range(cols):
            v = ((seed * 1103515245 + i * 137 + j * 97) % 2147483647) / 2147483647 - 0.5
            row.append(v * math.sqrt(2.0 / max(cols, 1)))
        out.append(row)
    return out


def _swish(x):
    return x / (1 + math.exp(-x))
