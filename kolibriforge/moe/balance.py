"""Load-balancing auxiliary loss.

MoE routers tend to collapse onto a handful of experts (routing collapse). The
standard fix is an auxiliary loss that rewards uniform expert usage. This is
important for throughput: if a few experts soak up all tokens, you lose the
sparse-compute benefit and hit hot-spot latency.
"""


def load_balance_loss(activation_counts, n_tokens):
    """Return a scalar in [0, +inf); lower is more balanced.

    Uses the squared coefficient of variation of the per-expert load fraction.
    Perfectly uniform load -> 0.0.
    """
    n = len(activation_counts)
    if n_tokens <= 0 or n == 0:
        return 0.0
    mean = n_tokens / n
    var = sum((c - mean) ** 2 for c in activation_counts) / n
    return var / (mean ** 2 + 1e-9)


def routing_entropy(gating_weights):
    """Shannon entropy of the gate distribution (higher = less collapsed)."""
    import math
    e = 0.0
    for p in gating_weights:
        if p > 0:
            e -= p * math.log(p)
    return e
