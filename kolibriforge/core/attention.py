"""Scaled dot-product attention over plain Python lists (mock-friendly)."""
import math


def _softmax_row(vals):
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps) or 1e-12
    return [e / s for e in exps]


def scaled_dot_product_attention(query, keys, values, scale=None, temperature=1.0):
    """query: list[d], keys: list[list[d]], values: list[list[dv]].

    Returns (output list[dv], attention_weights list[float]).
    """
    dk = len(query)
    scale = scale if scale is not None else (1.0 / math.sqrt(dk))
    scores = []
    for k in keys:
        scores.append(sum(q * kk for q, kk in zip(query, k)) * scale / temperature)
    weights = _softmax_row(scores)
    dv = len(values[0]) if values else 0
    out = [0.0] * dv
    for w, v in zip(weights, values):
        for j in range(dv):
            out[j] += w * v[j]
    return out, weights
