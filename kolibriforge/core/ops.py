"""Functional ops over Tensor."""
import math

from .tensor import Tensor


def softmax(x, axis=-1, temperature=1.0):
    x = x if isinstance(x, Tensor) else Tensor(x)
    out = []
    for row in x.data:
        scaled = [v / temperature for v in row]
        m = max(scaled)
        exps = [math.exp(v - m) for v in scaled]
        s = sum(exps) or 1e-12
        out.append([e / s for e in exps])
    return Tensor(out)


def relu(x):
    x = x if isinstance(x, Tensor) else Tensor(x)
    return Tensor([[max(0.0, v) for v in row] for row in x.data])


def gelu(x):
    x = x if isinstance(x, Tensor) else Tensor(x)
    return Tensor([[0.5 * v * (1 + math.erf(v / math.sqrt(2))) for v in row] for row in x.data])


def silu(x):
    x = x if isinstance(x, Tensor) else Tensor(x)
    return Tensor([[v / (1 + math.exp(-v)) for v in row] for row in x.data])


def layernorm(x, eps=1e-5):
    x = x if isinstance(x, Tensor) else Tensor(x)
    out = []
    for row in x.data:
        mu = sum(row) / len(row)
        var = sum((v - mu) ** 2 for v in row) / len(row)
        inv = 1.0 / math.sqrt(var + eps)
        out.append([(v - mu) * inv for v in row])
    return Tensor(out)


def cross_entropy(logits, target_idx, temperature=1.0):
    probs = softmax(logits, temperature=temperature)
    p = probs.data[0][target_idx]
    return -math.log(max(p, 1e-12))


def accuracy(pred_idx, target_idx):
    return 1.0 if pred_idx == target_idx else 0.0
