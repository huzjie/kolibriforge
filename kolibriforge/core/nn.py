"""Small neural building blocks over the tensor core."""
import math

from .tensor import Tensor


def _kaiming(fan_in, seed, i, j):
    v = (seed * 1103515245 + i * 137 + j * 97) % 2147483647
    return (v / 2147483647 - 0.5) * math.sqrt(2.0 / max(fan_in, 1))


class Linear:
    def __init__(self, in_dim, out_dim, seed=0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = [[_kaiming(in_dim, seed, i, j) for j in range(in_dim)] for i in range(out_dim)]
        self.bias = [0.0] * out_dim

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = []
        for row in x.data:
            out.append([sum(row[k] * self.weight[i][k] for k in range(self.in_dim)) + self.bias[i]
                        for i in range(self.out_dim)])
        return Tensor(out)


class Embedding:
    def __init__(self, vocab, dim, seed=0):
        self.vocab = vocab
        self.dim = dim
        self.table = [[_kaiming(vocab, seed, i, j) for j in range(dim)] for i in range(vocab)]

    def __call__(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return Tensor([self.table[i % self.vocab] for i in ids])
