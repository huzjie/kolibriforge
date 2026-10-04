"""A tiny, pure-Python tensor library (1D/2D) for the mock/CPU backends.

Zero external dependencies. Reverse-mode autodiff over the small op set the
scoring / MoE / Merlin heads actually use. Deliberately compact: the point is to
make the sparse-MoE and honesty math explicit and to let `train` run anywhere.
"""
import math
import random


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        if isinstance(data, Tensor):
            data = data.data
        if isinstance(data, (int, float)):
            data = [[float(data)]]
        elif data and not isinstance(data[0], (list, tuple)):
            data = [list(data)]
        self.data = [list(map(float, row)) for row in data]
        self.shape = (len(self.data), len(self.data[0]) if self.data else 0)
        self.requires_grad = requires_grad
        self.grad = None
        self._children = _children
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(shape={self.shape})"

    @staticmethod
    def _broadcast(a, b):
        if a.shape == b.shape:
            return a.data, b.data
        if a.shape[1] == b.shape[1] and b.shape[0] == 1:
            return a.data, b.data * a.shape[0]
        if b.shape[1] == a.shape[1] and a.shape[0] == 1:
            return a.data * b.shape[0], b.data
        if a.shape == (1, 1):
            return [[a.data[0][0]] * b.shape[1] for _ in range(b.shape[0])], b.data
        if b.shape == (1, 1):
            return a.data, [[b.data[0][0]] * a.shape[1] for _ in range(a.shape[0])]
        raise ValueError(f"broadcast mismatch {a.shape} vs {b.shape}")

    def _binary(self, other, fn, op):
        other = other if isinstance(other, Tensor) else Tensor([[float(other)]])
        lhs, rhs = self._broadcast(self, other)
        out = [[fn(x, y) for x, y in zip(rx, ry)] for rx, ry in zip(lhs, rhs)]
        return Tensor(out, self.requires_grad or other.requires_grad, (self, other), op)

    def __add__(self, other):
        return self._binary(other, lambda x, y: x + y, "+")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary(other, lambda x, y: x - y, "-")

    def __mul__(self, other):
        return self._binary(other, lambda x, y: x * y, "*")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary(other, lambda x, y: x / y, "/")

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        a, b = self.data, other.data
        if self.shape[1] != other.shape[0]:
            raise ValueError(f"matmul shape {self.shape} @ {other.shape}")
        out = [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
               for i in range(len(a))]
        return Tensor(out, self.requires_grad or other.requires_grad, (self, other), "matmul")

    def sum(self):
        return Tensor([[sum(sum(row) for row in self.data)]])

    def mean(self):
        n = self.shape[0] * self.shape[1]
        return Tensor([[sum(sum(row) for row in self.data) / n]])

    def reshape(self, *shape):
        flat = [x for row in self.data for x in row]
        if len(shape) == 2 and shape[0] * shape[1] == len(flat):
            rows, cols = shape
            return Tensor([flat[i * cols:(i + 1) * cols] for i in range(rows)])
        return Tensor(self.data)

    def tolist(self):
        return [list(row) for row in self.data]

    def item(self):
        return self.data[0][0]

    def backward(self):
        topo, visited = [], set()

        def build(t):
            if t not in visited:
                visited.add(t)
                for c in t._children:
                    build(c)
                topo.append(t)

        build(self)
        self.grad = [[1.0] * self.shape[1] for _ in range(self.shape[0])]
        for t in reversed(topo):
            t._backward()


def tensor(data, requires_grad=False):
    return Tensor(data, requires_grad=requires_grad)


def zeros(rows, cols):
    return Tensor([[0.0] * cols for _ in range(rows)])


def ones(rows, cols):
    return Tensor([[1.0] * cols for _ in range(rows)])


def randn(rows, cols, seed=0):
    r = random.Random(seed)
    return Tensor([[r.gauss(0, 1) for _ in range(cols)] for _ in range(rows)])
