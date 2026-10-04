"""A full MoE transformer block: attention + sparse MoE FFN + residual."""
import math

from .router import TopKRouter
from .experts import ExpertBank
from .balance import load_balance_loss


def _layernorm_row(row):
    mu = sum(row) / len(row)
    var = sum((v - mu) ** 2 for v in row) / len(row)
    inv = 1.0 / math.sqrt(var + 1e-5)
    return [(v - mu) * inv for v in row]


def _softmax_row(vals):
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps) or 1e-12
    return [e / s for e in exps]


class MoELayer:
    """Attention (single head, no causal mask for mock speed) + MoE FFN."""

    def __init__(self, n_experts, top_k, dim, expert_hidden, seed=0):
        self.dim = dim
        self.router = TopKRouter(n_experts, top_k, dim, seed=seed)
        self.bank = ExpertBank(n_experts, dim, expert_hidden, seed=seed + 100)
        # attention Q/K/V projections (dim x dim)
        self.wq = _mat(seed + 200, dim, dim)
        self.wk = _mat(seed + 201, dim, dim)
        self.wv = _mat(seed + 202, dim, dim)
        self.wo = _mat(seed + 203, dim, dim)

    def forward(self, xs):
        """xs: list of token vectors (seq, dim). Returns (outputs, aux_loss)."""
        n = len(xs)
        if n == 0:
            return [], 0.0
        # attention
        qs = [_mv(self.wq, x) for x in xs]
        ks = [_mv(self.wk, x) for x in xs]
        vs = [_mv(self.wv, x) for x in xs]
        attn_out = []
        scale = 1.0 / math.sqrt(self.dim)
        for q in qs:
            scores = [sum(qi * ki for qi, ki in zip(q, k)) * scale for k in ks]
            w = _softmax_row(scores)
            ctx = [0.0] * self.dim
            for wi, v in zip(w, vs):
                for d in range(self.dim):
                    ctx[d] += wi * v[d]
            attn_out.append(_mv(self.wo, ctx))
        # residual + norm
        h = [_layernorm_row([xs[i][d] + attn_out[i][d] for d in range(self.dim)])
             for i in range(n)]
        # MoE
        self.bank.reset_counts()
        selections = self.router.route_batch(h)
        moe_out = self.bank.dispatch(h, selections)
        # residual
        out = [[h[i][d] + moe_out[i][d] for d in range(self.dim)] for i in range(n)]
        aux = load_balance_loss(self.bank.activation_counts, n)
        return out, aux

    def activation_entropy(self):
        total = sum(self.bank.activation_counts) or 1
        import math
        e = 0.0
        for c in self.bank.activation_counts:
            p = c / total
            if p > 0:
                e -= p * math.log(p)
        return e


def _mat(seed, rows, cols):
    out = []
    for i in range(rows):
        row = []
        for j in range(cols):
            v = ((seed * 1103515245 + i * 137 + j * 97) % 2147483647) / 2147483647 - 0.5
            row.append(v * math.sqrt(2.0 / max(cols, 1)))
        out.append(row)
    return out


def _mv(m, x):
    return [sum(xi * m[i][j] for j, xi in enumerate(x)) for i in range(len(m))]
