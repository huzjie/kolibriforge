"""KolibriMoE: the full sparse-MoE transformer stack.

Emulates the 78.1B-total / ~3.46B-active sparse structure in miniature: an
embedding, `n_layers` MoE transformer blocks, and an LM head. The sparse ratio
(how many parameters are "active" per token) is surfaced explicitly so the
demo and the docs can state it honestly.
"""
from ..moe.layer import MoELayer
from .lm import LMHead


class KolibriMoE:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dim = cfg.hidden
        self.vocab = cfg.vocab
        self.n_layers = 2
        self.layers = [
            MoELayer(cfg.moe.num_experts, cfg.moe.top_k, cfg.hidden, cfg.moe.expert_dim,
                     seed=cfg.seed + i) for i in range(self.n_layers)
        ]
        self.head = LMHead(cfg.hidden, cfg.vocab, seed=cfg.seed + 1000)
        # rough parameter-count emulation for the "78B / 3.46B" framing
        self.total_params = self._count_params()
        self.active_params = int(self.total_params / cfg.moe.sparse_ratio)

    def _count_params(self):
        n = 0
        for layer in self.layers:
            for expert in layer.bank.experts:
                n += expert.dim * expert.hidden * 3  # w1, w3, w2 (approx)
            n += layer.router.n_experts * layer.router.dim
            n += self.dim * self.dim * 4  # attention q/k/v/o
        n += self.dim * self.vocab  # LM head
        n += self.vocab * self.dim  # embedding
        return n

    def forward(self, xs):
        """xs: list of token vectors (seq, dim). Returns (last_hidden, aux_losses)."""
        auxes = []
        h = xs
        for layer in self.layers:
            h, aux = layer.forward(h)
            auxes.append(aux)
        last = h[-1] if h else [0.0] * self.dim
        return last, sum(auxes)
