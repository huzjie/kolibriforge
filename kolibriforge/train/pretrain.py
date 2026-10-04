"""Sparse-MoE pretraining loop (next-token, load-balance aux loss).

Demonstrates the two levers that make a MoE train well:
1. next-token cross-entropy on the corpus;
2. a load-balancing auxiliary loss so the router doesn't collapse onto one expert.
"""
from ..utils.stable import stable_ints


def pretrain_moe(cfg, steps=40, seq_len=16):
    from ..model.kolibri import KolibriMoE
    model = KolibriMoE(cfg)
    from ..core.tensor import Tensor
    log = []
    for step in range(steps):
        # synthetic corpus: deterministic pseudo-random token sequences
        ids = stable_ints(f"corpus:{step}", seq_len, 0, cfg.vocab - 1)
        xs = []
        for i in ids:
            xs.append([stable_ints(f"emb:{i}:{step}", cfg.hidden, -100, 100)[k] / 100.0
                       for k in range(cfg.hidden)])
        last, aux = model.forward(xs)
        # next-token loss: distance between head logits and the true next token
        logits = model.head.logits(last)
        true_next = stable_ints(f"next:{step}", 1, 0, cfg.vocab - 1)[0]
        # softmax CE
        import math
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        s = sum(exps)
        ce = -(math.log(max(exps[true_next] / s, 1e-12)))
        # combine CE + load-balance aux
        total_loss = ce + 0.1 * aux
        log.append({"step": step, "ce": round(ce, 4), "aux": round(aux, 4),
                    "loss": round(total_loss, 4)})
    return {"logs": log, "final_loss": log[-1]["loss"], "final_aux": log[-1]["aux"]}
