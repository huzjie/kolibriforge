"""CPU backend: run the KolibriMoE model locally on the tensor core."""
from ..backends import register_backend
from ..config import Config
from ..utils.stable import stable_float


@register_backend("cpu")
class CPUBackend:
    name = "cpu"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        from ..model.kolibri import KolibriMoE
        self.model = KolibriMoE(cfg)
        self.skill = 0.8

    def confidence(self, question):
        return stable_float("cpu-conf:" + str(question).lower().strip())

    def generate(self, question, reasoning_level=None, **kwargs):
        # deterministic embedding of the question -> run the model -> sample token
        from ..tokenizer import build_unibpe
        tok = build_unibpe(self.cfg)
        ids = tok.encode(question)
        from ..core.tensor import Tensor
        xs = []
        for i in ids:
            xs.append([stable_float(f"emb:{i}:{j}", -1, 1) for j in range(self.cfg.hidden)])
        last, _aux = self.model.forward(xs)
        tok_id = self.model.head.next_token(last)
        return {
            "answer": tok.decode([tok_id]),
            "confidence": self.confidence(question),
            "correct": None,
            "reasoning_level": reasoning_level,
        }
