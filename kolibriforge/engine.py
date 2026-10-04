"""KolibriEngine: the top-level orchestration facade.

One object wires tokenizer + model + backends + Merlin-Arthur + reasoning budget
into the three operations the CLI / server / benchmarks all call:

- answer()   -- generate with honesty gating (the anti-hallucination path)
- abstain()  -- run only the Merlin-Arthur gate on a question
- tokenize() -- UniBPE compound-aware tokenization
"""
from .config import Config, load_config
from .utils.stable import stable_float


class KolibriEngine:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or load_config()
        self._backend = None

    @property
    def backend(self):
        if self._backend is None:
            from .backends import get_backend
            self._backend = get_backend(self.cfg.backend.name)(self.cfg)
        return self._backend

    def answer(self, question, reasoning_level=None):
        """Answer with the Merlin-Arthur honesty gate applied."""
        from .merlin.protocol import MerlinArthur
        ma = MerlinArthur(self.cfg)
        raw = self.backend.generate(question, reasoning_level=reasoning_level)
        return ma.decide(question, raw["answer"], skill=self.backend.skill,
                         raw_confidence=raw.get("confidence"))

    def abstain(self, question):
        """Gate-only: decide whether to answer at all."""
        from .merlin.protocol import MerlinArthur
        ma = MerlinArthur(self.cfg)
        # use the backend's confidence directly (no generation needed)
        conf = self.backend.confidence(question)
        return ma.decide(question, None, skill=self.backend.skill, raw_confidence=conf)

    def tokenize(self, text):
        from .tokenizer import build_unibpe
        return build_unibpe(self.cfg).tokenize_with_info(text)

    def route(self, query):
        from .reason.budget import ReasoningBudget
        return ReasoningBudget(self.cfg).plan(query)

    def honesty_signal(self, question):
        """Deterministic "is this knowable" signal shared by mock world model.

        This single source of truth keeps the mock backend, the benchmark scorer
        and the Arthur verifier consistent -- the same reason previous forks keep
        a `world_target` / `world_answer` helper in one place.
        """
        return stable_float("knowable:" + str(question).lower().strip())
