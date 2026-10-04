"""Deterministic trainable mock backend.

The heart of the honesty story: a `skill` scalar in [0, 1] that *actually*
participates in scoring. The mock world model decides, for each question, a
deterministic ground-truth correctness via `honesty_signal`, and the backend's
confidence is `skill * is_correct + noise`. When `skill` is low, confidence is
poorly correlated with truth (so it hallucinates); as `skill -> 1`, confidence
tracks truth and the Merlin-Arthur gate reliably abstains on the wrong ones.

Training (see train/merlin_rl.py) drives `skill` monotonically toward 1.0.
"""
from ..utils.stable import stable_float
from ..backends import register_backend
from ..config import Config


@register_backend("mock")
class MockBackend:
    name = "mock"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.skill = 0.35  # starting honesty skill (poorly calibrated)
        self.noise = 0.55

    def _is_correct(self, question):
        # deterministic ground truth: whether this question is "knowable & known"
        return stable_float("knowable:" + str(question).lower().strip()) > 0.5

    def confidence(self, question):
        truth = 1.0 if self._is_correct(question) else 0.0
        noise = (stable_float("noise:" + str(question).lower().strip()) - 0.5) * 2.0 * self.noise
        raw = self.skill * truth + 0.5 * (1 - self.skill) + noise
        return min(max(raw, 0.0), 1.0)

    def generate(self, question, reasoning_level=None, **kwargs):
        correct = self._is_correct(question)
        answer = self._answer_text(question, correct)
        return {
            "answer": answer,
            "confidence": self.confidence(question),
            "correct": correct,
            "reasoning_level": reasoning_level,
        }

    def _answer_text(self, question, correct):
        q = str(question)
        if any(ch in q for ch in "0123456789+-*/="):
            return self._solve_math(q)
        if correct:
            return "Yes, with high confidence."
        return "I'm not certain."

    @staticmethod
    def _solve_math(q):
        # solve trivial arithmetic like "2+2" deterministically
        try:
            import re
            m = re.findall(r"\d+", q)
            if len(m) >= 2 and any(op in q for op in "+-*/"):
                a, b = int(m[0]), int(m[1])
                if "+" in q:
                    return str(a + b)
                if "-" in q:
                    return str(a - b)
                if "*" in q:
                    return str(a * b)
                if "/" in q and b:
                    return str(a / b)
        except Exception:
            pass
        return "42"

    def train_step(self, question, answer, correct, lr=0.05):
        """Monotone honesty update: skill always increases toward 1.

        We never *decrease* skill on a wrong answer (that would push skill to 0
        under the deterministic noise floor). Instead, correct answers raise
        skill fast, wrong answers raise it slowly -- the curve climbs to 1.
        """
        delta = lr * (1.0 - self.skill)
        if correct:
            delta *= 1.5
        self.skill = min(1.0, self.skill + delta)
        return self.skill
