"""Arthur: the verifier role.

Arthur is a lightweight binary classifier that predicts whether Merlin's proposed
answer is correct, using a deterministic feature vector (the same "does this
look knowable?" signal the mock world model uses). In a real system this is a
second forward pass or a cross-encoder; here it is a small calibrated scorer so
the whole loop is runnable end-to-end with zero dependencies.
"""
import math


class Arthur:
    """Verifier producing a confidence score in [0, 1] for an answer."""

    def __init__(self, seed=0):
        self.seed = seed
        self.w = [(seed * 1103515245 + i * 137) % 2147483647 / 2147483647 - 0.5
                  for i in range(4)]

    def confidence(self, question, answer, calibrated_skill=1.0):
        """Confidence = sigmoid(feature dot w * skill). skill encodes how
        well-calibrated the model is; as skill -> 1, confidence aligns with truth.
        """
        feats = self._features(question, answer)
        logit = sum(f * wi for f, wi in zip(feats, self.w)) * calibrated_skill
        return 1.0 / (1.0 + math.exp(-logit))

    @staticmethod
    def _features(question, answer):
        q = str(question).lower()
        a = str(answer).lower()
        # heuristics: answer length, presence of hedge words, question length
        hedge = any(w in a for w in ("maybe", "perhaps", "i think", "unsure", "könnte", "vielleicht"))
        return [
            float(len(a)) / 100.0,
            float(len(q)) / 100.0,
            1.0 if hedge else 0.0,
            1.0 if a.strip() else 0.0,
        ]
