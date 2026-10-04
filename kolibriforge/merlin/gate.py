"""Abstention gate: decide accept / abstain / reject.

The gate turns a calibrated confidence into one of three actions:

- ACCEPT  : confidence >= high threshold -> answer confidently.
- ABSTAIN : low <= confidence < high -> say "I don't know" (honest abstention).
- REJECT  : confidence < low -> refuse (safety-relevant, out-of-scope).

The key anti-hallucination insight: a *wrong* answer is far more costly than a
*none* answer. So we'd rather abstain than guess.
"""

GATE_ACCEPT = "accept"
GATE_ABSTAIN = "abstain"
GATE_REJECT = "reject"


class AbstentionGate:
    def __init__(self, abstain_threshold=0.5, reject_threshold=0.2):
        self.abstain_threshold = abstain_threshold
        self.reject_threshold = reject_threshold

    def decide(self, confidence):
        if confidence >= self.abstain_threshold:
            return GATE_ACCEPT
        if confidence >= self.reject_threshold:
            return GATE_ABSTAIN
        return GATE_REJECT

    def should_answer(self, confidence):
        return self.decide(confidence) == GATE_ACCEPT
