"""Merlin-Arthur end-to-end protocol.

merlin -> propose answer
arthur -> verify / score confidence
gate   -> accept / abstain / reject

This module wires the three roles together and exposes the single `answer()`
call used by the engine, the mock backend and the HTTP server. The RL loop in
`train/merlin_rl.py` optimizes the honesty `skill` so that Arthur's confidence
becomes a trustworthy abstention signal.
"""
from .arthur import Arthur
from .honesty import HonestyCalibrator
from .gate import AbstentionGate, GATE_ACCEPT, GATE_ABSTAIN, GATE_REJECT


class MerlinArthur:
    def __init__(self, cfg):
        self.cfg = cfg
        self.arthur = Arthur(seed=cfg.seed)
        self.calibrator = HonestyCalibrator(temperature=1.0, shift=0.0)
        self.gate = AbstentionGate(
            abstain_threshold=cfg.merlin.abstain_threshold,
            reject_threshold=0.2,
        )

    def decide(self, question, answer, skill=1.0, raw_confidence=None):
        """Run the full protocol and return a structured verdict."""
        if raw_confidence is None:
            raw_confidence = self.arthur.confidence(question, answer, skill)
        conf = self.calibrator.calibrate(raw_confidence)
        action = self.gate.decide(conf)
        return {
            "question": question,
            "answer": answer if action == GATE_ACCEPT else ("I don't know" if action == GATE_ABSTAIN else None),
            "raw_confidence": round(raw_confidence, 4),
            "confidence": round(conf, 4),
            "action": action,
        }
