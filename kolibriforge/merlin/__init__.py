"""Merlin-Arthur honest-abstention protocol.

Named after the interactive proof system: Merlin proposes, Arthur verifies. Here
"Merlin" is the generator (the model that wants to answer), "Arthur" is the
verifier (the model that checks whether the answer is trustworthy). When Arthur
cannot vouch for an answer, the system abstains -- it says "I don't know" --
instead of hallucinating.
"""
from .arthur import Arthur
from .honesty import HonestyCalibrator, calibrate_confidence
from .gate import AbstentionGate, GATE_ACCEPT, GATE_ABSTAIN, GATE_REJECT
from .protocol import MerlinArthur

__all__ = [
    "Arthur", "HonestyCalibrator", "calibrate_confidence",
    "AbstentionGate", "GATE_ACCEPT", "GATE_ABSTAIN", "GATE_REJECT",
    "MerlinArthur",
]
