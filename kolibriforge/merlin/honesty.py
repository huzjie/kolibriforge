"""Honesty calibration utilities.

A model's confidence should be a *reliable* estimate of its correctness. Two
common failure modes are calibrated away here:

1. Over-confidence -- high confidence on wrong answers (the classic hallucination
   source). Fixed with temperature scaling on the confidence logit.
2. Under-confidence -- low confidence on correct answers (wastes the abstention
   budget). Fixed by shifting the confidence up when the answer is provably
   verifiable.
"""
import math


class HonestyCalibrator:
    def __init__(self, temperature=1.0, shift=0.0):
        self.temperature = temperature
        self.shift = shift

    def calibrate(self, raw_confidence):
        """Apply temperature scaling + shift, clamped to [0, 1]."""
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        logit = math.log(max(raw_confidence, 1e-9) / max(1 - raw_confidence, 1e-9))
        logit = logit / self.temperature + self.shift
        return 1.0 / (1.0 + math.exp(-logit))


def calibrate_confidence(raw_confidence, temperature=1.0, shift=0.0):
    return HonestyCalibrator(temperature, shift).calibrate(raw_confidence)


def expected_calibration_error(confidences, corrects, n_bins=10):
    """Expected Calibration Error (ECE). Lower = better calibrated."""
    if not confidences:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for c, corr in zip(confidences, corrects):
        b = min(int(c * n_bins), n_bins - 1)
        bins[b].append((c, corr))
    ece = 0.0
    n = len(confidences)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(1 for _, corr in b if corr) / len(b)
        ece += (len(b) / n) * abs(avg_conf - avg_acc)
    return ece
