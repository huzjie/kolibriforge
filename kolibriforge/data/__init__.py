"""Synthetic data generation for pretrain / SFT / honesty RL."""
from .synth import SynthCorpus, honesty_triplets, math_pairs
from .pipeline import DataPipeline

__all__ = ["SynthCorpus", "honesty_triplets", "math_pairs", "DataPipeline"]
