"""Generate and split synthetic honesty data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kolibriforge.data.synth import honesty_triplets, math_pairs
from kolibriforge.data.pipeline import DataPipeline

triplets = honesty_triplets(n=120)
pipe = DataPipeline(triplets)
print(f"honesty triplets: train={len(pipe.train)} eval={len(pipe.eval)}")
print("sample:", pipe.train[0])
print("math pairs:", math_pairs(3))
