"""Demonstrate Merlin-Arthur: honest abstention instead of hallucination."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kolibriforge.config import load_config
from kolibriforge.engine import KolibriEngine

cfg = load_config()
engine = KolibriEngine(cfg)

print("Before training (low honesty skill) ...")
for q in ["What is the capital of France?", "What is 2+2?", "explain quantum gravity"]:
    v = engine.answer(q)
    print(f"  {q:40s} -> action={v['action']:8s} answer={v['answer']}")

print("\nTraining honesty skill -> 1.0 ...")
from kolibriforge.backends import get_backend
backend = get_backend("mock")(cfg)
for i in range(80):
    backend.train_step(f"q{i}", "x", True, lr=0.1)
engine._backend = backend
print(f"  skill now = {backend.skill:.3f}")

print("\nAfter training ...")
for q in ["What is the capital of France?", "What is 2+2?", "explain quantum gravity"]:
    v = engine.answer(q)
    print(f"  {q:40s} -> action={v['action']:8s} answer={v['answer']}")
