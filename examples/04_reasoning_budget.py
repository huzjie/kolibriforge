"""Show the 4-level reasoning budget dial."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kolibriforge.config import load_config
from kolibriforge.reason.budget import ReasoningBudget

cfg = load_config()
rb = ReasoningBudget(cfg)
for q in ["hi", "What is 2+2?", "Why does the sky look blue?",
          "Prove that the sum of two even numbers is even, in full detail"]:
    p = rb.plan(q)
    print(f"{q[:50]:52s} difficulty={p['difficulty']:.2f} level={p['level']:6s} "
          f"extra_tokens={p['extra_tokens']}")
