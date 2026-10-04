"""Train the honesty policy with Merlin-Arthur RL and watch hallucination drop."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kolibriforge.config import load_config
from kolibriforge.train.merlin_rl import MerlinRLTrainer

cfg = load_config()
t = MerlinRLTrainer(cfg)
result = t.run(steps=80)
print(result)
