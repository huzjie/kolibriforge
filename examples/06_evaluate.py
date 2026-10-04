"""Run all registered benchmarks and print the results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kolibriforge.config import load_config
from kolibriforge.bench.runner import run_all

cfg = load_config()
results = run_all(cfg)
print("\n=== Benchmark summary ===")
for k, v in results.items():
    print(f"  {k:14s} {v}")
