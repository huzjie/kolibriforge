"""Run a forward pass through the sparse MoE and inspect routing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kolibriforge.config import load_config
from kolibriforge.model.kolibri import KolibriMoE
from kolibriforge.utils.stable import stable_vector

cfg = load_config()
model = KolibriMoE(cfg)
print(f"total_params={model.total_params}  active_params={model.active_params}  "
      f"sparse_ratio={cfg.moe.sparse_ratio}")

xs = [stable_vector(f"tok:{i}", cfg.hidden) for i in range(8)]
last, aux = model.forward(xs)
print(f"last_hidden_dim={len(last)}  aux_loss={aux:.4f}")
print("routing load distribution:", model.layers[0].bank.activation_counts)
print("routing entropy:", round(model.layers[0].activation_entropy(), 4))
