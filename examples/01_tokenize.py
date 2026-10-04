"""Run the UniBPE compound-aware tokenizer on German compound words."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kolibriforge.config import load_config
from kolibriforge.tokenizer import build_unibpe

cfg = load_config()
tok = build_unibpe(cfg)
for w in [
    "Donaudampfschifffahrtsgesellschaft",
    "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz",
    "Kraftfahrzeughaftpflichtversicherung",
]:
    info = tok.tokenize_with_info(w)
    print(f"{w[:40]:42s} chars={info['num_chars']:3d} tokens={info['num_tokens']:3d} "
          f"ratio={info['compression_ratio']}")
