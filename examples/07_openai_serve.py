"""Start the OpenAI-compatible HTTP server."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kolibriforge.config import load_config
from kolibriforge.serving.server import serve

cfg = load_config()
serve(cfg, host="127.0.0.1", port=8000)
