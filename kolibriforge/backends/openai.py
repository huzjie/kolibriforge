"""OpenAI-compatible remote backend (any /v1/chat/completions endpoint).

Works against OpenAI, Azure OpenAI, vLLM, Ollama and any OpenAI-compatible
gateway. Requires `api_base` (and optionally `api_key`) in the config backend
section; uses only the stdlib `urllib` so there is no hard dependency.
"""
import json
import urllib.request

from ..backends import register_backend
from ..config import Config


@register_backend("openai")
class OpenAIBackend:
    name = "openai"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_base = cfg.backend.api_base.rstrip("/")
        self.api_key = cfg.backend.api_key
        self.model = cfg.backend.model
        self.skill = 0.9

    def confidence(self, question):
        return 0.7

    def generate(self, question, reasoning_level=None, **kwargs):
        if not self.api_base:
            raise RuntimeError("openai backend requires backend.api_base in config")
        url = self.api_base + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": question}],
            "temperature": self.cfg.backend.temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        answer = data["choices"][0]["message"]["content"]
        return {"answer": answer, "confidence": 0.7, "correct": None,
                "reasoning_level": reasoning_level}
