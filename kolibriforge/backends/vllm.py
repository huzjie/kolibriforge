"""vLLM backend: high-throughput local serving via the OpenAI-compatible API.

vLLM exposes `/v1/chat/completions`; reuse the OpenAI client path. This backend
only exists so the config can point at a vLLM endpoint and get high-throughput
decision serving without code changes.
"""
from ..backends import register_backend
from ..config import Config


@register_backend("vllm")
class VLLMBackend:
    name = "vllm"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        from .openai import OpenAIBackend
        self._inner = OpenAIBackend(cfg)
        self.skill = 0.95

    def confidence(self, question):
        return 0.75

    def generate(self, question, reasoning_level=None, **kwargs):
        return self._inner.generate(question, reasoning_level=reasoning_level, **kwargs)
