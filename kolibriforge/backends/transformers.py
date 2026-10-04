"""Hugging Face transformers backend (optional dependency).

Only imported when selected; if `transformers` is not installed, it raises a
helpful error instead of crashing the import of the registry.
"""
from ..backends import register_backend
from ..config import Config


@register_backend("transformers")
class TransformersBackend:
    name = "transformers"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        try:
            import transformers  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "transformers backend requires `pip install transformers`") from e
        self.skill = 0.9
        self._model = None

    def confidence(self, question):
        return 0.7

    def generate(self, question, reasoning_level=None, **kwargs):
        raise NotImplementedError(
            "transformers backend is a thin adapter; load a real model with "
            "`AutoModelForCausalLM` in your own script or use the vllm/openai backend.")
