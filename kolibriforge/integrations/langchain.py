"""LangChain-compatible LLM wrapper (no hard dependency).

If `langchain_core` is installed, this class satisfies its `BaseLLM`-style
`_generate` contract; otherwise it still works as a plain callable so the rest
of the framework never imports LangChain.
"""


class KolibriLLM:
    """Wrap a KolibriEngine as a LangChain LLM."""

    def __init__(self, cfg=None):
        from ..engine import KolibriEngine
        self.engine = KolibriEngine(cfg)

    def _call(self, prompt, stop=None, **kwargs):
        verdict = self.engine.answer(prompt)
        return verdict["answer"] or ""

    def __call__(self, prompt, **kwargs):
        return self._call(prompt, **kwargs)

    @property
    def _identifying_params(self):
        return {"model": self.engine.cfg.backend.model}


def to_langchain(cfg=None):
    """Return a LangChain LLM if langchain_core is available, else the wrapper."""
    llm = KolibriLLM(cfg)
    try:
        from langchain_core.language_models.llms import LLM
        if isinstance(llm, LLM):
            return llm
    except Exception:
        pass
    return llm
