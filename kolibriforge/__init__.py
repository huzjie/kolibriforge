"""kolibriforge: sovereign MoE reasoning LLM with honest abstention.

A zero-dependency, fully runnable reference implementation of the techniques
behind Aleph Alpha's Kolibri-1 (released 2026-10-03), a 78B open-weight
Mixture-of-Experts model with a 1M-token context. This repo distills its four
most reusable engineering ideas into a single, self-contained framework:

1. Sparse MoE          -- 78.1B total params / ~3.46B active per token
2. UniBPE tokenizer    -- compound-word-aware byte-pair encoding (de/en)
3. Merlin-Arthur       -- honest "I don't know" abstention to prevent hallucination
4. Reasoning budget    -- 4 inference-time depth levels (none/low/medium/high)

Highlights
----------
- Zero-dependency tensor core (pure Python, autodiff over the small op set)
- Deterministic trainable mock backend + CPU / OpenAI / vLLM / HF backends
- Merlin-Arthur RL training: reward = correctness - lambda*hallucination penalty
- Benchmarks: honesty, abstention rate, AIME-style math, long-context retrieval,
  and tokenizer compression ratio
- OpenAI-compatible stdlib HTTP server + CLI + Docker/K8s/Helm/CI
"""
from .version import __version__
from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
