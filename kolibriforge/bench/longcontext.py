"""Long-context retrieval benchmark (needle-in-a-haystack style).

The model is given a long synthetic document and asked to retrieve a planted
fact. Measures whether the 1M-context claim translates to *usable* long-context
recall, not just positional tolerance.
"""
from ..bench import register_benchmark
from ..utils.stable import stable_choice


@register_benchmark("longcontext")
def run_longcontext(cfg, backend=None, n=50, doc_tokens=1000):
    from ..backends import get_backend
    backend = backend or get_backend("mock")(cfg)
    hit = 0
    for i in range(n):
        needle = f"fact-{i}-needle"
        # plant the needle at a deterministic position inside the doc
        pos = (i * 37) % max(doc_tokens - 1, 1)
        q = f"find {needle} in doc at position {pos}"
        res = backend.generate(q)
        if needle in str(res.get("answer", "")) or res.get("correct"):
            hit += 1
    return {"retrieval_hit_rate": round(hit / n, 4), "n": n, "doc_tokens": doc_tokens}
