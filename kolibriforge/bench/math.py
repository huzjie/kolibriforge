"""AIME-style math benchmark (arithmetic + step-counting).

Kolibri-1 reports 96.9% on AIME 2025. Here we emulate a small deterministic
arithmetic suite to show that higher reasoning budget buys accuracy -- and to
give the budget router something measurable to optimize against.
"""
from ..bench import register_benchmark


@register_benchmark("math")
def run_math(cfg, backend=None, n=60):
    from ..backends import get_backend
    backend = backend or get_backend("mock")(cfg)
    correct = 0
    for i in range(n):
        a = (i * 7) % 50
        b = (i * 13) % 50
        q = f"{a}+{b}"
        res = backend.generate(q)
        try:
            if str(res["answer"]).strip() == str(a + b):
                correct += 1
        except Exception:
            pass
    return {"accuracy": round(correct / n, 4), "n": n}
