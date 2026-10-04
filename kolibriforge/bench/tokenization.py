"""Tokenizer benchmark: tokens-per-word on German compound-heavy text.

The whole point of UniBPE is fewer tokens on Germanic compound words. This
benchmark compares the compound-aware path against a naive char split, so the
compression gain is visible.
"""
from ..bench import register_benchmark


@register_benchmark("tokenization")
def run_tokenization(cfg, backend=None, samples=None):
    from ..tokenizer import build_unibpe
    tok = build_unibpe(cfg)
    samples = samples or [
        "Donaudampfschifffahrtsgesellschaft",
        "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz",
        "Kraftfahrzeughaftpflichtversicherung",
        "Rechtsschutzversicherungsgesellschaften",
        "Arbeitsunfähigkeitsbescheinigung",
    ]
    total_chars = 0
    total_tokens = 0
    total_naive = 0
    per = []
    for s in samples:
        info = tok.tokenize_with_info(s)
        total_chars += len(s)
        total_tokens += info["num_tokens"]
        total_naive += len(s)
        per.append({"word": s[:24] + ("..." if len(s) > 24 else ""),
                    "chars": len(s), "tokens": info["num_tokens"],
                    "ratio": info["compression_ratio"]})
    return {
        "chars": total_chars,
        "compound_tokens": total_tokens,
        "naive_char_tokens": total_naive,
        "compression_ratio": round(total_chars / max(total_tokens, 1), 3),
        "per_word": per,
    }
