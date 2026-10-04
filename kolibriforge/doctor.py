"""Component / backend / benchmark smoke check."""
from .config import Config
from .utils.logging import get_logger

log = get_logger("kolibriforge.doctor")


def run_doctor(cfg: Config):
    ok = True
    try:
        from .tokenizer import build_unibpe
        tok = build_unibpe(cfg)
        ids = tok.encode("Donaudampfschifffahrtsgesellschaft")
        log.info("tokenizer OK (compound-aware, %d ids)", len(ids))
    except Exception as e:
        ok = False
        log.error("tokenizer FAILED: %s", e)

    try:
        from .moe import MoELayer
        from .merlin import MerlinArthur
        from .reason.budget import ReasoningBudget
        MoELayer(cfg.moe.num_experts, cfg.moe.top_k, cfg.hidden, cfg.moe.expert_dim)
        MerlinArthur(cfg)
        ReasoningBudget(cfg)
        log.info("moe / merlin / budget components OK")
    except Exception as e:
        ok = False
        log.error("components FAILED: %s", e)

    try:
        from .backends import list_backends, get_backend
        names = list_backends()
        b = get_backend(cfg.backend.name)(cfg)
        log.info("backends OK (%s) -> %s", ", ".join(names), type(b).__name__)
    except Exception as e:
        ok = False
        log.error("backends FAILED: %s", e)

    try:
        from .bench import list_benchmarks
        log.info("benchmarks OK (%s)", ", ".join(list_benchmarks()))
    except Exception as e:
        ok = False
        log.error("benchmarks FAILED: %s", e)

    log.info("doctor %s", "PASS" if ok else "FAIL")
    return ok
