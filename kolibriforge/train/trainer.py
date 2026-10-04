"""Unified training entrypoint used by `kolibriforge train`."""
from ..utils.logging import get_logger

log = get_logger("kolibriforge.train")


def run_training(cfg, steps=None, algorithm=None):
    from ..backends import get_backend
    from .merlin_rl import MerlinRLTrainer

    steps = steps or cfg.train.steps
    algo = algorithm or cfg.train.algorithm

    if algo == "merlin-rl":
        backend = get_backend("mock")(cfg)
        t = MerlinRLTrainer(cfg, backend=backend)
        result = t.run(steps=steps)
        log.info("merlin-rl: skill %.4f -> mean_reward %.4f, hallucination=%d abstain=%d",
                 result["final_skill"], result["mean_reward"],
                 result["hallucination_count"], result["abstain_count"])
        # evaluate calibration quality after training
        from ..merlin.honesty import expected_calibration_error
        confs, corrects = [], []
        for step in range(200):
            q = f"eval-{step}"
            corr = backend._is_correct(q)
            confs.append(backend.confidence(q))
            corrects.append(1.0 if corr else 0.0)
        ece = expected_calibration_error(confs, corrects)
        log.info("post-train ECE = %.4f", ece)
        result["ece"] = round(ece, 4)
        return result

    if algo == "pretrain":
        from .pretrain import pretrain_moe
        return pretrain_moe(cfg, steps=steps)

    if algo == "sft":
        from .sft import sft_honesty
        return sft_honesty(cfg, steps=steps)

    raise ValueError(f"unknown algorithm '{algo}'")
