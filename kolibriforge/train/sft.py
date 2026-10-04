"""Supervised fine-tuning for honest abstention.

Trains the backend's honesty skill on (question, answer, correct) triples. Unlike
the RL loop, this uses direct supervision: the ground-truth correctness is known
and the skill is nudged toward perfect calibration.
"""
from ..utils.stable import stable_float


def sft_honesty(cfg, backend=None, steps=40, lr=0.05):
    if backend is None:
        from ..backends import get_backend
        backend = get_backend("mock")(cfg)
    from ..merlin.protocol import MerlinArthur
    ma = MerlinArthur(cfg)
    history = []
    correct_count = 0
    for step in range(steps):
        q = f"question-{step}"
        correct = backend._is_correct(q)
        answer = backend._answer_text(q, correct)
        backend.train_step(q, answer, correct, lr=lr)
        verdict = ma.decide(q, answer, skill=backend.skill)
        if verdict["action"] == "accept" and correct:
            correct_count += 1
        history.append({"step": step, "skill": round(backend.skill, 4)})
    accuracy = correct_count / steps
    return {"history": history, "final_skill": backend.skill, "accuracy": accuracy}
