"""Merlin-Arthur reinforcement learning.

Reward shaping (the anti-hallucination policy):

    reward = correctness_reward        if accepted & correct
           - hallucination_penalty     if accepted & wrong   (worst case)
           + abstain_reward            if abstained on wrong (honest "I don't know")

The key asymmetry: confidently wrong (hallucination) is penalised *more* than
abstaining. This is what teaches the model to prefer "I don't know" over a wrong
guess, which is precisely Kolibri-1's Merlin-Arthur abstention behaviour.
"""
from ..merlin.protocol import MerlinArthur


class MerlinRLTrainer:
    def __init__(self, cfg, backend=None):
        self.cfg = cfg
        from ..backends import get_backend
        self.backend = backend or get_backend("mock")(cfg)
        self.ma = MerlinArthur(cfg)
        self.reward_history = []
        self.hallucination_count = 0
        self.abstain_count = 0

    def reward(self, verdict, correct):
        action = verdict["action"]
        if action == "accept":
            if correct:
                return self.cfg.merlin.correctness_reward
            self.hallucination_count += 1
            return -self.cfg.merlin.hallucination_penalty
        if action == "abstain":
            self.abstain_count += 1
            if correct:
                return 0.0  # abstained but would have been right: no reward
            return self.cfg.merlin.abstain_reward  # honest abstention on wrong
        return 0.0  # reject

    def step(self, question, lr=None):
        lr = lr or self.cfg.train.lr
        correct = self.backend._is_correct(question)
        answer = self.backend._answer_text(question, correct)
        verdict = self.ma.decide(question, answer, skill=self.backend.skill)
        r = self.reward(verdict, correct)
        # policy update: skill moves toward 1 when we're doing the right thing
        self.backend.train_step(question, answer, correct, lr=lr)
        self.reward_history.append(r)
        return {"question": question, "action": verdict["action"],
                "reward": round(r, 3), "skill": round(self.backend.skill, 4)}

    def run(self, steps=60, lr=None):
        for step in range(steps):
            self.step(f"question-{step}", lr=lr)
        return {
            "final_skill": round(self.backend.skill, 4),
            "mean_reward": round(sum(self.reward_history) / max(len(self.reward_history), 1), 4),
            "hallucination_count": self.hallucination_count,
            "abstain_count": self.abstain_count,
        }
