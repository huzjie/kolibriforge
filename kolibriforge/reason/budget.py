"""Reasoning budget: map task difficulty to a reasoning depth.

Kolibri-1 exposes four inference-time reasoning depths -- none / low / medium /
high -- so the user can trade answer quality against cost and latency. This
module formalizes that as a budget: a difficulty estimate is turned into a
depth level and an extra-token allowance.

The reusable idea: make "how much should this model think" an *explicit,
user-controllable dial*, not a hidden property. Hard, verifiable tasks get a big
budget; trivial or well-practiced tasks skip straight to the answer.
"""

LEVELS = ("none", "low", "medium", "high")
LEVEL_COSTS = {"none": 0, "low": 64, "medium": 192, "high": 512}


class ReasoningBudget:
    def __init__(self, cfg):
        self.cfg = cfg
        self.default_level = cfg.budget.default_level
        self.max_extra_tokens = cfg.budget.max_extra_tokens

    def plan(self, query):
        """Return a structured budget plan for a query."""
        difficulty = self.estimate_difficulty(query)
        level = self.route_level(difficulty)
        return {
            "query": query,
            "difficulty": round(difficulty, 4),
            "level": level,
            "extra_tokens": LEVEL_COSTS[level],
            "reasoning_enabled": level != "none",
        }

    def estimate_difficulty(self, query):
        """A cheap, deterministic difficulty heuristic (0..1).

        Reusable signal: length + interrogative density + arithmetic/verifiable
        markers. In a real system this is a small router model; here it is a
        stable function so tests and demos are reproducible.
        """
        q = str(query).lower()
        score = min(len(q) / 120.0, 1.0)
        interrogatives = sum(q.count(w) for w in ("why", "how", "prove", "explain",
                                                   "warum", "wie", "beweise", "erkläre"))
        score += 0.1 * min(interrogatives, 5)
        if any(ch in q for ch in "0123456789+-*/="):
            score += 0.15  # arithmetic / symbolic tasks need more thought
        return min(score, 1.0)

    def route_level(self, difficulty):
        if difficulty < 0.25:
            return "none"
        if difficulty < 0.5:
            return "low"
        if difficulty < 0.75:
            return "medium"
        return "high"
