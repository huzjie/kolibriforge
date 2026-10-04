"""Configuration model and loader (YAML with zero-dep fallback)."""
from dataclasses import dataclass, field
from pathlib import Path

from .utils.yamlish import parse as _yaml_parse


@dataclass
class BackendConfig:
    name: str = "mock"
    model: str = "kolibri-flash"
    api_base: str = ""
    api_key: str = ""
    temperature: float = 0.0


@dataclass
class MoEConfig:
    num_experts: int = 8
    top_k: int = 2
    hidden: int = 128
    expert_dim: int = 256
    # total parameters vs active parameters emulation ratio
    sparse_ratio: float = 22.6


@dataclass
class MerlinConfig:
    enabled: bool = True
    # below this confidence the model abstains ("I don't know")
    abstain_threshold: float = 0.5
    # reward terms for the Merlin-Arthur RL loop
    hallucination_penalty: float = 1.5
    abstain_reward: float = 0.2
    correctness_reward: float = 1.0


@dataclass
class BudgetConfig:
    enabled: bool = True
    levels: tuple = ("none", "low", "medium", "high")
    default_level: str = "medium"
    max_extra_tokens: int = 512


@dataclass
class TrainConfig:
    steps: int = 60
    lr: float = 0.05
    reward_noise: float = 0.5
    algorithm: str = "merlin-rl"


@dataclass
class Config:
    backend: BackendConfig = field(default_factory=BackendConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    merlin: MerlinConfig = field(default_factory=MerlinConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: str = "kolibri-flash"
    hidden: int = 128
    vocab: int = 256
    seed: int = 0

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        return cls(
            backend=BackendConfig(**d.get("backend", {})),
            moe=MoEConfig(**d.get("moe", {})),
            merlin=MerlinConfig(**d.get("merlin", {})),
            budget=BudgetConfig(**d.get("budget", {})),
            train=TrainConfig(**d.get("train", {})),
            model=d.get("model", "kolibri-flash"),
            hidden=int(d.get("hidden", 128)),
            vocab=int(d.get("vocab", 256)),
            seed=int(d.get("seed", 0)),
        )


def load_config(path=None):
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "default.yaml"
    p = Path(path)
    if not p.exists():
        return Config()
    text = p.read_text(encoding="utf-8")
    try:
        import yaml  # noqa: F401
        data = yaml.safe_load(text)
    except Exception:
        data = _yaml_parse(text)
    return Config.from_dict(data)
