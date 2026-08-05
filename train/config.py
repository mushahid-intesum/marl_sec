from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    hidden_sizes: List[int] = field(default_factory=lambda: [32, 32])


@dataclass
class TrainConfig:
    total_timesteps: int = 50000
    rollout_steps: int = 2048
    eval_interval: int = 5000
    eval_episodes: int = 10
    save_dir: str = "checkpoints"
    seed: int = 42
    log_interval: int = 1000
