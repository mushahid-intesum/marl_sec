from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class EnvConfig:
    name: str
    obs_dim: int
    act_dim: int
    num_agents: int
    obs_low: np.ndarray
    obs_high: np.ndarray
    hidden_sizes: List[int] = field(default_factory=lambda: [32, 32])


GRID_NAV_CONFIG = EnvConfig(
    name="grid_nav",
    obs_dim=9,
    act_dim=5,
    num_agents=2,
    obs_low=np.zeros(9, dtype=np.float32),
    obs_high=np.ones(9, dtype=np.float32),
    hidden_sizes=[32, 32],
)

SIMPLE_SPREAD_CONFIG = EnvConfig(
    name="simple_spread",
    obs_dim=18,
    act_dim=5,
    num_agents=3,
    obs_low=np.full(18, -2.0, dtype=np.float32),
    obs_high=np.full(18, 2.0, dtype=np.float32),
    hidden_sizes=[64, 64],
)

CARTPOLE_CONFIG = EnvConfig(
    name="cartpole",
    obs_dim=4,
    act_dim=2,
    num_agents=1,
    obs_low=np.array([-4.8, -3.0, -0.418, -3.0], dtype=np.float32),
    obs_high=np.array([4.8, 3.0, 0.418, 3.0], dtype=np.float32),
    hidden_sizes=[32, 32],
)

ALL_CONFIGS = [GRID_NAV_CONFIG, SIMPLE_SPREAD_CONFIG, CARTPOLE_CONFIG]
