import numpy as np
from typing import Optional
from envs.configs import EnvConfig

DEFAULT_N_SAMPLES = 1000


class ObservationSampler:

    def __init__(self, config: EnvConfig):
        self.config = config

    def uniform(self, n: int = DEFAULT_N_SAMPLES,
                seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        low = self.config.obs_low
        high = self.config.obs_high
        return rng.uniform(low, high, size=(n, self.config.obs_dim)).astype(np.float32)

    def on_policy(self, env, policy_fn, n: int = DEFAULT_N_SAMPLES,
                  seed: Optional[int] = None) -> np.ndarray:
        observations = []
        obs, _ = env.reset(seed=seed)
        agent_key = list(obs.keys())[0]

        while len(observations) < n:
            observations.append(obs[agent_key].copy())
            actions = {}
            for agent in obs:
                actions[agent] = policy_fn(obs[agent])
            obs, _, terminated, truncated, _ = env.step(actions)
            done = any(terminated.values()) or any(truncated.values())
            if done:
                obs, _ = env.reset()

        return np.array(observations[:n], dtype=np.float32)

    def adversarial(self, n: int = DEFAULT_N_SAMPLES,
                    seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        low = self.config.obs_low
        high = self.config.obs_high
        mid = (low + high) / 2.0
        samples = []
        for _ in range(n):
            obs = mid.copy()
            dim = rng.randint(0, self.config.obs_dim)
            if rng.random() > 0.5:
                obs[dim] = high[dim]
            else:
                obs[dim] = low[dim]
            samples.append(obs)
        return np.array(samples, dtype=np.float32)
