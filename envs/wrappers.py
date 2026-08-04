import numpy as np
import gymnasium as gym
from typing import Dict, Tuple, Optional
from envs.configs import EnvConfig, CARTPOLE_CONFIG, SIMPLE_SPREAD_CONFIG


class CartPoleWrapper:

    def __init__(self, config: EnvConfig = CARTPOLE_CONFIG):
        self.config = config
        self.env = gym.make("CartPole-v1")
        self.agents = ["agent_0"]
        self.num_agents = 1
        self.obs_dim = config.obs_dim
        self.act_dim = config.act_dim

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        clipped = np.clip(obs, self.config.obs_low, self.config.obs_high)
        span = self.config.obs_high - self.config.obs_low
        span = np.where(span == 0, 1.0, span)
        return ((clipped - self.config.obs_low) / span).astype(np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        obs, info = self.env.reset(seed=seed)
        norm_obs = self._normalize(obs)
        return {"agent_0": norm_obs}, {"agent_0": info}

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        obs, reward, terminated, truncated, info = self.env.step(actions["agent_0"])
        norm_obs = self._normalize(obs)
        return (
            {"agent_0": norm_obs},
            {"agent_0": float(reward)},
            {"agent_0": terminated},
            {"agent_0": truncated},
            {"agent_0": info},
        )

    def sample_observation(self) -> np.ndarray:
        return np.random.uniform(0.0, 1.0, size=(self.obs_dim,)).astype(np.float32)

    def close(self):
        self.env.close()


class SimpleSpreadWrapper:

    def __init__(self, config: EnvConfig = SIMPLE_SPREAD_CONFIG):
        from pettingzoo.mpe import simple_spread_v3
        self.config = config
        self.env = simple_spread_v3.parallel_env(N=3, max_cycles=25,
                                                  continuous_actions=False)
        self.agents = [f"agent_{i}" for i in range(config.num_agents)]
        self.num_agents = config.num_agents
        self.obs_dim = config.obs_dim
        self.act_dim = config.act_dim
        self._agent_name_map = {}

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        clipped = np.clip(obs, self.config.obs_low[:len(obs)],
                          self.config.obs_high[:len(obs)])
        span = self.config.obs_high[:len(obs)] - self.config.obs_low[:len(obs)]
        span = np.where(span == 0, 1.0, span)
        normalized = ((clipped - self.config.obs_low[:len(obs)]) / span).astype(np.float32)
        if len(normalized) < self.obs_dim:
            padded = np.zeros(self.obs_dim, dtype=np.float32)
            padded[:len(normalized)] = normalized
            return padded
        return normalized[:self.obs_dim]

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        obs, infos = self.env.reset(seed=seed)
        pz_agents = list(obs.keys())
        self._agent_name_map = {
            f"agent_{i}": pz_agents[i] for i in range(min(len(pz_agents), self.num_agents))
        }
        result_obs = {}
        result_infos = {}
        for our_name, pz_name in self._agent_name_map.items():
            result_obs[our_name] = self._normalize(obs[pz_name])
            result_infos[our_name] = infos.get(pz_name, {})
        return result_obs, result_infos

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        pz_actions = {}
        for our_name, action in actions.items():
            pz_name = self._agent_name_map.get(our_name)
            if pz_name is not None:
                pz_actions[pz_name] = action

        obs, rewards, terminated, truncated, infos = self.env.step(pz_actions)

        result_obs = {}
        result_rewards = {}
        result_term = {}
        result_trunc = {}
        result_infos = {}

        for our_name, pz_name in self._agent_name_map.items():
            if pz_name in obs:
                result_obs[our_name] = self._normalize(obs[pz_name])
                result_rewards[our_name] = float(rewards.get(pz_name, 0.0))
                result_term[our_name] = terminated.get(pz_name, False)
                result_trunc[our_name] = truncated.get(pz_name, False)
                result_infos[our_name] = infos.get(pz_name, {})
            else:
                result_obs[our_name] = np.zeros(self.obs_dim, dtype=np.float32)
                result_rewards[our_name] = 0.0
                result_term[our_name] = True
                result_trunc[our_name] = False
                result_infos[our_name] = {}

        return result_obs, result_rewards, result_term, result_trunc, result_infos

    def sample_observation(self) -> np.ndarray:
        return np.random.uniform(0.0, 1.0, size=(self.obs_dim,)).astype(np.float32)

    def close(self):
        self.env.close()
