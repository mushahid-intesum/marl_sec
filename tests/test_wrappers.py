import numpy as np
import pytest
from envs.wrappers import CartPoleWrapper, SimpleSpreadWrapper
from envs.configs import CARTPOLE_CONFIG, SIMPLE_SPREAD_CONFIG


class TestCartPoleWrapper:

    def test_reset_returns_dict(self):
        env = CartPoleWrapper()
        obs, infos = env.reset(seed=42)
        assert "agent_0" in obs
        env.close()

    def test_obs_shape(self):
        env = CartPoleWrapper()
        obs, _ = env.reset(seed=42)
        assert obs["agent_0"].shape == (4,)
        env.close()

    def test_obs_normalized_bounds(self):
        env = CartPoleWrapper()
        obs, _ = env.reset(seed=42)
        assert np.all(obs["agent_0"] >= 0.0)
        assert np.all(obs["agent_0"] <= 1.0)
        env.close()

    def test_obs_dtype(self):
        env = CartPoleWrapper()
        obs, _ = env.reset(seed=42)
        assert obs["agent_0"].dtype == np.float32
        env.close()

    def test_step_returns_correct_keys(self):
        env = CartPoleWrapper()
        env.reset(seed=42)
        obs, rewards, terminated, truncated, infos = env.step({"agent_0": 0})
        assert "agent_0" in obs
        assert "agent_0" in rewards
        assert "agent_0" in terminated
        assert "agent_0" in truncated
        env.close()

    def test_step_obs_normalized(self):
        env = CartPoleWrapper()
        env.reset(seed=42)
        obs, _, _, _, _ = env.step({"agent_0": 1})
        assert np.all(obs["agent_0"] >= 0.0)
        assert np.all(obs["agent_0"] <= 1.0)
        env.close()

    def test_sample_observation_shape(self):
        env = CartPoleWrapper()
        obs = env.sample_observation()
        assert obs.shape == (4,)
        env.close()

    def test_sample_observation_bounds(self):
        env = CartPoleWrapper()
        for _ in range(50):
            obs = env.sample_observation()
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)
        env.close()

    def test_multiple_steps(self):
        env = CartPoleWrapper()
        env.reset(seed=42)
        for _ in range(10):
            action = np.random.randint(0, 2)
            obs, r, term, trunc, _ = env.step({"agent_0": action})
            if term["agent_0"] or trunc["agent_0"]:
                break
            assert obs["agent_0"].shape == (4,)
        env.close()


class TestSimpleSpreadWrapper:

    def test_reset_returns_dict(self):
        env = SimpleSpreadWrapper()
        obs, infos = env.reset(seed=42)
        assert len(obs) == 3
        env.close()

    def test_obs_shape(self):
        env = SimpleSpreadWrapper()
        obs, _ = env.reset(seed=42)
        for agent in obs:
            assert obs[agent].shape == (18,)
        env.close()

    def test_obs_dtype(self):
        env = SimpleSpreadWrapper()
        obs, _ = env.reset(seed=42)
        for agent in obs:
            assert obs[agent].dtype == np.float32
        env.close()

    def test_obs_normalized_bounds(self):
        env = SimpleSpreadWrapper()
        obs, _ = env.reset(seed=42)
        for agent in obs:
            assert np.all(obs[agent] >= 0.0)
            assert np.all(obs[agent] <= 1.0)
        env.close()

    def test_step_returns_correct_agents(self):
        env = SimpleSpreadWrapper()
        obs, _ = env.reset(seed=42)
        actions = {a: np.random.randint(0, 5) for a in obs}
        obs2, rewards, term, trunc, infos = env.step(actions)
        assert len(obs2) == 3
        env.close()

    def test_sample_observation(self):
        env = SimpleSpreadWrapper()
        obs = env.sample_observation()
        assert obs.shape == (18,)
        assert obs.dtype == np.float32
        env.close()

    def test_multiple_steps(self):
        env = SimpleSpreadWrapper()
        obs, _ = env.reset(seed=42)
        for _ in range(5):
            actions = {a: np.random.randint(0, 5) for a in obs}
            obs, _, term, trunc, _ = env.step(actions)
            if any(term.values()) or any(trunc.values()):
                break
        env.close()
