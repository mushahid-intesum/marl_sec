import numpy as np
import pytest
from envs.configs import GRID_NAV_CONFIG, CARTPOLE_CONFIG
from collect.sampler import ObservationSampler


class TestUniformSampler:

    def test_shape(self):
        sampler = ObservationSampler(GRID_NAV_CONFIG)
        obs = sampler.uniform(100, seed=42)
        assert obs.shape == (100, 9)

    def test_bounds(self):
        sampler = ObservationSampler(GRID_NAV_CONFIG)
        obs = sampler.uniform(500, seed=42)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_dtype(self):
        sampler = ObservationSampler(GRID_NAV_CONFIG)
        obs = sampler.uniform(10, seed=42)
        assert obs.dtype == np.float32

    def test_cartpole_bounds(self):
        sampler = ObservationSampler(CARTPOLE_CONFIG)
        obs = sampler.uniform(100, seed=42)
        assert np.all(obs >= CARTPOLE_CONFIG.obs_low)
        assert np.all(obs <= CARTPOLE_CONFIG.obs_high)

    def test_deterministic_with_seed(self):
        sampler = ObservationSampler(GRID_NAV_CONFIG)
        obs1 = sampler.uniform(50, seed=123)
        obs2 = sampler.uniform(50, seed=123)
        np.testing.assert_array_equal(obs1, obs2)


class TestAdversarialSampler:

    def test_shape(self):
        sampler = ObservationSampler(GRID_NAV_CONFIG)
        obs = sampler.adversarial(100, seed=42)
        assert obs.shape == (100, 9)

    def test_dtype(self):
        sampler = ObservationSampler(GRID_NAV_CONFIG)
        obs = sampler.adversarial(10, seed=42)
        assert obs.dtype == np.float32

    def test_contains_extremes(self):
        sampler = ObservationSampler(GRID_NAV_CONFIG)
        obs = sampler.adversarial(1000, seed=42)
        assert np.any(obs == 0.0)
        assert np.any(obs == 1.0)


class TestOnPolicySampler:

    def test_shape(self):
        from envs.grid_nav import CooperativeGridNav
        env = CooperativeGridNav()
        sampler = ObservationSampler(GRID_NAV_CONFIG)
        policy_fn = lambda obs: np.random.randint(0, 5)
        obs = sampler.on_policy(env, policy_fn, n=50, seed=42)
        assert obs.shape == (50, 9)

    def test_dtype(self):
        from envs.grid_nav import CooperativeGridNav
        env = CooperativeGridNav()
        sampler = ObservationSampler(GRID_NAV_CONFIG)
        policy_fn = lambda obs: np.random.randint(0, 5)
        obs = sampler.on_policy(env, policy_fn, n=20, seed=42)
        assert obs.dtype == np.float32
