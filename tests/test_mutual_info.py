import numpy as np
import pytest
from collect.dataset import TimingDataset
from analysis.mutual_info import compute_mi_total, compute_mi_per_op, compute_mi_bootstrap


def _make_perfect_leakage_dataset(n=500, act_dim=5):
    ds = TimingDataset(obs_dim=4, act_dim=act_dim)
    for i in range(n):
        action = i % act_dim
        cycles = 10000 + action * 5000
        obs = np.random.rand(4).astype(np.float32)
        ds.add_sample(obs, action, cycles)
    return ds


def _make_zero_leakage_dataset(n=500, act_dim=5):
    ds = TimingDataset(obs_dim=4, act_dim=act_dim)
    for i in range(n):
        action = i % act_dim
        cycles = 10000
        obs = np.random.rand(4).astype(np.float32)
        ds.add_sample(obs, action, cycles)
    return ds


def _make_noisy_leakage_dataset(n=1000, act_dim=5):
    ds = TimingDataset(obs_dim=4, act_dim=act_dim)
    rng = np.random.RandomState(42)
    for i in range(n):
        action = i % act_dim
        cycles = 10000 + action * 200 + rng.randint(0, 100)
        obs = rng.rand(4).astype(np.float32)
        ds.add_sample(obs, action, cycles)
    return ds


class TestMITotal:

    def test_perfect_leakage_high_mi(self):
        ds = _make_perfect_leakage_dataset()
        result = compute_mi_total(ds)
        assert result["mi_bits"] > 1.5

    def test_zero_leakage_low_mi(self):
        ds = _make_zero_leakage_dataset()
        result = compute_mi_total(ds)
        assert result["mi_bits"] < 0.5

    def test_max_possible_bits(self):
        ds = _make_perfect_leakage_dataset(act_dim=5)
        result = compute_mi_total(ds)
        assert result["max_possible_bits"] == pytest.approx(np.log2(5), rel=0.01)

    def test_noisy_leakage_moderate_mi(self):
        ds = _make_noisy_leakage_dataset()
        result = compute_mi_total(ds)
        assert 0.0 < result["mi_bits"] < result["max_possible_bits"] * 1.1


class TestMIPerOp:

    def test_returns_dict(self):
        ds = _make_noisy_leakage_dataset()
        result = compute_mi_per_op(ds)
        assert isinstance(result, dict)

    def test_zero_for_constant_ops(self):
        ds = _make_zero_leakage_dataset()
        result = compute_mi_per_op(ds)
        for v in result.values():
            assert v < 0.5


class TestMIBootstrap:

    def test_returns_ci(self):
        ds = _make_noisy_leakage_dataset()
        result = compute_mi_bootstrap(ds, n_bootstrap=20)
        assert "ci_low" in result
        assert "ci_high" in result
        assert result["ci_low"] <= result["ci_high"]

    def test_ci_covers_mean(self):
        ds = _make_noisy_leakage_dataset()
        result = compute_mi_bootstrap(ds, n_bootstrap=20)
        assert result["ci_low"] <= result["mi_mean"] <= result["ci_high"]
