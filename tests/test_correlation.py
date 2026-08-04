import numpy as np
import pytest
from collect.dataset import TimingDataset
from analysis.correlation import (timing_kruskal_wallis, timing_anova,
                                   timing_spearman, timing_cohens_d,
                                   timing_summary_stats)


def _make_separated_dataset(n=200, act_dim=3):
    ds = TimingDataset(obs_dim=4, act_dim=act_dim)
    rng = np.random.RandomState(42)
    for i in range(n):
        action = i % act_dim
        cycles = 5000 + action * 2000 + rng.randint(0, 100)
        obs = rng.rand(4).astype(np.float32)
        ds.add_sample(obs, action, cycles)
    return ds


def _make_identical_dataset(n=200, act_dim=3):
    ds = TimingDataset(obs_dim=4, act_dim=act_dim)
    rng = np.random.RandomState(42)
    for i in range(n):
        action = i % act_dim
        cycles = 5000 + rng.randint(0, 10)
        obs = rng.rand(4).astype(np.float32)
        ds.add_sample(obs, action, cycles)
    return ds


class TestKruskalWallis:

    def test_significant_for_separated(self):
        ds = _make_separated_dataset()
        result = timing_kruskal_wallis(ds)
        assert result["p_value"] < 0.01

    def test_not_significant_for_identical(self):
        ds = _make_identical_dataset()
        result = timing_kruskal_wallis(ds)
        assert result["p_value"] > 0.01

    def test_returns_n_groups(self):
        ds = _make_separated_dataset(act_dim=4)
        result = timing_kruskal_wallis(ds)
        assert result["n_groups"] == 4


class TestAnova:

    def test_significant_for_separated(self):
        ds = _make_separated_dataset()
        result = timing_anova(ds)
        assert result["p_value"] < 0.01

    def test_not_significant_for_identical(self):
        ds = _make_identical_dataset()
        result = timing_anova(ds)
        assert result["p_value"] > 0.01


class TestSpearman:

    def test_returns_per_dim(self):
        ds = _make_separated_dataset()
        result = timing_spearman(ds)
        assert len(result) == 4

    def test_keys_format(self):
        ds = _make_separated_dataset()
        result = timing_spearman(ds)
        assert "obs_dim_0" in result
        assert "correlation" in result["obs_dim_0"]


class TestCohensD:

    def test_large_for_separated(self):
        ds = _make_separated_dataset()
        result = timing_cohens_d(ds)
        assert len(result) > 0
        max_d = max(abs(v) for v in result.values())
        assert max_d > 1.0

    def test_small_for_identical(self):
        ds = _make_identical_dataset()
        result = timing_cohens_d(ds)
        if result:
            max_d = max(abs(v) for v in result.values())
            assert max_d < 1.0


class TestSummaryStats:

    def test_returns_per_action(self):
        ds = _make_separated_dataset(act_dim=3)
        result = timing_summary_stats(ds)
        assert len(result) == 3

    def test_contains_expected_keys(self):
        ds = _make_separated_dataset()
        result = timing_summary_stats(ds)
        first = list(result.values())[0]
        assert "mean" in first
        assert "std" in first
        assert "median" in first
        assert "count" in first
        assert "cv" in first
