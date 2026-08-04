import numpy as np
import pytest
from collect.dataset import TimingDataset
from analysis.classifier import train_timing_classifier, evaluate_classifier


def _make_perfect_dataset(n=500, act_dim=5):
    ds = TimingDataset(obs_dim=4, act_dim=act_dim)
    for i in range(n):
        action = i % act_dim
        cycles = 10000 + action * 5000
        obs = np.random.rand(4).astype(np.float32)
        ds.add_sample(obs, action, cycles)
    return ds


def _make_random_dataset(n=500, act_dim=5):
    ds = TimingDataset(obs_dim=4, act_dim=act_dim)
    rng = np.random.RandomState(42)
    for i in range(n):
        action = i % act_dim
        cycles = 10000 + rng.randint(0, 10)
        obs = rng.rand(4).astype(np.float32)
        ds.add_sample(obs, action, cycles)
    return ds


class TestRandomForest:

    def test_perfect_leakage_high_accuracy(self):
        ds = _make_perfect_dataset()
        result = train_timing_classifier(ds, model_type="rf", use_per_op=False)
        assert result["accuracy"] > 0.9

    def test_no_leakage_near_random(self):
        ds = _make_random_dataset()
        result = train_timing_classifier(ds, model_type="rf", use_per_op=False)
        assert result["accuracy"] < 0.4

    def test_returns_confusion_matrix(self):
        ds = _make_perfect_dataset()
        result = train_timing_classifier(ds, model_type="rf")
        assert result["confusion_matrix"].shape == (5, 5)

    def test_returns_all_keys(self):
        ds = _make_perfect_dataset()
        result = train_timing_classifier(ds, model_type="rf")
        assert "accuracy" in result
        assert "f1_macro" in result
        assert "random_baseline" in result
        assert "n_train" in result
        assert "n_test" in result


class TestMLP:

    def test_perfect_leakage_high_accuracy(self):
        ds = _make_perfect_dataset()
        result = train_timing_classifier(ds, model_type="mlp", use_per_op=False)
        assert result["accuracy"] > 0.9

    def test_no_leakage_near_random(self):
        ds = _make_random_dataset()
        result = train_timing_classifier(ds, model_type="mlp", use_per_op=False)
        assert result["accuracy"] < 0.4


class TestEvaluateClassifier:

    def test_lift_over_random(self):
        ds = _make_perfect_dataset()
        result = train_timing_classifier(ds, model_type="rf", use_per_op=False)
        evaluation = evaluate_classifier(result)
        assert evaluation["lift_over_random"] > 3.0
        assert evaluation["above_random"] is True

    def test_no_leakage_low_lift(self):
        ds = _make_random_dataset()
        result = train_timing_classifier(ds, model_type="rf", use_per_op=False)
        evaluation = evaluate_classifier(result)
        assert evaluation["lift_over_random"] < 2.0
