import os
import tempfile
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
from collect.dataset import TimingDataset
from analysis.plots import (plot_timing_distributions, plot_mi_heatmap,
                             plot_confusion_matrix, plot_per_layer_breakdown)


def _make_dataset(n=100, act_dim=5):
    ds = TimingDataset(obs_dim=4, act_dim=act_dim)
    rng = np.random.RandomState(42)
    for i in range(n):
        action = i % act_dim
        cycles = 5000 + action * 500 + rng.randint(0, 200)
        obs = rng.rand(4).astype(np.float32)
        op_cycles = np.array([100 + action * 10, 200, 300, 150, 50, 80],
                             dtype=np.uint32)
        ds.add_sample(obs, action, cycles, op_cycles)
    return ds


class TestTimingDistributions:

    def test_returns_figure(self):
        ds = _make_dataset()
        fig = plot_timing_distributions(ds)
        assert fig is not None

    def test_saves_to_file(self):
        ds = _make_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.png")
            fig = plot_timing_distributions(ds, save_path=path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0


class TestMIHeatmap:

    def test_returns_figure(self):
        mi_data = {
            "env_a": {"fp32": 0.5, "int8": 0.3},
            "env_b": {"fp32": 1.2, "int8": 0.8},
        }
        fig = plot_mi_heatmap(mi_data)
        assert fig is not None

    def test_saves_to_file(self):
        mi_data = {
            "env_a": {"fp32": 0.5, "int8": 0.3},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "heatmap.png")
            fig = plot_mi_heatmap(mi_data, save_path=path)
            assert os.path.exists(path)


class TestConfusionMatrix:

    def test_returns_figure(self):
        cm = np.array([[40, 5, 5], [3, 42, 5], [2, 3, 45]])
        fig = plot_confusion_matrix(cm)
        assert fig is not None

    def test_saves_to_file(self):
        cm = np.array([[10, 2], [3, 15]])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cm.png")
            fig = plot_confusion_matrix(cm, save_path=path)
            assert os.path.exists(path)


class TestPerLayerBreakdown:

    def test_returns_figure(self):
        ds = _make_dataset()
        fig = plot_per_layer_breakdown(ds, n_ops=4)
        assert fig is not None

    def test_saves_to_file(self):
        ds = _make_dataset()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "layers.png")
            fig = plot_per_layer_breakdown(ds, n_ops=3, save_path=path)
            assert os.path.exists(path)
