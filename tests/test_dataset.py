import os
import tempfile
import numpy as np
import pytest
from collect.dataset import TimingDataset


def _make_dataset(n=100, obs_dim=9, act_dim=5):
    ds = TimingDataset(obs_dim=obs_dim, act_dim=act_dim)
    for i in range(n):
        obs = np.random.rand(obs_dim).astype(np.float32)
        action = i % act_dim
        cycles = 1000 + action * 100 + np.random.randint(0, 50)
        op_cycles = np.array([200 + action * 10, 300, 400, 100], dtype=np.uint32)
        ds.add_sample(obs, action, cycles, op_cycles)
    return ds


class TestAddSample:

    def test_length_increases(self):
        ds = TimingDataset(obs_dim=9, act_dim=5)
        assert len(ds) == 0
        ds.add_sample(np.zeros(9, dtype=np.float32), 0, 1000)
        assert len(ds) == 1

    def test_stores_correct_values(self):
        ds = TimingDataset(obs_dim=4, act_dim=2)
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        ds.add_sample(obs, 1, 5000)
        assert ds.actions[0] == 1
        assert ds.total_cycles[0] == 5000
        np.testing.assert_array_equal(ds.observations[0], obs)

    def test_op_cycles_padded(self):
        ds = TimingDataset(obs_dim=4, act_dim=2, max_ops=8)
        op = np.array([100, 200], dtype=np.uint32)
        ds.add_sample(np.zeros(4, dtype=np.float32), 0, 300, op)
        assert len(ds.per_op_cycles[0]) == 8
        assert ds.per_op_cycles[0][0] == 100
        assert ds.per_op_cycles[0][2] == 0


class TestAddBatch:

    def test_batch_adds_all(self):
        ds = TimingDataset(obs_dim=4, act_dim=2)
        obs = np.random.rand(10, 4).astype(np.float32)
        actions = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        cycles = np.array([100] * 10, dtype=np.uint64)
        ds.add_batch(obs, actions, cycles)
        assert len(ds) == 10


class TestToArrays:

    def test_shapes(self):
        ds = _make_dataset(50)
        arrays = ds.to_arrays()
        assert arrays["observations"].shape == (50, 9)
        assert arrays["actions"].shape == (50,)
        assert arrays["total_cycles"].shape == (50,)
        assert arrays["per_op_cycles"].shape == (50, 16)

    def test_dtypes(self):
        ds = _make_dataset(10)
        arrays = ds.to_arrays()
        assert arrays["observations"].dtype == np.float32
        assert arrays["actions"].dtype == np.int32
        assert arrays["total_cycles"].dtype == np.uint64


class TestSaveLoad:

    def test_round_trip(self):
        ds = _make_dataset(30)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            ds.save(path)
            loaded = TimingDataset.load(path)
            assert len(loaded) == 30
            assert loaded.obs_dim == 9
            assert loaded.act_dim == 5

    def test_values_preserved(self):
        ds = _make_dataset(10)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.npz")
            ds.save(path)
            loaded = TimingDataset.load(path)
            orig = ds.to_arrays()
            back = loaded.to_arrays()
            np.testing.assert_array_equal(orig["actions"], back["actions"])
            np.testing.assert_array_almost_equal(orig["observations"],
                                                  back["observations"])

    def test_empty_dataset(self):
        ds = TimingDataset(obs_dim=4, act_dim=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.npz")
            ds.save(path)
            loaded = TimingDataset.load(path)
            assert len(loaded) == 0


class TestGroupByAction:

    def test_correct_groups(self):
        ds = _make_dataset(100, act_dim=5)
        groups = ds.get_timing_by_action()
        assert len(groups) == 5
        total = sum(len(v) for v in groups.values())
        assert total == 100

    def test_per_op_groups(self):
        ds = _make_dataset(50, act_dim=3)
        groups = ds.get_per_op_by_action(0)
        assert len(groups) == 3
