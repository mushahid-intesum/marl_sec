import numpy as np
import pytest
from train.buffer import RolloutBuffer


class TestAddAndSize:

    def test_empty_buffer(self):
        buf = RolloutBuffer(100, 4)
        assert buf.size == 0

    def test_add_increases_size(self):
        buf = RolloutBuffer(100, 4)
        buf.add(np.zeros(4), 0, 1.0, False, -0.5, 0.3)
        assert buf.size == 1

    def test_stores_values(self):
        buf = RolloutBuffer(100, 4)
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        buf.add(obs, 2, 1.5, False, -0.7, 0.5)
        np.testing.assert_array_equal(buf.observations[0], obs)
        assert buf.actions[0] == 2
        assert buf.rewards[0] == pytest.approx(1.5)
        assert buf.dones[0] == 0.0
        assert buf.log_probs[0] == pytest.approx(-0.7)
        assert buf.values[0] == pytest.approx(0.5)

    def test_capacity_wraps(self):
        buf = RolloutBuffer(5, 2)
        for i in range(10):
            buf.add(np.zeros(2), 0, 1.0, False, 0.0, 0.0)
        assert buf.size == 5


class TestGAE:

    def test_basic_gae(self):
        buf = RolloutBuffer(3, 2)
        for i in range(3):
            buf.add(np.zeros(2), 0, 1.0, False, 0.0, 0.5)
        buf.compute_gae(0.5, gamma=0.99, gae_lambda=0.95)
        assert buf.advantages[0] != 0.0
        assert buf.returns[0] != 0.0

    def test_returns_greater_than_advantages(self):
        buf = RolloutBuffer(5, 2)
        for i in range(5):
            buf.add(np.zeros(2), 0, 1.0, False, 0.0, 1.0)
        buf.compute_gae(1.0, gamma=0.99, gae_lambda=0.95)
        for i in range(5):
            assert buf.returns[i] == pytest.approx(buf.advantages[i] + buf.values[i])

    def test_done_resets_advantage(self):
        buf = RolloutBuffer(4, 2)
        buf.add(np.zeros(2), 0, 1.0, False, 0.0, 0.5)
        buf.add(np.zeros(2), 0, 1.0, True, 0.0, 0.5)
        buf.add(np.zeros(2), 0, 1.0, False, 0.0, 0.5)
        buf.add(np.zeros(2), 0, 1.0, False, 0.0, 0.5)
        buf.compute_gae(0.5, gamma=0.99, gae_lambda=0.95)
        assert buf.advantages[1] != buf.advantages[0]


class TestBatches:

    def test_yields_correct_keys(self):
        buf = RolloutBuffer(10, 4)
        for _ in range(10):
            buf.add(np.random.rand(4).astype(np.float32), 0, 1.0, False, 0.0, 0.0)
        buf.compute_gae(0.0, 0.99, 0.95)
        for batch in buf.get_batches(5):
            assert "observations" in batch
            assert "actions" in batch
            assert "advantages" in batch
            assert "returns" in batch
            assert "log_probs" in batch
            break

    def test_batch_sizes(self):
        buf = RolloutBuffer(20, 4)
        for _ in range(20):
            buf.add(np.random.rand(4).astype(np.float32), 0, 1.0, False, 0.0, 0.0)
        buf.compute_gae(0.0, 0.99, 0.95)
        sizes = [batch["observations"].shape[0] for batch in buf.get_batches(8)]
        assert sum(sizes) == 20

    def test_reset_clears(self):
        buf = RolloutBuffer(10, 4)
        for _ in range(5):
            buf.add(np.zeros(4), 0, 1.0, False, 0.0, 0.0)
        assert buf.size == 5
        buf.reset()
        assert buf.size == 0
