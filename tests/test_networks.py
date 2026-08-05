import torch
import numpy as np
import pytest
from train.networks import ActorNetwork, CriticNetwork


class TestActorNetwork:

    def test_output_shape(self):
        net = ActorNetwork(9, 5, [32, 32])
        x = torch.randn(4, 9)
        y = net(x)
        assert y.shape == (4, 5)

    def test_single_input(self):
        net = ActorNetwork(4, 2, [16])
        x = torch.randn(1, 4)
        y = net(x)
        assert y.shape == (1, 2)

    def test_get_action_returns_valid(self):
        net = ActorNetwork(9, 5, [32, 32])
        obs = np.random.rand(9).astype(np.float32)
        action, log_prob = net.get_action(obs)
        assert 0 <= action < 5
        assert log_prob <= 0.0

    def test_get_action_deterministic(self):
        net = ActorNetwork(4, 2, [16])
        net.eval()
        obs = np.random.rand(4).astype(np.float32)
        a1, _ = net.get_action(obs, deterministic=True)
        a2, _ = net.get_action(obs, deterministic=True)
        assert a1 == a2

    def test_evaluate_actions_shapes(self):
        net = ActorNetwork(9, 5, [32, 32])
        obs = torch.randn(10, 9)
        actions = torch.randint(0, 5, (10,))
        log_probs, entropy = net.evaluate_actions(obs, actions)
        assert log_probs.shape == (10,)
        assert entropy.shape == (10,)

    def test_evaluate_log_probs_negative(self):
        net = ActorNetwork(4, 2, [16])
        obs = torch.randn(5, 4)
        actions = torch.randint(0, 2, (5,))
        log_probs, _ = net.evaluate_actions(obs, actions)
        assert torch.all(log_probs <= 0.0)

    def test_entropy_positive(self):
        net = ActorNetwork(4, 2, [16])
        obs = torch.randn(5, 4)
        actions = torch.randint(0, 2, (5,))
        _, entropy = net.evaluate_actions(obs, actions)
        assert torch.all(entropy >= 0.0)


class TestCriticNetwork:

    def test_output_shape(self):
        net = CriticNetwork(9, [32, 32])
        x = torch.randn(4, 9)
        y = net(x)
        assert y.shape == (4,)

    def test_single_input(self):
        net = CriticNetwork(4, [16])
        x = torch.randn(1, 4)
        y = net(x)
        assert y.shape == (1,)

    def test_returns_scalar_per_input(self):
        net = CriticNetwork(9, [32, 32])
        x = torch.randn(8, 9)
        y = net(x)
        assert y.dim() == 1
        assert y.shape[0] == 8
