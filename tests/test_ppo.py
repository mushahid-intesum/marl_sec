import numpy as np
import torch
import pytest
from train.config import PPOConfig
from train.ppo import PPOAgent
from train.buffer import RolloutBuffer


class TestPPOAgent:

    def test_get_action_valid(self):
        agent = PPOAgent(4, 2, PPOConfig())
        obs = np.random.rand(4).astype(np.float32)
        action, log_prob, value = agent.get_action(obs)
        assert 0 <= action < 2
        assert log_prob <= 0.0
        assert isinstance(value, float)

    def test_get_value(self):
        agent = PPOAgent(9, 5, PPOConfig())
        obs = np.random.rand(9).astype(np.float32)
        val = agent.get_value(obs)
        assert isinstance(val, float)

    def test_update_returns_stats(self):
        agent = PPOAgent(4, 2, PPOConfig(n_epochs=1, batch_size=8))
        buf = RolloutBuffer(16, 4)
        for _ in range(16):
            obs = np.random.rand(4).astype(np.float32)
            action, lp, val = agent.get_action(obs)
            buf.add(obs, action, 1.0, False, lp, val)
        buf.compute_gae(0.0, 0.99, 0.95)
        stats = agent.update(buf)
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "entropy" in stats

    def test_update_changes_params(self):
        agent = PPOAgent(4, 2, PPOConfig(n_epochs=2, batch_size=8))
        buf = RolloutBuffer(16, 4)
        for _ in range(16):
            obs = np.random.rand(4).astype(np.float32)
            action, lp, val = agent.get_action(obs)
            buf.add(obs, action, float(action), False, lp, val)
        buf.compute_gae(0.0, 0.99, 0.95)

        params_before = [p.clone() for p in agent.actor.parameters()]
        agent.update(buf)
        params_after = list(agent.actor.parameters())

        changed = False
        for p_before, p_after in zip(params_before, params_after):
            if not torch.allclose(p_before, p_after):
                changed = True
                break
        assert changed

    def test_save_load_roundtrip(self, tmp_path):
        agent = PPOAgent(4, 2, PPOConfig())
        obs = np.random.rand(4).astype(np.float32)
        a1, _, _ = agent.get_action(obs, deterministic=True)

        path = str(tmp_path / "test.pt")
        agent.save(path)

        agent2 = PPOAgent(4, 2, PPOConfig())
        agent2.load(path)
        a2, _, _ = agent2.get_action(obs, deterministic=True)
        assert a1 == a2


class TestPPOTraining:

    def test_short_cartpole_run(self):
        from train.runner import train_cartpole
        from train.config import TrainConfig
        tc = TrainConfig(total_timesteps=256, rollout_steps=64,
                         log_interval=10000, save_dir="/tmp/test_marl_ckpt")
        result = train_cartpole(train_config=tc)
        assert result["actor"] is not None
        assert len(result["episode_rewards"]) > 0

    def test_short_grid_nav_run(self):
        from train.runner import train_grid_nav
        from train.config import TrainConfig
        tc = TrainConfig(total_timesteps=256, rollout_steps=64,
                         log_interval=10000, save_dir="/tmp/test_marl_ckpt")
        result = train_grid_nav(train_config=tc)
        assert "agent_0" in result["actors"]
        assert "agent_1" in result["actors"]
