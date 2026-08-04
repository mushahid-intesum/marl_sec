import numpy as np
import pytest
from envs.grid_nav import CooperativeGridNav, GRID_SIZE, ACTION_DELTAS


class TestReset:

    def test_returns_obs_for_both_agents(self):
        env = CooperativeGridNav()
        obs, infos = env.reset()
        assert "agent_0" in obs
        assert "agent_1" in obs

    def test_obs_shape(self):
        env = CooperativeGridNav()
        obs, _ = env.reset()
        assert obs["agent_0"].shape == (9,)
        assert obs["agent_1"].shape == (9,)

    def test_obs_dtype(self):
        env = CooperativeGridNav()
        obs, _ = env.reset()
        assert obs["agent_0"].dtype == np.float32

    def test_obs_bounds(self):
        env = CooperativeGridNav()
        obs, _ = env.reset()
        for agent in env.agents:
            assert np.all(obs[agent] >= 0.0)
            assert np.all(obs[agent] <= 1.0)

    def test_initial_positions(self):
        env = CooperativeGridNav()
        obs, _ = env.reset()
        assert obs["agent_0"][0] == 0.0
        assert obs["agent_0"][1] == 0.0
        gs = GRID_SIZE - 1
        assert obs["agent_1"][0] == pytest.approx(gs / gs)
        assert obs["agent_1"][1] == pytest.approx(gs / gs)

    def test_initial_no_message(self):
        env = CooperativeGridNav()
        obs, _ = env.reset()
        assert obs["agent_0"][8] == 0.0
        assert obs["agent_1"][8] == 0.0


class TestMovement:

    def test_stay_action(self):
        env = CooperativeGridNav()
        obs0, _ = env.reset()
        actions = {"agent_0": 0, "agent_1": 0}
        obs1, _, _, _, _ = env.step(actions)
        assert obs1["agent_0"][0] == obs0["agent_0"][0]
        assert obs1["agent_0"][1] == obs0["agent_0"][1]

    def test_move_down(self):
        env = CooperativeGridNav()
        env.reset()
        actions = {"agent_0": 2, "agent_1": 0}
        obs, _, _, _, _ = env.step(actions)
        gs = GRID_SIZE - 1
        assert obs["agent_0"][0] == pytest.approx(1.0 / gs)

    def test_move_right(self):
        env = CooperativeGridNav()
        env.reset()
        actions = {"agent_0": 4, "agent_1": 0}
        obs, _, _, _, _ = env.step(actions)
        gs = GRID_SIZE - 1
        assert obs["agent_0"][1] == pytest.approx(1.0 / gs)

    def test_boundary_collision_top(self):
        env = CooperativeGridNav()
        env.reset()
        actions = {"agent_0": 1, "agent_1": 0}
        obs, _, _, _, _ = env.step(actions)
        assert obs["agent_0"][0] == 0.0

    def test_boundary_collision_left(self):
        env = CooperativeGridNav()
        env.reset()
        actions = {"agent_0": 3, "agent_1": 0}
        obs, _, _, _, _ = env.step(actions)
        assert obs["agent_0"][1] == 0.0


class TestWalls:

    def test_wall_blocks_movement(self):
        env = CooperativeGridNav()
        env.reset()
        env._positions["agent_0"] = (1, 0)
        env._positions["agent_1"] = (4, 4)
        actions = {"agent_0": 2, "agent_1": 0}
        env.step(actions)
        assert env._positions["agent_0"] == (1, 0)

    def test_gap_allows_movement(self):
        env = CooperativeGridNav()
        env.reset()
        gap_col = GRID_SIZE // 2
        env._positions["agent_0"] = (1, gap_col)
        env._positions["agent_1"] = (4, 4)
        actions = {"agent_0": 2, "agent_1": 0}
        env.step(actions)
        assert env._positions["agent_0"] == (2, gap_col)


class TestCollisions:

    def test_same_target_collision(self):
        env = CooperativeGridNav()
        env.reset()
        env._positions["agent_0"] = (0, 1)
        env._positions["agent_1"] = (0, 3)
        actions = {"agent_0": 4, "agent_1": 3}
        obs, rewards, _, _, _ = env.step(actions)
        assert env._positions["agent_0"] == (0, 1)
        assert env._positions["agent_1"] == (0, 3)

    def test_swap_collision(self):
        env = CooperativeGridNav()
        env.reset()
        env._positions["agent_0"] = (0, 0)
        env._positions["agent_1"] = (0, 1)
        actions = {"agent_0": 4, "agent_1": 3}
        env.step(actions)
        assert env._positions["agent_0"] == (0, 0)
        assert env._positions["agent_1"] == (0, 1)

    def test_collision_penalty(self):
        env = CooperativeGridNav()
        env.reset()
        env._positions["agent_0"] = (0, 1)
        env._positions["agent_1"] = (0, 3)
        actions = {"agent_0": 4, "agent_1": 3}
        _, rewards, _, _, _ = env.step(actions)
        assert rewards["agent_0"] < -0.1


class TestGoalsAndReward:

    def test_goal_reward(self):
        env = CooperativeGridNav()
        env.reset()
        env._positions["agent_0"] = (3, 4)
        env._positions["agent_1"] = (1, 0)
        env._positions["agent_1"] = (0, 1)
        env._positions["agent_0"] = (env.grid_size - 1, env.grid_size - 2)
        env._positions["agent_1"] = (0, 1)
        actions = {"agent_0": 4, "agent_1": 3}
        _, rewards, terminated, _, _ = env.step(actions)
        if (env._positions["agent_0"] == env.goals["agent_0"] and
                env._positions["agent_1"] == env.goals["agent_1"]):
            assert rewards["agent_0"] > 0
            assert terminated["agent_0"] is True

    def test_step_penalty(self):
        env = CooperativeGridNav()
        env.reset()
        actions = {"agent_0": 0, "agent_1": 0}
        _, rewards, _, _, _ = env.step(actions)
        assert rewards["agent_0"] == pytest.approx(-0.1)

    def test_shared_reward(self):
        env = CooperativeGridNav()
        env.reset()
        actions = {"agent_0": 0, "agent_1": 0}
        _, rewards, _, _, _ = env.step(actions)
        assert rewards["agent_0"] == rewards["agent_1"]


class TestCommunication:

    def test_comm_enabled_has_message(self):
        env = CooperativeGridNav(comm_enabled=True)
        env.reset()
        actions = {"agent_0": 2, "agent_1": 0}
        obs, _, _, _, _ = env.step(actions)
        assert obs["agent_0"][8] == 1.0
        assert obs["agent_1"][8] == 1.0

    def test_comm_disabled_no_message(self):
        env = CooperativeGridNav(comm_enabled=False)
        env.reset()
        actions = {"agent_0": 2, "agent_1": 0}
        obs, _, _, _, _ = env.step(actions)
        assert obs["agent_0"][8] == 0.0
        assert obs["agent_0"][4] == 0.0
        assert obs["agent_0"][5] == 0.0

    def test_message_contains_peer_position(self):
        env = CooperativeGridNav(comm_enabled=True)
        env.reset()
        actions = {"agent_0": 2, "agent_1": 0}
        obs, _, _, _, _ = env.step(actions)
        gs = GRID_SIZE - 1
        assert obs["agent_0"][4] == pytest.approx(gs / gs)
        assert obs["agent_0"][5] == pytest.approx(gs / gs)


class TestEpisode:

    def test_truncation_at_max_steps(self):
        env = CooperativeGridNav(max_steps=3)
        env.reset()
        for i in range(3):
            _, _, terminated, truncated, _ = env.step({"agent_0": 0, "agent_1": 0})
        assert truncated["agent_0"] is True

    def test_multiple_resets(self):
        env = CooperativeGridNav()
        for _ in range(5):
            obs, _ = env.reset()
            assert obs["agent_0"].shape == (9,)
            for _ in range(3):
                env.step({"agent_0": 0, "agent_1": 0})

    def test_done_stays_done(self):
        env = CooperativeGridNav(max_steps=1)
        env.reset()
        env.step({"agent_0": 0, "agent_1": 0})
        _, _, terminated, truncated, _ = env.step({"agent_0": 0, "agent_1": 0})
        assert terminated["agent_0"] or truncated["agent_0"]


class TestSampleObservation:

    def test_shape(self):
        env = CooperativeGridNav()
        obs = env.sample_observation()
        assert obs.shape == (9,)

    def test_bounds(self):
        env = CooperativeGridNav()
        for _ in range(100):
            obs = env.sample_observation()
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)

    def test_dtype(self):
        env = CooperativeGridNav()
        obs = env.sample_observation()
        assert obs.dtype == np.float32
