import numpy as np
from typing import Dict, Tuple, Optional, Set


GRID_SIZE = 5
MAX_STEPS = 50
WALL_ROW = 2
GAP_COL = 2
STEP_PENALTY = -0.1
COLLISION_PENALTY = -0.5
GOAL_REWARD = 10.0
DISTANCE_REWARD_SCALE = 0.5

ACTION_DELTAS = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}

AGENT_IDS = ["agent_0", "agent_1"]


class CooperativeGridNav:

    def __init__(self, grid_size: int = GRID_SIZE, max_steps: int = MAX_STEPS,
                 comm_enabled: bool = True):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.comm_enabled = comm_enabled
        self.walls = self._build_walls()
        self.agents = list(AGENT_IDS)
        self.num_agents = len(self.agents)
        self.obs_dim = 9
        self.act_dim = 5
        self.starts = {
            "agent_0": (0, 0),
            "agent_1": (grid_size - 1, grid_size - 1),
        }
        self.goals = {
            "agent_0": (grid_size - 1, grid_size - 1),
            "agent_1": (0, 0),
        }
        self._positions: Dict[str, Tuple[int, int]] = {}
        self._last_actions: Dict[str, int] = {}
        self._prev_distances: Dict[str, int] = {}
        self._step_count: int = 0
        self._done: bool = False

    def _build_walls(self) -> Set[Tuple[int, int]]:
        walls = set()
        mid = self.grid_size // 2
        gap = self.grid_size // 2
        for c in range(self.grid_size):
            if c != gap:
                walls.add((mid, c))
        return walls

    def _manhattan(self, agent: str) -> int:
        r, c = self._positions[agent]
        gr, gc = self.goals[agent]
        return abs(r - gr) + abs(c - gc)

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        self._positions = {a: self.starts[a] for a in self.agents}
        self._last_actions = {a: 0 for a in self.agents}
        self._prev_distances = {a: self._manhattan(a) for a in self.agents}
        self._step_count = 0
        self._done = False
        obs = {a: self._get_obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def _normalize_pos(self, row: int, col: int) -> Tuple[float, float]:
        gs = max(self.grid_size - 1, 1)
        return row / gs, col / gs

    def _normalize_delta(self, delta: int) -> float:
        return (delta + 1) / 2.0

    def _get_obs(self, agent: str) -> np.ndarray:
        r, c = self._positions[agent]
        gr, gc = self.goals[agent]
        other = self.agents[1] if agent == self.agents[0] else self.agents[0]

        nr, nc = self._normalize_pos(r, c)
        ngr, ngc = self._normalize_pos(gr, gc)

        if self.comm_enabled:
            pr, pc = self._positions[other]
            npr, npc = self._normalize_pos(pr, pc)
            other_action = self._last_actions[other]
            dr, dc = ACTION_DELTAS[other_action]
            ndr = self._normalize_delta(dr)
            ndc = self._normalize_delta(dc)
            has_msg = 1.0 if self._step_count > 0 else 0.0
        else:
            npr, npc = 0.0, 0.0
            ndr, ndc = 0.5, 0.5
            has_msg = 0.0

        return np.array([nr, nc, ngr, ngc, npr, npc, ndr, ndc, has_msg],
                        dtype=np.float32)

    def _is_valid(self, row: int, col: int) -> bool:
        if row < 0 or row >= self.grid_size:
            return False
        if col < 0 or col >= self.grid_size:
            return False
        if (row, col) in self.walls:
            return False
        return True

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        if self._done:
            obs = {a: self._get_obs(a) for a in self.agents}
            zeros = {a: 0.0 for a in self.agents}
            dones = {a: True for a in self.agents}
            return obs, zeros, dones, dones, {a: {} for a in self.agents}

        self._step_count += 1

        intended = {}
        for agent in self.agents:
            r, c = self._positions[agent]
            dr, dc = ACTION_DELTAS[actions[agent]]
            nr, nc = r + dr, c + dc
            if not self._is_valid(nr, nc):
                nr, nc = r, c
            intended[agent] = (nr, nc)

        a0, a1 = self.agents[0], self.agents[1]
        had_collision = False

        if intended[a0] == intended[a1]:
            intended[a0] = self._positions[a0]
            intended[a1] = self._positions[a1]
            had_collision = True

        if (intended[a0] == self._positions[a1] and
                intended[a1] == self._positions[a0]):
            intended[a0] = self._positions[a0]
            intended[a1] = self._positions[a1]
            had_collision = True

        for agent in self.agents:
            self._positions[agent] = intended[agent]
            self._last_actions[agent] = actions[agent]

        both_at_goal = all(
            self._positions[a] == self.goals[a] for a in self.agents
        )
        timed_out = self._step_count >= self.max_steps

        rewards = {}
        for agent in self.agents:
            r = STEP_PENALTY
            if had_collision:
                r += COLLISION_PENALTY
            if both_at_goal:
                r += GOAL_REWARD
            curr_dist = self._manhattan(agent)
            prev_dist = self._prev_distances[agent]
            r += DISTANCE_REWARD_SCALE * (prev_dist - curr_dist)
            self._prev_distances[agent] = curr_dist
            rewards[agent] = r

        self._done = both_at_goal or timed_out

        obs = {a: self._get_obs(a) for a in self.agents}
        terminated = {a: both_at_goal for a in self.agents}
        truncated = {a: timed_out and not both_at_goal for a in self.agents}
        infos = {a: {} for a in self.agents}

        return obs, rewards, terminated, truncated, infos

    def sample_observation(self) -> np.ndarray:
        return np.random.uniform(0.0, 1.0, size=(self.obs_dim,)).astype(np.float32)
