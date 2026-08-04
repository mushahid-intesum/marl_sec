import os
import numpy as np
from typing import Optional, Dict, Any

MAX_OPS = 16


class TimingDataset:

    def __init__(self, obs_dim: int, act_dim: int, max_ops: int = MAX_OPS):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_ops = max_ops
        self.observations: list = []
        self.actions: list = []
        self.total_cycles: list = []
        self.per_op_cycles: list = []
        self.metadata: Dict[str, Any] = {}

    def add_sample(self, obs: np.ndarray, action: int, total_cyc: int,
                   op_cycles: Optional[np.ndarray] = None):
        self.observations.append(obs.astype(np.float32))
        self.actions.append(action)
        self.total_cycles.append(total_cyc)
        if op_cycles is not None:
            padded = np.zeros(self.max_ops, dtype=np.uint32)
            n = min(len(op_cycles), self.max_ops)
            padded[:n] = op_cycles[:n]
            self.per_op_cycles.append(padded)
        else:
            self.per_op_cycles.append(np.zeros(self.max_ops, dtype=np.uint32))

    def add_batch(self, obs_batch: np.ndarray, actions: np.ndarray,
                  total_cycles: np.ndarray,
                  per_op_batch: Optional[np.ndarray] = None):
        n = len(obs_batch)
        for i in range(n):
            op_cyc = per_op_batch[i] if per_op_batch is not None else None
            self.add_sample(obs_batch[i], int(actions[i]),
                            int(total_cycles[i]), op_cyc)

    def __len__(self) -> int:
        return len(self.actions)

    def to_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "observations": np.array(self.observations, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.int32),
            "total_cycles": np.array(self.total_cycles, dtype=np.uint64),
            "per_op_cycles": np.array(self.per_op_cycles, dtype=np.uint32),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        arrays = self.to_arrays()
        np.savez(path,
                 observations=arrays["observations"],
                 actions=arrays["actions"],
                 total_cycles=arrays["total_cycles"],
                 per_op_cycles=arrays["per_op_cycles"],
                 obs_dim=np.array(self.obs_dim),
                 act_dim=np.array(self.act_dim),
                 max_ops=np.array(self.max_ops))

    @classmethod
    def load(cls, path: str) -> "TimingDataset":
        data = np.load(path)
        ds = cls(
            obs_dim=int(data["obs_dim"]),
            act_dim=int(data["act_dim"]),
            max_ops=int(data["max_ops"]),
        )
        ds.observations = list(data["observations"])
        ds.actions = list(data["actions"])
        ds.total_cycles = list(data["total_cycles"])
        ds.per_op_cycles = list(data["per_op_cycles"])
        return ds

    def get_timing_by_action(self) -> Dict[int, np.ndarray]:
        result = {}
        actions_arr = np.array(self.actions)
        cycles_arr = np.array(self.total_cycles, dtype=np.float64)
        for a in range(self.act_dim):
            mask = actions_arr == a
            if np.any(mask):
                result[a] = cycles_arr[mask]
        return result

    def get_per_op_by_action(self, op_idx: int) -> Dict[int, np.ndarray]:
        result = {}
        actions_arr = np.array(self.actions)
        ops_arr = np.array(self.per_op_cycles, dtype=np.float64)
        for a in range(self.act_dim):
            mask = actions_arr == a
            if np.any(mask):
                result[a] = ops_arr[mask, op_idx]
        return result
