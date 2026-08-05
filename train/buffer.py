import numpy as np
import torch
from typing import Optional


class RolloutBuffer:

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: int, reward: float,
            done: bool, log_prob: float, value: float):
        idx = self.ptr % self.capacity
        self.observations[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.log_probs[idx] = log_prob
        self.values[idx] = value
        self.ptr += 1
        self.size = min(self.ptr, self.capacity)

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        last_adv = 0.0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_value = last_value
                next_done = 0.0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            last_adv = delta + gamma * gae_lambda * (1 - self.dones[t]) * last_adv
            self.advantages[t] = last_adv
        self.returns[:self.size] = self.advantages[:self.size] + self.values[:self.size]

    def get_batches(self, batch_size: int):
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)
            batch_idx = indices[start:end]
            yield {
                "observations": torch.FloatTensor(self.observations[batch_idx]),
                "actions": torch.LongTensor(self.actions[batch_idx]),
                "log_probs": torch.FloatTensor(self.log_probs[batch_idx]),
                "advantages": torch.FloatTensor(self.advantages[batch_idx]),
                "returns": torch.FloatTensor(self.returns[batch_idx]),
            }

    def reset(self):
        self.ptr = 0
        self.size = 0
