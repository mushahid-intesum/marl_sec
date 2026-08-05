import torch
import torch.nn as nn
import numpy as np
from typing import List


class ActorNetwork(nn.Module):

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        x = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
            log_prob = torch.log(probs[0, action]).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits = self.forward(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class CriticNetwork(nn.Module):

    def __init__(self, obs_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
