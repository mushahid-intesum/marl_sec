import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from train.config import PPOConfig
from train.networks import ActorNetwork, CriticNetwork
from train.buffer import RolloutBuffer


class PPOAgent:

    def __init__(self, obs_dim: int, act_dim: int, config: PPOConfig):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.config = config

        self.actor = ActorNetwork(obs_dim, act_dim, config.hidden_sizes)
        self.critic = CriticNetwork(obs_dim, config.hidden_sizes)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.lr)

    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        action, log_prob = self.actor.get_action(obs, deterministic)
        with torch.no_grad():
            value = self.critic(torch.FloatTensor(obs).unsqueeze(0)).item()
        return action, log_prob, value

    def get_value(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            return self.critic(torch.FloatTensor(obs).unsqueeze(0)).item()

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.config.n_epochs):
            for batch in buffer.get_batches(self.config.batch_size):
                obs = batch["observations"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                new_log_probs, entropy = self.actor.evaluate_actions(obs, actions)
                ratio = torch.exp(new_log_probs - old_log_probs)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps,
                                    1.0 + self.config.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                self.actor_optim.zero_grad()
                actor_loss = policy_loss + self.config.entropy_coef * entropy_loss
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                self.actor_optim.step()

                values = self.critic(obs)
                value_loss = nn.functional.mse_loss(values, returns)

                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                self.critic_optim.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
