import os
import numpy as np
import torch
from typing import Dict
from train.config import PPOConfig, TrainConfig
from train.ppo import PPOAgent
from train.buffer import RolloutBuffer
from envs.grid_nav import CooperativeGridNav
from envs.wrappers import CartPoleWrapper, SimpleSpreadWrapper
from envs.configs import CARTPOLE_CONFIG, GRID_NAV_CONFIG, SIMPLE_SPREAD_CONFIG


def train_cartpole(ppo_config: PPOConfig = None,
                   train_config: TrainConfig = None) -> Dict:
    if ppo_config is None:
        ppo_config = PPOConfig()
    if train_config is None:
        train_config = TrainConfig()

    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    env = CartPoleWrapper()
    agent = PPOAgent(CARTPOLE_CONFIG.obs_dim, CARTPOLE_CONFIG.act_dim, ppo_config)
    buffer = RolloutBuffer(train_config.rollout_steps, CARTPOLE_CONFIG.obs_dim)

    obs, _ = env.reset(seed=train_config.seed)
    current_obs = obs["agent_0"]
    timestep = 0
    episode_rewards = []
    ep_reward = 0.0

    while timestep < train_config.total_timesteps:
        buffer.reset()

        for _ in range(train_config.rollout_steps):
            action, log_prob, value = agent.get_action(current_obs)
            obs, rewards, terminated, truncated, _ = env.step({"agent_0": action})
            done = terminated["agent_0"] or truncated["agent_0"]

            buffer.add(current_obs, action, rewards["agent_0"], done, log_prob, value)
            ep_reward += rewards["agent_0"]
            timestep += 1

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                obs, _ = env.reset()
                current_obs = obs["agent_0"]
            else:
                current_obs = obs["agent_0"]

        last_value = agent.get_value(current_obs)
        buffer.compute_gae(last_value, ppo_config.gamma, ppo_config.gae_lambda)
        stats = agent.update(buffer)

        if timestep % train_config.log_interval < train_config.rollout_steps:
            recent = episode_rewards[-10:] if episode_rewards else [0]
            print(f"  t={timestep}, mean_reward={np.mean(recent):.1f}, "
                  f"policy_loss={stats['policy_loss']:.4f}, "
                  f"entropy={stats['entropy']:.4f}")

    env.close()

    os.makedirs(train_config.save_dir, exist_ok=True)
    save_path = os.path.join(train_config.save_dir, "cartpole_ppo.pt")
    agent.save(save_path)

    return {
        "agent": agent,
        "actor": agent.actor,
        "save_path": save_path,
        "episode_rewards": episode_rewards,
        "mean_reward": float(np.mean(episode_rewards[-20:])) if episode_rewards else 0.0,
    }


def _train_multiagent(env, config, agents, buffers, train_config, env_name):
    obs, _ = env.reset()
    agent_keys = list(obs.keys())
    current_obs = {a: obs[a] for a in agent_keys}
    timestep = 0
    episode_rewards = {a: [] for a in agent_keys}
    ep_reward = {a: 0.0 for a in agent_keys}

    while timestep < train_config.total_timesteps:
        for a in agent_keys:
            buffers[a].reset()

        for _ in range(train_config.rollout_steps):
            actions_dict = {}
            log_probs = {}
            values = {}

            for a in agent_keys:
                action, lp, val = agents[a].get_action(current_obs[a])
                actions_dict[a] = action
                log_probs[a] = lp
                values[a] = val

            obs, rewards, terminated, truncated, _ = env.step(actions_dict)
            done = any(terminated.values()) or any(truncated.values())

            for a in agent_keys:
                buffers[a].add(current_obs[a], actions_dict[a], rewards[a],
                               done, log_probs[a], values[a])
                ep_reward[a] += rewards[a]

            timestep += 1

            if done:
                for a in agent_keys:
                    episode_rewards[a].append(ep_reward[a])
                    ep_reward[a] = 0.0
                obs, _ = env.reset()

            current_obs = {a: obs[a] for a in agent_keys}

        for a in agent_keys:
            last_value = agents[a].get_value(current_obs[a])
            buffers[a].compute_gae(last_value, agents[a].config.gamma,
                                   agents[a].config.gae_lambda)
            agents[a].update(buffers[a])

        if timestep % train_config.log_interval < train_config.rollout_steps:
            for a in agent_keys:
                recent = episode_rewards[a][-10:] if episode_rewards[a] else [0]
                print(f"  {a}: t={timestep}, mean_reward={np.mean(recent):.2f}")

    os.makedirs(train_config.save_dir, exist_ok=True)
    result = {"agents": {}, "actors": {}, "save_paths": {}, "episode_rewards": episode_rewards}

    for a in agent_keys:
        save_path = os.path.join(train_config.save_dir, f"{env_name}_{a}.pt")
        agents[a].save(save_path)
        result["agents"][a] = agents[a]
        result["actors"][a] = agents[a].actor
        result["save_paths"][a] = save_path

    return result


def train_grid_nav(ppo_config: PPOConfig = None,
                   train_config: TrainConfig = None,
                   comm_enabled: bool = True) -> Dict:
    if ppo_config is None:
        ppo_config = PPOConfig()
    if train_config is None:
        train_config = TrainConfig()

    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    env = CooperativeGridNav(comm_enabled=comm_enabled)
    config = GRID_NAV_CONFIG

    agents = {}
    buffers = {}
    for a in env.agents:
        agents[a] = PPOAgent(config.obs_dim, config.act_dim, ppo_config)
        buffers[a] = RolloutBuffer(train_config.rollout_steps, config.obs_dim)

    return _train_multiagent(env, config, agents, buffers, train_config, "grid_nav")


def train_simple_spread(ppo_config: PPOConfig = None,
                        train_config: TrainConfig = None) -> Dict:
    if ppo_config is None:
        ppo_config = PPOConfig(hidden_sizes=[64, 64])
    if train_config is None:
        train_config = TrainConfig()

    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    env = SimpleSpreadWrapper()
    config = SIMPLE_SPREAD_CONFIG

    obs, _ = env.reset(seed=train_config.seed)
    agent_keys = list(obs.keys())

    agents = {}
    buffers = {}
    for a in agent_keys:
        agents[a] = PPOAgent(config.obs_dim, config.act_dim, ppo_config)
        buffers[a] = RolloutBuffer(train_config.rollout_steps, config.obs_dim)

    result = _train_multiagent(env, config, agents, buffers, train_config, "simple_spread")
    env.close()
    return result
