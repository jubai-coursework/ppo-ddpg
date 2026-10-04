import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

from models.ddpg_networks import DDPGActor, DDPGCritic
from memory.replay_buffer import ReplayBuffer
from utils.config import DDPGConfig


class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, config: DDPGConfig,
                 action_low: float = -1.0, action_high: float = 1.0, device: str = "auto"):
        self.config = config
        self.action_low = action_low
        self.action_high = action_high

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Actor networks
        self.actor = DDPGActor(state_dim, action_dim, action_low=action_low, action_high=action_high).to(self.device)
        self.actor_target = DDPGActor(state_dim, action_dim, action_low=action_low, action_high=action_high).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)

        # Critic networks
        self.critic = DDPGCritic(state_dim, action_dim).to(self.device)
        self.critic_target = DDPGCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        # Replay buffer
        self.buffer = ReplayBuffer(config.buffer_size, state_dim, action_dim)

        # Noise
        self.noise_std = config.noise_std
        self.noise_decay = config.noise_decay
        self.min_noise = config.min_noise

    def select_action(self, state: np.ndarray, training: bool = True, add_noise: bool = True) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).cpu().numpy()

        if training and add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise
            action = np.clip(action, self.action_low, self.action_high)

        return action

    def decay_noise(self):
        self.noise_std = max(self.min_noise, self.noise_std * self.noise_decay)

    def push_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.buffer) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.config.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_value = rewards + (1 - dones) * self.config.gamma * target_q.squeeze()

        current_q = self.critic(states, actions).squeeze()
        critic_loss = nn.MSELoss()(current_q, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
        self.critic_optimizer.step()

        # Update Actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])