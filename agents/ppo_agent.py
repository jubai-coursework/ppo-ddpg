import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

from models.actor_critic import ActorCritic
from memory.rollout_buffer import RolloutBuffer
from utils.config import PPOConfig


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig, device: str = "auto"):
        self.config = config

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.lr)

        self.buffer = RolloutBuffer(config.rollout_steps, state_dim, action_dim)

    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, float, float]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mu, std, value = self.network(state_tensor)

        if not training:
            action = mu.squeeze(0).cpu().numpy()
            return action, 0.0, 0.0

        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, done, log_prob, value, next_state):
        self.buffer.add(state, action, reward, done, log_prob, value, next_state)

    def update(self, last_value: float):
        self.buffer.compute_returns_and_advantages(
            last_value, self.config.gamma, self.config.gae_lambda
        )

        states, actions, old_log_probs, returns, advantages = self.buffer.get()
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        dataset_size = states.shape[0]
        indices = np.arange(dataset_size)

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]

                log_probs, values, entropy = self.network.evaluate(batch_states, batch_actions)

                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values, batch_returns)

                entropy_loss = -entropy.mean()

                loss = (policy_loss +
                        self.config.value_coef * value_loss +
                        self.config.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

        self.buffer.clear()

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])