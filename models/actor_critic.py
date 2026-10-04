import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head
        self.actor_mu = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state):
        features = self.shared(state)
        mu = self.actor_mu(features)
        std = torch.exp(self.actor_log_std)
        value = self.critic(features)
        return mu, std, value

    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mu, std, value = self.forward(state)

        if deterministic:
            action = mu
        else:
            dist = Normal(mu, std)
            action = dist.sample()

        log_prob = self._compute_log_prob(mu, std, action)
        return action.squeeze(0).numpy(), log_prob.squeeze(0).numpy(), value.squeeze(0).numpy()

    def evaluate(self, states, actions):
        mu, std, values = self.forward(states)
        dist = Normal(mu, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, values.squeeze(-1), entropy

    def _compute_log_prob(self, mu, std, action):
        dist = Normal(mu, std)
        return dist.log_prob(action).sum(dim=-1)