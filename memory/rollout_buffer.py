import torch
import numpy as np


class RolloutBuffer:
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.full = False

        # Pre-allocate storage
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)

        # Computed returns and advantages
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)

    def add(self, state, action, reward, done, log_prob, value, next_state):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.next_states[self.ptr] = next_state

        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float, gamma: float, lam: float):
        """Compute GAE (Generalized Advantage Estimation)"""
        last_gae = 0

        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = last_gae + self.values[t]

        # Normalize advantages
        valid_advantages = self.advantages[:self.ptr]
        mean = np.mean(valid_advantages)
        std = np.std(valid_advantages)
        if std > 1e-8:
            self.advantages[:self.ptr] = (valid_advantages - mean) / (std + 1e-8)

    def get(self):
        """Return all data as tensors"""
        indices = slice(0, self.ptr)
        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.log_probs[indices]),
            torch.FloatTensor(self.returns[indices]),
            torch.FloatTensor(self.advantages[indices])
        )

    def clear(self):
        self.ptr = 0
        self.full = False

    def __len__(self):
        return self.ptr