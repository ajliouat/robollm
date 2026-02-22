"""
MLP networks for Soft Actor-Critic.

Actor:  obs → 256 → 256 → (mean, log_std)  — squashed Gaussian
Critic: (obs, action) → 256 → 256 → Q-value  — twin Q networks
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class Actor(nn.Module):
    """Squashed Gaussian policy: maps observations to actions in [-1, 1]."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        """Sample action, return (action, log_prob, mean).

        Action is tanh-squashed. log_prob is corrected for tanh.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        # Reparameterization trick
        x_t = dist.rsample()
        action = torch.tanh(x_t)

        # Log prob with tanh correction
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, torch.tanh(mean)

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action (mean, squashed)."""
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class Critic(nn.Module):
    """Single Q-network: (obs, action) → scalar Q-value."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.q_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_head(x)


class TwinCritic(nn.Module):
    """Twin Q-networks for clipped double-Q trick."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = Critic(obs_dim, act_dim, hidden)
        self.q2 = Critic(obs_dim, act_dim, hidden)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        return self.q1(obs, action), self.q2(obs, action)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Only Q1 — used for policy gradient (no need for Q2)."""
        return self.q1(obs, action)
