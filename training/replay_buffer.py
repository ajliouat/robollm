"""
Replay buffer for off-policy RL.

Fixed-size circular buffer storing (obs, action, reward, next_obs, done)
transitions. Efficient numpy storage with random batch sampling.
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Circular replay buffer with uniform random sampling.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions.
    obs_dim : int
        Observation dimensionality.
    act_dim : int
        Action dimensionality.
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.size = 0
        self._ptr = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        i = self._ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Sample a random batch and return as tensors."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idxs], device=device),
            "actions": torch.as_tensor(self.actions[idxs], device=device),
            "rewards": torch.as_tensor(self.rewards[idxs], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idxs], device=device),
            "dones": torch.as_tensor(self.dones[idxs], device=device),
        }

    def __len__(self) -> int:
        return self.size
