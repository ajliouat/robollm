"""
Soft Actor-Critic (SAC) agent.

Clean from-scratch implementation following Haarnoja et al. 2018:
- Squashed Gaussian policy
- Twin Q-critics (clipped double-Q)
- Automatic entropy tuning (target entropy = -dim(A))
- Polyak-averaged target networks
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from policies.networks import Actor, TwinCritic
from training.replay_buffer import ReplayBuffer


@dataclass
class SACConfig:
    """Hyperparameters for SAC."""

    # Buffer / batch
    buffer_size: int = 1_000_000
    batch_size: int = 256
    warmup_steps: int = 5_000  # random actions before learning

    # Learning rates
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4

    # SAC parameters
    gamma: float = 0.99
    tau: float = 0.005  # Polyak averaging coefficient
    target_entropy: float | None = None  # None → -dim(A)

    # Network
    hidden_dim: int = 256

    # Training
    updates_per_step: int = 1

    # Device
    device: str = "cpu"


class SACAgent:
    """Soft Actor-Critic agent.

    Usage::

        agent = SACAgent(obs_dim=29, act_dim=4)
        action = agent.select_action(obs)
        agent.update(batch)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config: SACConfig | None = None,
    ):
        self.config = config or SACConfig()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = torch.device(self.config.device)

        # ── Networks ─────────────────────────────────────────────────────
        h = self.config.hidden_dim
        self.actor = Actor(obs_dim, act_dim, h).to(self.device)
        self.critic = TwinCritic(obs_dim, act_dim, h).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        # Freeze target params
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ── Optimizers ───────────────────────────────────────────────────
        self.actor_opt = Adam(
            self.actor.parameters(), lr=self.config.actor_lr,
        )
        self.critic_opt = Adam(
            self.critic.parameters(), lr=self.config.critic_lr,
        )

        # ── Entropy tuning ───────────────────────────────────────────────
        target_ent = self.config.target_entropy
        if target_ent is None:
            target_ent = -float(act_dim)  # heuristic: -dim(A)
        self.target_entropy = target_ent

        self.log_alpha = torch.tensor(
            0.0, dtype=torch.float32, device=self.device, requires_grad=True,
        )
        self.alpha_opt = Adam([self.log_alpha], lr=self.config.alpha_lr)

        # ── Replay buffer ────────────────────────────────────────────────
        self.buffer = ReplayBuffer(
            self.config.buffer_size, obs_dim, act_dim,
        )

        # ── Step counter ─────────────────────────────────────────────────
        self.total_steps = 0
        self._update_count = 0

    # ─── Properties ─────────────────────────────────────────────────────────

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    # ─── Action selection ───────────────────────────────────────────────────

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select action for a single observation.

        During warmup, returns uniform random action. Otherwise, samples
        from the actor (stochastic) or uses the mean (deterministic).
        """
        if self.total_steps < self.config.warmup_steps and not deterministic:
            return np.random.uniform(-1, 1, size=self.act_dim).astype(
                np.float32,
            )

        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(obs_t)
            else:
                action, _, _ = self.actor.sample(obs_t)

        return action.cpu().numpy().flatten()

    # ─── Training step ──────────────────────────────────────────────────────

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.add(obs, action, reward, next_obs, done)
        self.total_steps += 1

    def update(self) -> dict[str, float]:
        """Run one SAC update step. Returns loss metrics.

        Skips if buffer has fewer transitions than batch_size.
        """
        cfg = self.config
        if len(self.buffer) < cfg.batch_size:
            return {}

        metrics: dict[str, float] = {}

        for _ in range(cfg.updates_per_step):
            batch = self.buffer.sample(cfg.batch_size, self.device)
            m = self._update_step(batch)
            for k, v in m.items():
                metrics[k] = v

        return metrics

    def _update_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        alpha = self.log_alpha.exp().detach()

        # ── 1. Critic loss ───────────────────────────────────────────────
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_obs)
            q1_target, q2_target = self.critic_target(next_obs, next_action)
            q_target = torch.min(q1_target, q2_target)
            td_target = rewards + (1.0 - dones) * self.config.gamma * (
                q_target - alpha * next_log_prob
            )

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── 2. Actor loss ────────────────────────────────────────────────
        new_action, log_prob, _ = self.actor.sample(obs)
        q1_new = self.critic.q1_forward(obs, new_action)
        actor_loss = (alpha * log_prob - q1_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── 3. Alpha (entropy temperature) loss ──────────────────────────
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ── 4. Target network update (Polyak) ───────────────────────────
        with torch.no_grad():
            tau = self.config.tau
            for p, p_target in zip(
                self.critic.parameters(), self.critic_target.parameters(),
            ):
                p_target.data.mul_(1.0 - tau).add_(p.data * tau)

        self._update_count += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": alpha.item(),
            "q1_mean": q1.mean().item(),
            "log_prob_mean": log_prob.mean().item(),
        }

    # ─── Save / Load ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save all model weights and optimizer states."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "alpha_opt": self.alpha_opt.state_dict(),
                "total_steps": self.total_steps,
                "update_count": self._update_count,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load model weights and optimizer states."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_(True)
        self.alpha_opt = Adam([self.log_alpha], lr=self.config.alpha_lr)
        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        self.total_steps = ckpt["total_steps"]
        self._update_count = ckpt["update_count"]
