"""Readable DQN, Monte Carlo policy gradient, TD actor-critic and PPO updates."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


ALGORITHMS = ("dqn", "reinforce", "reinforce_baseline", "actor_critic", "ppo")


@dataclass(frozen=True)
class RLConfig:
    algorithm: str = "ppo"
    environment: str = "cartpole"
    seed: int = 7
    total_steps: int = 20_000
    board_size: int = 3
    opponent: str = "random"
    hidden_size: int = 64
    learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    rollout_steps: int = 512
    batch_size: int = 64
    replay_capacity: int = 10_000
    learning_starts: int = 256
    train_frequency: int = 4
    target_tau: float = 0.01
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.6
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    clip_ratio: float = 0.2
    target_kl: float = 0.02
    entropy_coefficient: float = 0.01
    evaluation_interval: int = 5_000
    evaluation_games: int = 20
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}")
        if self.environment not in ("cartpole", "amazons"):
            raise ValueError("environment must be cartpole or amazons")
        if self.board_size not in (3, 4, 5, 10) or self.opponent not in ("random", "near_opponent"):
            raise ValueError("Invalid board size or fixed opponent")
        for name in ("total_steps", "hidden_size", "rollout_steps", "batch_size", "replay_capacity",
                     "learning_starts", "train_frequency", "ppo_epochs", "evaluation_interval", "evaluation_games"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0 or self.evaluation_games % 2:
            raise ValueError("seed must be nonnegative; evaluation_games must be even to balance sides")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        for name in ("learning_rate", "critic_learning_rate", "target_kl"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("gamma", "gae_lambda", "epsilon_start", "epsilon_end"):
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("target_tau", "clip_ratio", "epsilon_decay_fraction"):
            if not np.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.epsilon_end > self.epsilon_start or not np.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0:
            raise ValueError("Invalid epsilon schedule or entropy coefficient")


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    mask: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    terminated: bool
    truncated: bool = False
    log_probability: float = 0.0
    value: float = 0.0
    next_value: float = 0.0


def mlp(observation_size: int, hidden_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(observation_size, hidden_size), nn.Tanh(),
                         nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                         nn.Linear(hidden_size, output_size))


class ActorCriticNet(nn.Module):
    """Separate actor/critic parameters keep the baseline out of actor gradients."""

    def __init__(self, observation_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.actor = mlp(observation_size, hidden_size, action_size)
        self.critic = mlp(observation_size, hidden_size, 1)
        for network in (self.actor, self.critic):
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=2**0.5)
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(states), self.critic(states).squeeze(-1)


def masked_distribution(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    if not masks.any(dim=-1).all():
        raise ValueError("A policy observation must have at least one legal action")
    return Categorical(logits=logits.masked_fill(~masks, float("-inf")))


@torch.no_grad()
def dqn_targets(rewards: torch.Tensor, terminated: torch.Tensor, next_q: torch.Tensor,
                next_masks: torch.Tensor, gamma: float) -> torch.Tensor:
    """Only actual termination disables bootstrapping, never a time limit."""
    active = ~terminated
    next_values = torch.zeros_like(rewards)
    if active.any():
        if not next_masks[active].any(dim=1).all():
            raise ValueError("Nonterminal next states must have legal actions")
        next_values[active] = next_q[active].masked_fill(~next_masks[active], float("-inf")).max(dim=1).values
    return rewards + gamma * next_values


def monte_carlo_returns(transitions: Sequence[Transition], gamma: float) -> np.ndarray:
    """Full sampled episode returns; a time limit defines the MC episode end."""
    if not transitions or not (transitions[-1].terminated or transitions[-1].truncated):
        raise ValueError("REINFORCE requires complete episodes")
    returns = np.empty(len(transitions), dtype=np.float32)
    running = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        item = transitions[index]
        if item.terminated or item.truncated:
            running = 0.0
        running = item.reward + gamma * running
        returns[index] = running
    return returns


def generalized_advantages(transitions: Sequence[Transition], gamma: float,
                           gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap final observations at truncation, but never cross a reset.

    At a rollout boundary the last delta still uses its own next_value. At an
    episode boundary the recursive trace stops even when bootstrapping is valid.
    """
    advantages = np.empty(len(transitions), dtype=np.float32)
    running = 0.0
    for index in range(len(transitions) - 1, -1, -1):
        item = transitions[index]
        delta = item.reward + gamma * (not item.terminated) * item.next_value - item.value
        continuation = not (item.terminated or item.truncated)
        running = delta + gamma * gae_lambda * continuation * running
        advantages[index] = running
    values = np.asarray([item.value for item in transitions], dtype=np.float32)
    return advantages, advantages + values


def ppo_surrogate(ratios: torch.Tensor, advantages: torch.Tensor, clip_ratio: float) -> torch.Tensor:
    return torch.minimum(ratios * advantages, ratios.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages)


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: list[Transition] = []
        self.index = 0

    def add(self, transition: Transition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.index] = transition
        self.index = (self.index + 1) % self.capacity

    def sample(self, size: int) -> list[Transition]:
        return random.sample(self.data, size)


class DQNAgent:
    def __init__(self, observation_size: int, action_size: int, config: RLConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = mlp(observation_size, config.hidden_size, action_size).to(self.device)
        self.target = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.memory = ReplayMemory(config.replay_capacity)

    def epsilon(self, step: int) -> float:
        fraction = min(1.0, step / (self.config.total_steps * self.config.epsilon_decay_fraction))
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    @torch.no_grad()
    def act(self, state: np.ndarray, mask: np.ndarray, *, step: int = 0,
            greedy: bool = False) -> tuple[int, float, float]:
        if not mask.any():
            raise ValueError("Cannot act without legal actions")
        if not greedy and random.random() < self.epsilon(step):
            return int(random.choice(np.flatnonzero(mask).tolist())), 0.0, 0.0
        q = self.model(torch.as_tensor(state, device=self.device).unsqueeze(0))[0]
        action = q.masked_fill(~torch.as_tensor(mask, device=self.device), float("-inf")).argmax()
        return int(action.item()), 0.0, 0.0

    def update(self) -> dict[str, float]:
        batch = self.memory.sample(self.config.batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), device=self.device)
        next_masks = torch.as_tensor(np.stack([item.next_mask for item in batch]), device=self.device)
        actions = torch.tensor([item.action for item in batch], device=self.device)
        rewards = torch.tensor([item.reward for item in batch], device=self.device)
        terminated = torch.tensor([item.terminated for item in batch], device=self.device)
        with torch.no_grad():
            targets = dqn_targets(rewards, terminated, self.target(next_states), next_masks, self.config.gamma)
        q = self.model(states).gather(1, actions[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target.parameters(), self.model.parameters()):
                target_parameter.lerp_(parameter, self.config.target_tau)
        return {"loss": loss.item(), "td_error": (q.detach() - targets).abs().mean().item(),
                "target_mean": targets.mean().item(), "q_mean": q.detach().mean().item(),
                "gradient_norm": float(gradient_norm)}


class PolicyGradientAgent:
    def __init__(self, observation_size: int, action_size: int, config: RLConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = ActorCriticNet(observation_size, action_size, config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam([
            {"params": self.model.actor.parameters(), "lr": config.learning_rate},
            {"params": self.model.critic.parameters(), "lr": config.critic_learning_rate},
        ])

    @torch.no_grad()
    def value(self, state: np.ndarray) -> float:
        state_tensor = torch.as_tensor(state, device=self.device).unsqueeze(0)
        return float(self.model.critic(state_tensor).item())

    @torch.no_grad()
    def act(self, state: np.ndarray, mask: np.ndarray, *, step: int = 0,
            greedy: bool = False) -> tuple[int, float, float]:
        logits, values = self.model(torch.as_tensor(state, device=self.device).unsqueeze(0))
        distribution = masked_distribution(logits, torch.as_tensor(mask, device=self.device).unsqueeze(0))
        action = distribution.probs.argmax(dim=-1) if greedy else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(values.item())

    def update(self, batch: Sequence[Transition]) -> dict[str, float]:
        config = self.config
        states = torch.as_tensor(np.stack([item.state for item in batch]), device=self.device)
        masks = torch.as_tensor(np.stack([item.mask for item in batch]), device=self.device)
        actions = torch.tensor([item.action for item in batch], device=self.device)
        old_log_probs = torch.tensor([item.log_probability for item in batch], device=self.device)
        rollout_values = np.asarray([item.value for item in batch], dtype=np.float32)
        if config.algorithm in ("reinforce", "reinforce_baseline"):
            returns = monte_carlo_returns(batch, config.gamma)
            advantages = returns if config.algorithm == "reinforce" else returns - rollout_values
        else:
            # TD(0) actor-critic is GAE with lambda=0; PPO uses a longer trace.
            advantages, returns = generalized_advantages(batch, config.gamma,
                0.0 if config.algorithm == "actor_critic" else config.gae_lambda)
        raw_variance = float(np.var(advantages))
        return_variance = float(np.var(returns))
        advantages_tensor = torch.as_tensor(advantages.copy(), device=self.device)
        if config.algorithm in ("actor_critic", "ppo") and len(batch) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / advantages_tensor.std(unbiased=False).clamp_min(1e-8)
        else:
            # Scale MC gradients without subtracting a hidden constant baseline.
            advantages_tensor /= max(1.0, float(np.std(advantages)))
        returns_tensor = torch.as_tensor(returns, device=self.device)
        epochs = config.ppo_epochs if config.algorithm == "ppo" else 1
        metrics: list[dict[str, float]] = []
        for _ in range(epochs):
            indices = torch.randperm(len(batch), device=self.device) if config.algorithm == "ppo" else torch.arange(len(batch), device=self.device)
            minibatch_size = config.batch_size if config.algorithm == "ppo" else len(batch)
            for selected in indices.split(minibatch_size):
                logits, values = self.model(states[selected])
                distribution = masked_distribution(logits, masks[selected])
                log_probs = distribution.log_prob(actions[selected])
                log_ratios = log_probs - old_log_probs[selected]
                ratios = log_ratios.exp()
                approx_kl = float(((ratios - 1) - log_ratios).detach().mean())
                if config.algorithm == "ppo" and approx_kl > config.target_kl:
                    return self._metrics(metrics, raw_variance, return_variance)
                if config.algorithm == "ppo":
                    actor_loss = -ppo_surrogate(ratios, advantages_tensor[selected], config.clip_ratio).mean()
                else:
                    actor_loss = -(log_probs * advantages_tensor[selected]).mean()
                critic_loss = torch.zeros((), device=self.device) if config.algorithm == "reinforce" else F.mse_loss(values, returns_tensor[selected])
                entropy = distribution.entropy().mean()
                loss = actor_loss + 0.5 * critic_loss - config.entropy_coefficient * entropy
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Clip each network independently; large critic errors cannot suppress actor updates.
                gradient_norm = nn.utils.clip_grad_norm_(self.model.actor.parameters(), 5.0)
                nn.utils.clip_grad_norm_(self.model.critic.parameters(), 5.0)
                self.optimizer.step()
                metrics.append({"loss": loss.item(), "policy_loss": actor_loss.item(),
                    "value_loss": critic_loss.item(), "entropy": entropy.item(), "approx_kl": approx_kl,
                    "clip_fraction": float(((ratios - 1).abs() > config.clip_ratio).float().mean().detach()),
                    "gradient_norm": float(gradient_norm)})
        return self._metrics(metrics, raw_variance, return_variance)

    @staticmethod
    def _metrics(metrics: list[dict[str, float]], advantage_variance: float,
                 return_variance: float) -> dict[str, float]:
        result = {key: float(np.mean([item[key] for item in metrics])) for key in metrics[0]} if metrics else {}
        result.update(advantage_variance=advantage_variance, return_variance=return_variance,
                      optimizer_steps=float(len(metrics)))
        return result


Agent = DQNAgent | PolicyGradientAgent


def create_agent(observation_size: int, action_size: int, config: RLConfig) -> Agent:
    agent_type = DQNAgent if config.algorithm == "dqn" else PolicyGradientAgent
    return agent_type(observation_size, action_size, config)
