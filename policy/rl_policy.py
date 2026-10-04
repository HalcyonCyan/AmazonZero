"""Use an Amazons RL-extension checkpoint with the existing MatchRunner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from policy.environment import AmazonsEnv
from policy.strategies import Policy
from train.rl_algorithms import DQNAgent, masked_distribution
from train.rl_extensions import load_agent


class LearnedRLPolicy(Policy):
    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        self.agent, self.config = load_agent(Path(checkpoint), device)
        if self.config.environment != "amazons":
            raise ValueError("LearnedRLPolicy requires an Amazons checkpoint")

    @torch.no_grad()
    def probabilities(self, env: AmazonsEnv, *, explore: bool = False) -> np.ndarray:
        if env.board_size != self.config.board_size:
            raise ValueError("Checkpoint board size does not match the environment")
        state = torch.as_tensor(env.encode().reshape(1, -1), device=self.agent.device)
        mask = torch.as_tensor(env.legal_mask()[None], device=self.agent.device)
        if isinstance(self.agent, DQNAgent):
            if not mask.any():
                raise ValueError("Cannot choose an action in a terminal state")
            action = self.agent.model(state).masked_fill(~mask, float("-inf")).argmax().item()
            result = np.zeros(env.action_size, dtype=np.float32)
            result[action] = 1.0
            return result
        logits, _ = self.agent.model(state)
        return masked_distribution(logits, mask).probs[0].cpu().numpy()
