"""Fixed, mixed and iteration-based curriculum opponent selection."""

from __future__ import annotations

import random
from dataclasses import dataclass

from policy.strategies import POLICY_NAMES


TRAINING_MODES = (*POLICY_NAMES, "mixed", "curriculum")


@dataclass(frozen=True)
class OpponentSchedule:
    mode: str = "self_play"
    stage_iterations: int = 5

    def __post_init__(self) -> None:
        if self.mode not in TRAINING_MODES:
            raise ValueError(f"Unknown training mode {self.mode!r}; choose from {TRAINING_MODES}")
        if self.stage_iterations < 1:
            raise ValueError("stage_iterations must be positive")

    def choose(self, iteration: int) -> str:
        """Choose once per game; absolute iterations preserve curriculum on resume."""
        if iteration < 1:
            raise ValueError("iteration must be positive")
        if self.mode == "mixed":
            return random.choice(POLICY_NAMES)
        if self.mode == "curriculum":
            index = min((iteration - 1) // self.stage_iterations, len(POLICY_NAMES) - 1)
            return POLICY_NAMES[index]
        return self.mode
