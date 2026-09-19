"""Policy/value network, value perspective rules, GPU training and checkpoints."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from policy.environment import AmazonsEnv
from policy.value import perspective_value


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class PolicyValueNet(nn.Module):
    """Compact AlphaZero-style residual network for an N x N board."""

    def __init__(
        self,
        board_size: int,
        channels: int = 64,
        residual_blocks: int = 4,
    ):
        super().__init__()
        self.board_size = board_size
        self.channels = channels
        self.residual_blocks = residual_blocks
        cells = board_size**2

        self.stem = nn.Sequential(
            nn.Conv2d(AmazonsEnv.INPUT_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(residual_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * cells, cells),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(cells, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, states: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.trunk(self.stem(states))
        policy_logits = self.policy_head(features)
        values = self.value_head(features).squeeze(1)
        return policy_logits, values

    def config(self) -> Dict[str, int]:
        return {
            "board_size": self.board_size,
            "channels": self.channels,
            "residual_blocks": self.residual_blocks,
        }


class NetworkEvaluator:
    """Centralized, masked policy and value inference for MCTS."""

    def __init__(
        self,
        model: PolicyValueNet,
        device: torch.device,
        use_amp: bool = True,
    ):
        self.model = model
        self.device = device
        self.use_amp = bool(use_amp and device.type == "cuda")

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def evaluate(self, env: AmazonsEnv) -> Tuple[np.ndarray, float]:
        """Return legal-action probabilities and value for current player."""
        if env.is_terminal():
            return np.zeros(env.action_size, dtype=np.float32), -1.0
        probabilities, values = self.evaluate_batch(
            env.encode()[None, ...], env.legal_mask()[None, ...]
        )
        return probabilities[0], float(values[0])

    def evaluate_batch(
        self,
        states: np.ndarray,
        legal_masks: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batched API ready for a future parallel self-play inference server."""
        if not np.all(legal_masks.any(axis=1)):
            raise ValueError("Every nonterminal inference state needs a legal action")

        state_tensor = torch.from_numpy(states).to(self.device)
        mask_tensor = torch.from_numpy(legal_masks).to(self.device)
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            logits, values = self.model(state_tensor)
            logits = logits.masked_fill(~mask_tensor, -1e9)
            probabilities = torch.softmax(logits, dim=1)
        return (
            probabilities.float().cpu().numpy(),
            values.float().cpu().numpy(),
        )


class ModelTrainer:
    """Owns optimizer, mixed-precision training, loss and checkpoint state."""

    def __init__(
        self,
        model: PolicyValueNet,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        # torch.amp is the current API; the fallback supports older PyTorch 2.x.
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    def train_batch(
        self,
        states: np.ndarray,
        target_policies: np.ndarray,
        legal_masks: np.ndarray,
        target_values: np.ndarray,
    ) -> Dict[str, float]:
        state_tensor = torch.from_numpy(states).to(self.device)
        target_policy_tensor = torch.from_numpy(target_policies).to(self.device)
        mask_tensor = torch.from_numpy(legal_masks).to(self.device)
        target_value_tensor = torch.from_numpy(target_values).to(self.device)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            logits, predicted_values = self.model(state_tensor)
            logits = logits.masked_fill(~mask_tensor, -1e9)
            log_policy = F.log_softmax(logits, dim=1)
            policy_loss = -(target_policy_tensor * log_policy).sum(dim=1).mean()
            value_loss = F.mse_loss(predicted_values, target_value_tensor)
            total_loss = policy_loss + value_loss

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss": float(total_loss.detach().item()),
            "policy_loss": float(policy_loss.detach().item()),
            "value_loss": float(value_loss.detach().item()),
            "gradient_norm": float(gradient_norm.detach().item()),
        }

    def save_checkpoint(
        self,
        path: str | Path,
        iteration: int,
        extra: Optional[Dict] = None,
    ) -> None:
        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "model_config": self.model.config(),
            "iteration": int(iteration),
            "extra": extra or {},
        }
        torch.save(checkpoint, Path(path))

    def load_checkpoint(self, path: str | Path) -> Dict:
        checkpoint = torch.load(Path(path), map_location=self.device)
        if checkpoint["model_config"] != self.model.config():
            raise ValueError(
                "Checkpoint model configuration does not match current model: "
                f"{checkpoint['model_config']} != {self.model.config()}"
            )
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scaler_state" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        return checkpoint


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device
