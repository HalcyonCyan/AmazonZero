"""Frozen Conv/BN fusion and CUDA Graph inference for repeated MCTS calls."""

from __future__ import annotations

import copy
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.utils import fuse_conv_bn_eval

from model.model import NetworkEvaluator, PolicyValueNet, ResidualBlock
from policy.environment import AmazonsEnv


def fused_inference_model(model: PolicyValueNet, device: torch.device) -> PolicyValueNet:
    """Build an independent eval-only snapshot; keep training weights and BN intact."""
    snapshot = copy.deepcopy(model).to(device).eval()
    snapshot.requires_grad_(False)
    for head in (snapshot.stem, snapshot.policy_head, snapshot.value_head):
        head[0] = fuse_conv_bn_eval(head[0], head[1])
        head[1] = nn.Identity()
    for block in snapshot.trunk:
        if isinstance(block, ResidualBlock):
            block.conv1 = fuse_conv_bn_eval(block.conv1, block.bn1)
            block.bn1 = nn.Identity()
            block.conv2 = fuse_conv_bn_eval(block.conv2, block.bn2)
            block.bn2 = nn.Identity()
    return snapshot


class FrozenNetworkEvaluator(NetworkEvaluator):
    """Single-caller inference snapshot, refreshed after each training iteration.

    Only graph_batch_size is captured; other sizes use fused eager inference.
    The graph's buffers are reused, so do not share this instance across threads.
    Rebuild this evaluator after updating the source model's weights or BN stats.
    """

    def __init__(
        self,
        model: PolicyValueNet,
        device: torch.device,
        use_amp: bool = True,
        *,
        use_cuda_graph: bool = True,
        graph_batch_size: int = 1,
    ):
        if graph_batch_size < 1:
            raise ValueError("graph_batch_size must be positive")
        super().__init__(fused_inference_model(model, device), device, use_amp)
        self.graph_batch_size = graph_batch_size
        self.graph = None
        if use_cuda_graph and device.type == "cuda":
            self._capture_graph()

    def _capture_graph(self) -> None:
        size = self.model.board_size
        with torch.cuda.device(self.device), torch.inference_mode():
            self._states = torch.zeros(
                (self.graph_batch_size, AmazonsEnv.INPUT_CHANNELS, size, size),
                dtype=torch.float32, device=self.device,
            )
            self._masks = torch.ones(
                (self.graph_batch_size, size**2), dtype=torch.bool, device=self.device
            )
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._forward_buffers()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self._probabilities, self._values = self._forward_buffers()
            torch.cuda.current_stream(self.device).wait_stream(stream)

    def _forward_buffers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Disable AMP's weight cache during capture: replay owns fixed buffers.
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp, cache_enabled=False):
            logits, values = self.model(self._states)
        logits = logits.float().masked_fill(~self._masks, -1e9)
        return torch.softmax(logits, dim=1), values.float()

    def evaluate_batch(
        self, states: np.ndarray, legal_masks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        expected = (len(states), AmazonsEnv.INPUT_CHANNELS, self.model.board_size, self.model.board_size)
        if states.shape != expected or states.dtype != np.float32:
            raise ValueError(f"states must be float32 with shape {expected}")
        if legal_masks.shape != (len(states), self.model.board_size**2) or legal_masks.dtype != np.bool_:
            raise ValueError("legal_masks must be boolean with one row per state")
        if len(states) == 0 or not np.all(legal_masks.any(axis=1)):
            raise ValueError("Every nonterminal inference state needs a legal action")
        if self.graph is None or len(states) != self.graph_batch_size:
            return super().evaluate_batch(states, legal_masks)
        with torch.cuda.device(self.device), torch.inference_mode():
            self._states.copy_(torch.from_numpy(np.ascontiguousarray(states)))
            self._masks.copy_(torch.from_numpy(np.ascontiguousarray(legal_masks)))
            self.graph.replay()
            # Fresh CPU copies prevent later replay from mutating previous results.
            return self._probabilities.cpu().numpy(), self._values.cpu().numpy()
