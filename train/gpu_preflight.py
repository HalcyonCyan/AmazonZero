"""Verify real CUDA kernels, AMP updates, MCTS and checkpoint round trips."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from model.model import ModelTrainer, NetworkEvaluator, PolicyValueNet, resolve_device
from policy.environment import AmazonsEnv
from policy.mcts import MCTS, MCTSConfig


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(7)
    np.random.seed(7)
    device = resolve_device("cuda")
    torch.cuda.manual_seed_all(7)
    sample = torch.randn(16, 16, device=device)
    product = sample @ sample.T
    assert product.shape == (16, 16) and torch.isfinite(product).all()

    trainer = ModelTrainer(PolicyValueNet(4, 32, 2), device, learning_rate=3e-4)
    evaluator = NetworkEvaluator(trainer.model, device, use_amp=True)
    env = AmazonsEnv(4)
    states = np.repeat(env.encode()[None], 64, axis=0)
    masks = np.repeat(env.legal_mask()[None], 64, axis=0)
    targets = masks.astype(np.float32)
    targets /= targets.sum(axis=1, keepdims=True)
    values = np.ones(64, dtype=np.float32)
    before = trainer.model.stem[0].weight.detach().clone()
    for _ in range(2):
        metrics = trainer.train_batch(states, targets, masks, values)
        assert all(np.isfinite(value) for value in metrics.values()), metrics
    assert not torch.equal(before, trainer.model.stem[0].weight)
    assert all(float(state["step"]) == 2 for state in trainer.optimizer.state.values())
    probabilities, predictions = evaluator.evaluate_batch(states[:4], masks[:4])
    assert np.isfinite(probabilities).all() and np.isfinite(predictions).all()
    assert np.all(probabilities[~masks[:4]] == 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    root = MCTS(evaluator, MCTSConfig(simulations=4)).search(env)
    assert root.visit_count == 4

    destination = Path(".aris/compute").resolve()
    assert destination.is_relative_to(Path.cwd().resolve())
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as directory:
        temporary_root = Path(directory).resolve()
        assert temporary_root.is_relative_to(destination)
        checkpoint_path = temporary_root / "witness.pt"
        trainer.save_checkpoint(checkpoint_path, 1)
        restored = ModelTrainer(PolicyValueNet(4, 32, 2), device)
        restored.load_checkpoint(checkpoint_path)
        restored_probs, restored_values = NetworkEvaluator(
            restored.model, device, use_amp=True
        ).evaluate_batch(states[:4], masks[:4])
        np.testing.assert_allclose(restored_probs, probabilities, atol=1e-6)
        np.testing.assert_allclose(restored_values, predictions, atol=1e-6)
    torch.cuda.synchronize()
    result = {
        "torch": str(torch.__version__), "numpy": str(np.__version__),
        "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "amp": trainer.use_amp, "optimizer_steps": 2, "mcts_simulations": 4,
        "checkpoint_roundtrip": True, "metrics": metrics,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
    }
    (destination / "gpu-preflight.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("WITNESS CUDA_AMP_TRAIN_OK " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
