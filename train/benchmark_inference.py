"""Measure real checkpoint inference and complete three-stage turns."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from model.inference import FrozenNetworkEvaluator
from model.model import NetworkEvaluator, PolicyValueNet, resolve_device
from policy.environment import AmazonsEnv
from policy.mcts import MCTS, MCTSConfig
from policy.strategies import MCTSPolicy


def summarize(times: list[float]) -> dict[str, float]:
    return {"median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def opening_positions(size: int, count: int, seed: int) -> list[AmazonsEnv]:
    rng = np.random.default_rng(seed)
    positions = [AmazonsEnv(size)]
    while len(positions) < count:
        env = AmazonsEnv(size)
        for _ in range(6):
            if env.is_terminal():
                break
            env.step(int(rng.choice(env.legal_actions())))
        if not env.is_terminal():
            positions.append(env)
    return positions


def complete_turn(env: AmazonsEnv, policy: MCTSPolicy) -> tuple[int, int, int]:
    policy.reset()
    actions = []
    for _ in range(3):
        action = policy.select_action(env, temperature=0.0)
        env.step(action)
        policy.observe_action(action)
        actions.append(action)
    return tuple(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--simulations", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="experiments/inference_benchmark.json")
    args = parser.parse_args()
    if min(args.cpu_threads, args.samples, args.positions, args.repeats, *args.simulations) < 1:
        parser.error("thread count, samples, positions, repeats and simulations must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = PolicyValueNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    positions = opening_positions(model.board_size, args.positions, args.seed)
    states = positions[0].encode()[None]
    masks = positions[0].legal_mask()[None]
    evaluators = {"eager": NetworkEvaluator(model, device, not args.no_amp)}
    started = time.perf_counter()
    evaluators["fused"] = FrozenNetworkEvaluator(model, device, not args.no_amp, use_cuda_graph=False)
    fusion_setup = time.perf_counter() - started
    graph_setup = None
    if device.type == "cuda":
        started = time.perf_counter()
        evaluators["cuda_graph"] = FrozenNetworkEvaluator(model, device, not args.no_amp)
        graph_setup = time.perf_counter() - started
    result = {
        "checkpoint": args.checkpoint,
        "iteration": checkpoint["iteration"],
        "model_config": model.config(),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "torch": str(torch.__version__),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": evaluators["eager"].use_amp,
        "cpu_threads": args.cpu_threads,
        "seed": args.seed,
        "samples": args.samples,
        "positions": args.positions,
        "repeats": args.repeats,
        "measurement_order": "Interleaved modes with rotating order to reduce GPU clock/order bias",
        "setup_seconds": {"fusion": fusion_setup, "cuda_graph": graph_setup},
        "inference_batch_1": {},
        "complete_turns": {},
        "output_errors": {},
    }
    # Compare all three stages with changing state and legal masks.
    for name, evaluator in evaluators.items():
        errors = {"policy_max_abs": 0.0, "value_max_abs": 0.0}
        for opening in positions:
            env = opening.clone()
            for _ in range(3):
                baseline_p, baseline_v = evaluators["eager"].evaluate(env)
                probabilities, value = evaluator.evaluate(env)
                errors["policy_max_abs"] = max(errors["policy_max_abs"], float(np.max(np.abs(baseline_p - probabilities))))
                errors["value_max_abs"] = max(errors["value_max_abs"], abs(baseline_v - value))
                if not np.isfinite(probabilities).all() or not np.isfinite(value):
                    raise RuntimeError(f"{name} produced nonfinite outputs")
                np.testing.assert_allclose(probabilities.sum(), 1.0, atol=1e-6)
                if np.any(probabilities[~env.legal_mask()] != 0):
                    raise RuntimeError(f"{name} assigned probability to illegal actions")
                env.step(int(np.argmax(baseline_p)))
        if max(errors.values()) > (5e-3 if result["amp"] else 1e-5):
            raise RuntimeError(f"{name} output error exceeds tolerance: {errors}")
        result["output_errors"][name] = errors
    names = list(evaluators)
    latency_samples = {name: [] for name in names}
    for _ in range(20):
        for evaluator in evaluators.values():
            evaluator.evaluate_batch(states, masks)
    for index in range(args.samples):
        offset = index % len(names)
        for name in names[offset:] + names[:offset]:
            started = time.perf_counter()
            evaluators[name].evaluate_batch(states, masks)
            latency_samples[name].append(1000 * (time.perf_counter() - started))
    for name in names:
        result["inference_batch_1"][name] = summarize(latency_samples[name])
        result["complete_turns"][name] = {}
    for simulations in args.simulations:
        policies = {
            name: MCTSPolicy(MCTS(evaluator, MCTSConfig(simulations=simulations)))
            for name, evaluator in evaluators.items()
        }
        for policy in policies.values():
            complete_turn(positions[0].clone(), policy)
        turn_samples = {name: [] for name in names}
        turn_actions = {name: [] for name in names}
        for repeat in range(args.repeats):
            for index, opening in enumerate(positions):
                offset = (repeat * len(positions) + index) % len(names)
                for name in names[offset:] + names[:offset]:
                    env = opening.clone()
                    started = time.perf_counter()
                    turn_actions[name].append(complete_turn(env, policies[name]))
                    turn_samples[name].append(1000 * (time.perf_counter() - started))
        for name in names:
            result["complete_turns"][name][str(simulations)] = {
                **summarize(turn_samples[name]), "actions": turn_actions[name],
            }
    for name in evaluators:
        latency = result["inference_batch_1"][name]
        latency["speedup_vs_eager"] = result["inference_batch_1"]["eager"]["median_ms"] / latency["median_ms"]
        for simulations in args.simulations:
            turn = result["complete_turns"][name][str(simulations)]
            baseline = result["complete_turns"]["eager"][str(simulations)]
            turn["speedup_vs_eager"] = baseline["median_ms"] / turn["median_ms"]
            turn["action_agreement_vs_eager"] = float(np.mean([
                actual == expected for actual, expected in zip(turn["actions"], baseline["actions"])
            ]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "complete_turns"}, ensure_ascii=False, indent=2))
    print("Complete turn medians (milliseconds):")
    for name, budgets in result["complete_turns"].items():
        for budget, metrics in budgets.items():
            print(f"  {name} simulations={budget}: {metrics['median_ms']:.3f} ms, "
                  f"speedup={metrics['speedup_vs_eager']:.2f}x, "
                  f"action_agreement={metrics['action_agreement_vs_eager']:.1%}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
