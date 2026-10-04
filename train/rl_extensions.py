"""Run reproducible comparisons: python -m train.rl_extensions --help."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import TextIO

import numpy as np
import torch

from train.rl_algorithms import ALGORITHMS, Agent, DQNAgent, PolicyGradientAgent, RLConfig, Transition, create_agent
from train.rl_environments import make_environment


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def evaluate(agent: Agent, config: RLConfig, *, uniform: bool = False) -> dict[str, float]:
    """Fixed evaluation seeds, greedy learner, balanced Amazons first/second sides.

    Independent environment RNGs keep evaluation from changing training data.
    """
    opponents = ("random", "near_opponent") if config.environment == "amazons" else ("random",)
    results: dict[str, float] = {}
    for opponent in opponents:
        env = make_environment(config.environment, config.board_size, opponent)
        scores: list[float] = []
        try:
            for episode in range(config.evaluation_games):
                evaluation_seed = 100_000 + config.seed * 1_000 + episode
                rng = np.random.default_rng(evaluation_seed + 50_000)
                observation = env.reset(seed=evaluation_seed, learner_player=1 + episode % 2)
                score = 0.0
                while True:
                    action = int(rng.choice(np.flatnonzero(observation.mask))) if uniform else agent.act(observation.state, observation.mask, greedy=True)[0]
                    outcome = env.step(action)
                    score += outcome.reward
                    observation = outcome.observation
                    if outcome.terminated or outcome.truncated:
                        break
                scores.append(score)
        finally:
            env.close()
        prefix = opponent if config.environment == "amazons" else "cartpole"
        results[f"{prefix}_return_mean"] = float(np.mean(scores))
        results[f"{prefix}_return_std"] = float(np.std(scores, ddof=1))
        if config.environment == "amazons":
            wins = np.asarray(scores) > 0
            results[f"{prefix}_win_rate"] = float(wins.mean())
            results[f"{prefix}_first_win_rate"] = float(wins[::2].mean())
            results[f"{prefix}_second_win_rate"] = float(wins[1::2].mean())
    return results


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_json_line(stream: TextIO, payload: dict) -> None:
    stream.write(json.dumps(payload, allow_nan=False) + "\n")
    stream.flush()


def run_experiment(config: RLConfig, output: Path) -> dict:
    """MC methods finish the final episode; TD methods use the exact step budget."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Experiment directory is not empty: {output}; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    env = make_environment(config.environment, config.board_size, config.opponent)
    agent = create_agent(env.observation_size, env.action_size, config)
    dimensions = {"observation_size": env.observation_size, "action_size": env.action_size}
    software = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__}
    if config.environment == "cartpole":
        import gymnasium
        software["gymnasium"] = gymnasium.__version__
    write_json(output / "config.json", {**asdict(config), "software": software, **dimensions})
    start = time.perf_counter()
    initial = evaluate(agent, config)
    uniform = evaluate(agent, config, uniform=True)
    evaluations = [{"steps": 0, **initial}]
    rollout: list[Transition] = []
    completed_episodes = 0
    step = 0
    updates = 0
    optimizer_steps = 0
    episode_steps = 0
    episode_return = 0.0
    episode_scores: list[float] = []
    latest_metrics: dict[str, float] = {}
    next_evaluation = config.evaluation_interval
    observation = env.reset(seed=config.seed)
    monte_carlo = config.algorithm in ("reinforce", "reinforce_baseline")
    episode_finished = False
    try:
        with (output / "episodes.jsonl").open("w", encoding="utf-8") as episode_log, \
             (output / "updates.jsonl").open("w", encoding="utf-8") as update_log:
            while step < config.total_steps or (monte_carlo and not episode_finished):
                action, log_probability, value = agent.act(observation.state, observation.mask, step=step)
                outcome = env.step(action)
                next_observation = outcome.observation
                next_value = agent.value(next_observation.state) if isinstance(agent, PolicyGradientAgent) and not outcome.terminated else 0.0
                transition = Transition(observation.state, observation.mask, action, outcome.reward,
                    next_observation.state, next_observation.mask, outcome.terminated, outcome.truncated,
                    log_probability, value, next_value)
                step += 1
                episode_steps += 1
                episode_return += outcome.reward
                episode_finished = outcome.terminated or outcome.truncated
                did_update = False
                if isinstance(agent, DQNAgent):
                    agent.memory.add(transition)
                    if step >= max(config.learning_starts, config.batch_size) and step % config.train_frequency == 0:
                        latest_metrics = agent.update()
                        latest_metrics["epsilon"] = agent.epsilon(step)
                        did_update = True
                else:
                    rollout.append(transition)
                    ready = len(rollout) >= config.rollout_steps or step >= config.total_steps
                    if ready and (not monte_carlo or episode_finished):
                        latest_metrics = agent.update(rollout)
                        rollout.clear()
                        did_update = True
                if did_update:
                    updates += 1
                    optimizer_steps += int(latest_metrics.get("optimizer_steps", 1))
                    write_json_line(update_log, {"steps": step, "update": updates, **latest_metrics})
                if episode_finished:
                    completed_episodes += 1
                    episode_scores.append(episode_return)
                    write_json_line(episode_log, {"steps": step, "episode": completed_episodes,
                        "length": episode_steps, "return": episode_return,
                        "terminated": outcome.terminated, "truncated": outcome.truncated})
                    episode_steps = 0
                    episode_return = 0.0
                    if step < config.total_steps:
                        observation = env.reset(seed=config.seed + completed_episodes)
                else:
                    observation = next_observation
                # MC evaluation waits for a completed update; no incomplete episodes are fitted.
                if step >= next_evaluation and (not monte_carlo or episode_finished):
                    metrics = evaluate(agent, config)
                    evaluations.append({"steps": step, **metrics})
                    write_json(output / "evaluations.json", evaluations)
                    score_key = "random_win_rate" if config.environment == "amazons" else "cartpole_return_mean"
                    print(f"{config.environment}/{config.algorithm}/seed={config.seed} steps={step} "
                          f"eval={metrics[score_key]:.3f} episodes={completed_episodes}", flush=True)
                    next_evaluation = (step // config.evaluation_interval + 1) * config.evaluation_interval
    finally:
        env.close()
    final = evaluations[-1] if evaluations[-1]["steps"] == step else {"steps": step, **evaluate(agent, config)}
    if evaluations[-1]["steps"] != step:
        evaluations.append(final)
    write_json(output / "evaluations.json", evaluations)
    checkpoint = {"config": asdict(config), **dimensions, "steps": step,
        "model_state": agent.model.state_dict(), "optimizer_state": agent.optimizer.state_dict()}
    if isinstance(agent, DQNAgent):
        checkpoint["target_state"] = agent.target.state_dict()
    torch.save(checkpoint, output / "model.pt")
    elapsed = time.perf_counter() - start
    result = {"config": asdict(config), "software": software, "actual_steps": step,
        "episodes": completed_episodes, "updates": updates, "optimizer_steps": optimizer_steps,
        "elapsed_seconds": elapsed, "uniform_baseline": uniform, "initial": initial,
        "final": {key: value for key, value in final.items() if key != "steps"},
        "last_20_training_return": float(np.mean(episode_scores[-20:])) if episode_scores else None,
        "last_update": latest_metrics, "checkpoint": str(output / "model.pt")}
    write_json(output / "result.json", result)
    print(f"Saved {output} ({elapsed:.1f}s)", flush=True)
    return result


def load_agent(path: Path, device: str = "cpu") -> tuple[Agent, RLConfig]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = replace(RLConfig(**checkpoint["config"]), device=device)
    agent = create_agent(checkpoint["observation_size"], checkpoint["action_size"], config)
    agent.model.load_state_dict(checkpoint["model_state"])
    agent.model.eval()
    return agent, config


def summarize(results: list[dict], output: Path) -> None:
    rows: list[dict] = []
    for environment, algorithm in sorted({(item["config"]["environment"], item["config"]["algorithm"]) for item in results}):
        group = [item for item in results if item["config"]["environment"] == environment and item["config"]["algorithm"] == algorithm]
        row: dict = {"environment": environment, "algorithm": algorithm, "seeds": len(group)}
        for key in group[0]["final"]:
            # Between-seed SD of evaluation means is distinct from within-episode SD.
            if key.endswith("_std"):
                continue
            for label in ("uniform_baseline", "initial", "final"):
                values = [item[label][key] for item in group]
                row[f"{label}_{key}"] = float(np.mean(values))
                row[f"{label}_{key}_seed_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row["mean_actual_steps"] = float(np.mean([item["actual_steps"] for item in group]))
        row["mean_seconds"] = float(np.mean([item["elapsed_seconds"] for item in group]))
        rows.append(row)
    write_json(output / "summary.json", {"runs": results, "aggregate": rows})
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    plot_results(results, output)


def plot_results(results: list[dict], output: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install requirements-rl.txt to also export learning_curves.png", flush=True)
        return
    environments = sorted({item["config"]["environment"] for item in results})
    fig, axes = plt.subplots(1, len(environments), figsize=(7 * len(environments), 4.5), squeeze=False)
    for axis, environment in zip(axes[0], environments):
        metric = "random_win_rate" if environment == "amazons" else "cartpole_return_mean"
        for algorithm in ALGORITHMS:
            group = [item for item in results if item["config"]["environment"] == environment and item["config"]["algorithm"] == algorithm]
            if not group:
                continue
            grid = np.linspace(0, min(item["actual_steps"] for item in group), 100)
            curves = []
            for item in group:
                evaluations = json.loads((Path(item["checkpoint"]).parent / "evaluations.json").read_text(encoding="utf-8"))
                curves.append(np.interp(grid, [point["steps"] for point in evaluations], [point[metric] for point in evaluations]))
            mean = np.mean(curves, axis=0)
            std = np.std(curves, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
            line, = axis.plot(grid, mean, label=algorithm)
            axis.fill_between(grid, mean - std, mean + std, alpha=0.12, color=line.get_color())
        baseline = np.mean([item["uniform_baseline"][metric] for item in results if item["config"]["environment"] == environment])
        axis.axhline(baseline, color="gray", linestyle="--", label="uniform baseline")
        axis.set(title=f"{environment}: mean +/- seed SD", xlabel="Learner environment steps",
                 ylabel="Win rate vs random" if environment == "amazons" else "Greedy evaluation return")
        if environment == "amazons":
            axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=("cartpole", "amazons", "both"), default="cartpole")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 42])
    parser.add_argument("--total-steps", type=int, default=20_000)
    parser.add_argument("--board-size", type=int, choices=(3, 4, 5, 10), default=3)
    parser.add_argument("--opponent", choices=("random", "near_opponent"), default="random")
    parser.add_argument("--output", type=Path, default=Path("experiments/rl_extensions"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--evaluate-checkpoint", type=Path)
    defaults = RLConfig()
    for option, field_type in (("hidden_size", int), ("learning_rate", float), ("critic_learning_rate", float),
        ("gamma", float), ("rollout_steps", int), ("batch_size", int), ("replay_capacity", int),
        ("learning_starts", int), ("train_frequency", int), ("target_tau", float),
        ("epsilon_start", float), ("epsilon_end", float), ("epsilon_decay_fraction", float),
        ("gae_lambda", float), ("ppo_epochs", int), ("clip_ratio", float), ("target_kl", float),
        ("entropy_coefficient", float), ("evaluation_interval", int), ("evaluation_games", int)):
        parser.add_argument("--" + option.replace("_", "-"), dest=option, type=field_type, default=getattr(defaults, option))
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.algorithms)) != len(args.algorithms):
        parser.error("Do not repeat seeds or algorithms")
    torch.set_num_threads(args.torch_threads)
    if args.evaluate_checkpoint:
        agent, config = load_agent(args.evaluate_checkpoint, args.device)
        config = replace(config, evaluation_games=args.evaluation_games)
        print(json.dumps(evaluate(agent, config), indent=2))
        return
    excluded = {"env", "algorithms", "seeds", "output", "torch_threads", "evaluate_checkpoint"}
    kwargs = {key: value for key, value in vars(args).items() if key not in excluded}
    environments = ("cartpole", "amazons") if args.env == "both" else (args.env,)
    configurations = [RLConfig(environment=environment, algorithm=algorithm, seed=seed, **kwargs)
        for environment in environments for algorithm in args.algorithms for seed in args.seeds]
    # Validate all output directories before starting any expensive training.
    for config in configurations:
        destination = args.output / config.environment / config.algorithm / f"seed_{config.seed}"
        if destination.exists() and any(destination.iterdir()):
            parser.error(f"Experiment directory is not empty: {destination}; choose a new --output")
    results = []
    for config in configurations:
        destination = args.output / config.environment / config.algorithm / f"seed_{config.seed}"
        results.append(run_experiment(config, destination))
        # Partial summaries survive if a later run is interrupted.
        summarize(results, args.output)


if __name__ == "__main__":
    main()
