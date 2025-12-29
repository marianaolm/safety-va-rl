from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def load_tb_scalars(log_dir: Path):
    ea = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance={"scalars": 0},
    )
    ea.Reload()

    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        scalars[tag] = (steps, values)

    return scalars


def plot_curve(steps, values, title, ylabel, out_path: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(steps, values)
    plt.xlabel("Environment timesteps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def run_sb3_training_metrics(run_dir: Path):
    logs_dir = run_dir / "logs"
    out_dir = run_dir / "metrics"
    out_dir.mkdir(exist_ok=True)

    if not logs_dir.exists():
        raise RuntimeError(f"Log directory not found: {logs_dir}")

    run_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        raise RuntimeError(f"No TensorBoard runs found in {logs_dir}")

    log_root = run_dirs[0]

    scalars = load_tb_scalars(log_root)

    if "rollout/ep_rew_mean" in scalars:
        steps, values = scalars["rollout/ep_rew_mean"]
        plot_curve(
            steps,
            values,
            "Episode return vs timesteps",
            "Episode return",
            out_dir / "episode_return.png",
        )

    if "custom/episode_cost" in scalars:
        steps, values = scalars["custom/episode_cost"]
        plot_curve(
            steps,
            values,
            "Episode safety cost vs timesteps",
            "Episode safety cost",
            out_dir / "episode_cost.png",
        )

    if "custom/success_rate" in scalars:
        steps, values = scalars["custom/success_rate"]
        plot_curve(
            steps,
            values,
            "Success rate vs timesteps",
            "Success rate",
            out_dir / "success_rate.png",
        )

    reward_components = [
        k for k in scalars.keys()
        if k.startswith("custom/reward_")
    ]

    if reward_components:
        plt.figure(figsize=(10, 5))
        for k in reward_components:
            steps, values = scalars[k]
            plt.plot(steps, values, label=k.replace("custom/", ""))

        plt.xlabel("Environment timesteps")
        plt.ylabel("Reward component")
        plt.title("Reward decomposition vs timesteps")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            out_dir / "reward_decomposition.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

