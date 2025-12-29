from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


def plot_curve(x, y, title, ylabel, out_path: Path):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def find_progress_csv(run_dir: Path) -> Path:
    log_root = run_dir / "logs"
    if not log_root.exists():
        raise RuntimeError(f"Log directory not found: {log_root}")

    csv_files = list(log_root.rglob("progress.csv"))
    if not csv_files:
        raise RuntimeError(f"No progress.csv found under {log_root}")

    return csv_files[0]


def load_csv_columns(csv_path: Path, columns):
    data = {c: [] for c in columns}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for c in columns:
                if c in row and row[c] != "":
                    data[c].append(float(row[c]))

    return data


def run_omnisafe_training_metrics(run_dir: Path):
    out_dir = run_dir / "metrics"
    out_dir.mkdir(exist_ok=True)

    csv_path = find_progress_csv(run_dir)

    required_cols = [
        "TotalEnvSteps",
        "Metrics/EpRet",
        "Metrics/EpCost",
        "Metrics/EpLen",
    ]

    data = load_csv_columns(csv_path, required_cols)

    steps = np.array(data["TotalEnvSteps"])

    if len(steps) == 0:
        raise RuntimeError("No data found in progress.csv")

    if data["Metrics/EpRet"]:
        plot_curve(
            steps,
            np.array(data["Metrics/EpRet"]),
            "Episode return vs timesteps",
            "Episode return",
            out_dir / "episode_return.png",
        )

    if data["Metrics/EpCost"]:
        plot_curve(
            steps,
            np.array(data["Metrics/EpCost"]),
            "Episode safety cost vs timesteps",
            "Episode safety cost",
            out_dir / "episode_cost.png",
        )

    if data["Metrics/EpLen"]:
        plot_curve(
            steps,
            np.array(data["Metrics/EpLen"]),
            "Episode length vs timesteps",
            "Episode length",
            out_dir / "episode_length.png",
        )
