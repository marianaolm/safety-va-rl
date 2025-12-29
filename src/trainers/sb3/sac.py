from stable_baselines3 import SAC
from pathlib import Path

from src.trainers.sb3.trainer import train_sb3


def train(exp: dict, run_dir: Path):
    train_sb3(
        model_class=SAC,
        exp=exp,
        run_dir=run_dir,
        model_kwargs={
            "device": "cuda",
            "batch_size": 2048,
            "learning_starts": 10_000,
            "train_freq": (16, "step"),
            "gradient_steps": 16,
        },
    )