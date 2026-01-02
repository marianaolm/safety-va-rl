from stable_baselines3 import PPO
from pathlib import Path

from src.trainers.sb3.trainer import train_sb3


def train(exp: dict, run_dir: Path):
    train_sb3(
        model_class=PPO,
        exp=exp,
        run_dir=run_dir,
        model_kwargs={
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 256,
            "n_epochs": 10,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "seed": exp.get("seed", 0),
            "device": "cuda",
        },
    )
