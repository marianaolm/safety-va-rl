from stable_baselines3 import PPO
from pathlib import Path

from src.trainers.sb3.trainer import train_sb3


def train(exp: dict, run_dir: Path):
    model_kwargs = {"device": exp.get("device", "cuda"),}

    for k in [
        "learning_rate",
        "n_steps",
        "batch_size",
        "gamma",
        "clip_range",
    ]:
        if k in exp:
            model_kwargs[k] = exp[k]

    train_sb3(
        model_class=PPO, 
        exp=exp,
        run_dir=run_dir,
        model_kwargs=model_kwargs,
    )