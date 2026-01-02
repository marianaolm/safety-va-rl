from stable_baselines3 import SAC
from pathlib import Path

from src.trainers.sb3.trainer import train_sb3


def train(exp: dict, run_dir: Path):
    model_kwargs = {"device": exp.get("device", "cuda"),}

    model_kwargs.setdefault("learning_starts", 1_000)
    model_kwargs.setdefault("train_freq", (1, "step"))
    model_kwargs.setdefault("gradient_steps", 1)
    model_kwargs.setdefault("batch_size", 256)

    for k in [
        "batch_size",
        "learning_rate",
        "tau",
        "gamma",
        "train_freq",
        "gradient_steps",
        "learning_starts",
    ]:
        if k in exp:
            model_kwargs[k] = exp[k]

    train_sb3(
        model_class=SAC,
        exp=exp,
        run_dir=run_dir,
        model_kwargs=model_kwargs,
    )