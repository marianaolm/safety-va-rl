from pathlib import Path

from src.trainers.omnisafe.trainer import train_omnisafe
from src.evaluation.video.omnisafe_video import save_video_from_agent


def train(exp: dict, run_dir: Path, exp_name: str):
    total_steps = exp["timesteps"]

    custom_cfgs = {
        "train_cfgs": {
            "total_steps": total_steps,
            "vector_env_nums": exp.get("n_envs", 4),
            "parallel": 1,
            "torch_threads": 1,
            "device": exp.get("device", 0),
        },
        "algo_cfgs": {
            "steps_per_epoch": 4000,
        },
        "lagrange_cfgs": {
            "cost_limit": exp.get("cost_limit", 100),
            "lambda_lr": exp.get("lambda_lr", 0.05),
        },
    }

    agent = train_omnisafe(
        algo="PPOLag",
        exp=exp,
        run_dir=run_dir,
        custom_cfgs=custom_cfgs,
    )

    if exp.get("save_video", False):
        save_video_from_agent(agent, exp, run_dir, exp_name)
