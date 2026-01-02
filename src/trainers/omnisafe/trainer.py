from pathlib import Path

from src.trainers.omnisafe.compat import patch_linear_lr
patch_linear_lr()

import omnisafe


def train_omnisafe(
    algo: str,
    exp: dict,
    run_dir: Path,
    custom_cfgs: dict,
):
    """
    Generic OmniSafe trainer.
    """

    patch_linear_lr()

    env_id = exp["env_id"]

    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    custom_cfgs = dict(custom_cfgs) 
    
    custom_cfgs["logger_cfgs"] = {
        "log_dir": str(log_dir),
        "use_tensorboard": True,
        "use_wandb": False,
    }

    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs,)

    print(f"Training {algo} on {env_id}")
    agent.learn()

    return agent