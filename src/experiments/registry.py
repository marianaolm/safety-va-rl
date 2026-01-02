from pathlib import Path
from .definitions import EXPERIMENTS
from .sweep_definitions import SWEEPS


def get_experiment(name: str):
    if name in EXPERIMENTS:
        return EXPERIMENTS[name]
    if name in SWEEPS:
        return SWEEPS[name]
    raise ValueError(f"Unknown experiment or sweep: {name}")


def get_run_dir(exp):
    algo = exp["algorithm"]
    env = exp["env_id"]
    steps = exp["timesteps"]

    return (Path("experiments")/ algo/ env/ f"n_timesteps_{steps}")
