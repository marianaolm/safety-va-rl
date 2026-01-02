import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.registry import get_experiment
from src.sweeps.sb3.sac_sweep import SACSpec
from src.sweeps.sb3.ppo_sweep import PPOSpec
from src.sweeps.sb3.sweep_runner import run_sb3_sweep


def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    sweep_name = sys.argv[1]
    sweep = get_experiment(sweep_name)

    if sweep["algorithm"] == "sac":
        spec = SACSpec.from_sweep_definition(sweep)
    elif sweep["algorithm"] == "ppo":
        spec = PPOSpec.from_sweep_definition(sweep)

    run_sb3_sweep(
        study_name=sweep_name,
        storage_path = (
            Path("experiments/sweeps")
            / sweep["algorithm"]
            / sweep["env_id"]
            / f"{sweep_name}.db"
        ),
        spec=spec,
        n_trials=sweep["n_trials"],
        lambda_cost=sweep.get("lambda_cost", 1.0),
    )


if __name__ == "__main__":
    main()
