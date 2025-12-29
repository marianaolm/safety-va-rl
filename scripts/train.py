import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.registry import get_experiment, get_run_dir
from src.trainers.sb3 import sac, ppo


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/train.py <experiment_name>")
        sys.exit(1)

    exp_name = sys.argv[1]
    exp = get_experiment(exp_name)
    run_dir = get_run_dir(exp)

    if exp["algorithm"] == "sac":
        sac.train(exp, run_dir)
    elif exp["algorithm"] == "ppo":
        ppo.train(exp, run_dir)
    elif exp["algorithm"] == "ppolag":
        from src.trainers.omnisafe import ppolag
        ppolag.train(exp, run_dir, exp_name)
    else:
        raise ValueError(f"Unknown algorithm {exp['algorithm']}")


if __name__ == "__main__":
    main()
