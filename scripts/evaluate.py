import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.registry import get_experiment, get_run_dir


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluate.py <experiment_name> [metrics|final|all]")
        sys.exit(1)

    exp_name = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) >= 3 else "all"


    exp = get_experiment(exp_name)
    run_dir = get_run_dir(exp)

    backend = exp.get("backend", "sb3")

    if mode in ("metrics", "all"):
        if backend == "sb3":
            from src.evaluation.training_metrics.sb3_metrics import (run_sb3_training_metrics,)
            run_sb3_training_metrics(run_dir)

        elif backend == "omnisafe":
            from src.evaluation.training_metrics.omnisafe_metrics import (run_omnisafe_training_metrics,)
            run_omnisafe_training_metrics(run_dir)

        else:
            raise ValueError(f"Unknown backend: {backend}")

    if mode in ("final", "all"):
        if backend == "sb3":
            from src.evaluation.final_eval.sb3_eval import run_sb3_final_eval

            run_sb3_final_eval(exp, run_dir)

        elif backend == "omnisafe":
            from src.evaluation.final_eval.omnisafe_eval import run_omnisafe_final_eval

            run_omnisafe_final_eval(exp, run_dir)

        else:
            raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    main()
