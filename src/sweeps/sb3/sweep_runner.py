from pathlib import Path
import json
import optuna

from src.experiments.registry import get_run_dir
from src.evaluation.final_eval.sb3_eval import run_sb3_final_eval


def run_sb3_sweep(study_name: str, storage_path: Path, spec, n_trials: int = 10, lambda_cost: float = 1.0):
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial):
        hparams = spec.sample_hyperparams(trial)
        exp = spec.build_exp(hparams)
        exp["device"] = exp.get("device", "cuda")

        base_dir = get_run_dir(exp)
        run_dir = base_dir / f"sweep_trial_{trial.number:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        spec.train(exp, run_dir)

        results = run_sb3_final_eval(exp, run_dir, n_episodes=30)

        reward = results["reward"]["mean"]
        cost = results["cost"]["mean"]
        objective_value = reward - lambda_cost * cost

        trial.set_user_attr("reward_mean", reward)
        trial.set_user_attr("cost_mean", cost)
        trial.set_user_attr("run_dir", str(run_dir))

        with open(run_dir / "sweep_trial.json", "w") as f:
            json.dump({
                "trial": trial.number,
                "hyperparameters": hparams,
                "reward_mean": reward,
                "cost_mean": cost,
                "objective": objective_value,
            }, f, indent=2)

        return objective_value

    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    summary = {
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
        "best_reward": study.best_trial.user_attrs["reward_mean"],
        "best_cost": study.best_trial.user_attrs["cost_mean"],
        "best_run_dir": study.best_trial.user_attrs["run_dir"],
    }

    summary_path = storage_path.parent / f"{study_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_path}")
