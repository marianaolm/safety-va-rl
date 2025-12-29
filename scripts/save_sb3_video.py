import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.experiments.registry import get_experiment, get_run_dir
from src.evaluation.video.sb3_video import save_video


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/save_sb3_video.py <experiment_name>")
        sys.exit(1)

    exp_name = sys.argv[1]
    exp = get_experiment(exp_name)

    if exp.get("backend") != "sb3":
        raise RuntimeError("save_sb3_video.py only works for SB3 experiments")

    run_dir = get_run_dir(exp)
    save_video(exp_name, exp, run_dir)


if __name__ == "__main__":
    main()
