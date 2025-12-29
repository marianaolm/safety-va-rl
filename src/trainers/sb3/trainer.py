import os
import safety_gymnasium
from pathlib import Path

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed

from src.wrappers.SafetyGymSB3Wrapper import SafetyGymSB3Wrapper
from src.trainers.sb3.callbacks import SafetyLoggingCallback


def make_env(env_id: str, rank: int, seed: int = 0):
    def _init():
        env = safety_gymnasium.make(env_id)
        env = SafetyGymSB3Wrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_sb3(model_class, exp: dict, run_dir: Path, model_kwargs: dict):
    """
    Generic SB3 trainer.
    """

    env_id = exp["env_id"]
    timesteps = exp["timesteps"]
    seed = exp.get("seed", 0)
    n_envs = exp.get("n_envs", int(os.environ.get("N_ENVS", "8")))

    set_random_seed(seed)

    # envs 
    env = SubprocVecEnv(
        [make_env(env_id, i, seed) for i in range(n_envs)],
        start_method="spawn",
    )
    env = VecMonitor(env)

    # dirs 
    log_dir = run_dir / "logs"
    model_dir = run_dir / "model"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    # model
    model = model_class(
        "MlpPolicy",
        env,
        tensorboard_log=str(log_dir),
        seed=seed,
        verbose=1,
        **model_kwargs,
    )

    model.learn(
        total_timesteps=timesteps,
        tb_log_name="train",
        callback=SafetyLoggingCallback(),
        progress_bar=False,
    )

    model.save(model_dir / "model.zip")
    env.close()
