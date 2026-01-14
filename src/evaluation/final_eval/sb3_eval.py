from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

import safety_gymnasium
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# from src.wrappers.SafetyGymSB3Wrapper import SafetyGymSB3Wrapper
from src.wrappers.FastSafeRewardWrapper import FastSafeRewardWrapper

def make_env(env_id: str):
    env = safety_gymnasium.make(env_id)
    env = FastSafeRewardWrapper(env)
    return env


def run_sb3_final_eval(exp: dict, run_dir: Path, n_episodes: int = 50):
    env_id = exp["env_id"]
    algo = exp["algorithm"]

    model_path = run_dir / "model" / "model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    env = DummyVecEnv([lambda: make_env(env_id)])

    if algo == "sac":
        model = SAC.load(model_path, env=env)
    elif algo == "ppo":
        model = PPO.load(model_path, env=env)
    else:
        raise ValueError(f"Unsupported SB3 algorithm: {algo}")

    episode_rewards = []
    episode_lengths = []
    episode_costs = []
    episode_successes = []

    for _ in range(n_episodes):
        obs = env.reset()
        done = False

        ep_reward = 0.0
        ep_length = 0
        ep_cost = 0.0
        ep_success = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)

            info = infos[0]
            done_flag = done[0]

            ep_reward += float(reward[0])
            ep_length += 1

            if "cost" in info:
                ep_cost += float(info["cost"])

            if info.get("goal_met", False) or info.get("is_success", False) or info.get("success", False):
                ep_success = 1.0

            if done_flag:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_costs.append(ep_cost)
        episode_successes.append(ep_success)

    env.close()

    rewards = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    costs = np.array(episode_costs)
    successes = np.array(episode_successes)

    results = {
        "environment": env_id,
        "algorithm": algo,
        "n_episodes": n_episodes,
        "reward": {
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
        },
        "episode_length": {
            "mean": float(lengths.mean()),
        },
        "cost": {
            "mean": float(costs.mean()),
            "max": float(costs.max()),
            "unsafe_rate": float((costs > 0).mean()),
        },
        "success_rate": float(successes.mean()),
    }

    out_dir = run_dir / "final_eval"
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


    plt.figure(figsize=(8, 6))
    plt.hist(rewards, bins=30, edgecolor="black")
    plt.xlabel("Episode return")
    plt.ylabel("Number of episodes")
    plt.title("Distribution of episode returns")
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(out_dir / "reward_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()


    plt.figure(figsize=(8, 6))
    plt.hist(costs, bins=30, edgecolor="black")
    plt.xlabel("Episode safety cost")
    plt.ylabel("Number of episodes")
    plt.title("Distribution of episode safety costs")
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(out_dir / "cost_histogram.png", dpi=200, bbox_inches="tight")
    plt.close()

    return results