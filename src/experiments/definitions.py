EXPERIMENTS = {
    "sac_10k": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 10_000,
        "seed": 0,
        "n_envs": 8,
    },

    "sac_400k": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 400_000,
        "seed": 0,
        "n_envs": 8
    },

    "ppo_400k": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 400_000,
        "seed": 0,
        "n_envs": 8,
    },

    "sac_700k_goal1": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 700_000,
        "seed": 0,
        "n_envs": 8
    },

    "ppo_700k_goal1": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 700_000,
        "seed": 0,
        "n_envs": 8,
    },

    "sac_1m500k_goal1": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 1_500_000,
        "seed": 0,
        "n_envs": 8
    },

    "ppo_1m500k_goal1": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 1_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "sac_1m500k": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 1_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "sac_2m500k": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 2_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "sac_3m500k": {
        "backend": "sb3",
        "algorithm": "sac",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 3_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_1m500k": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 1_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_2m500k": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 2_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppo_3m500k": {
        "backend": "sb3",
        "algorithm": "ppo",
        "env_id": "SafetyPointGoal2-v0",
        "timesteps": 3_500_000,
        "seed": 0,
        "n_envs": 8,
    },

    "ppolag_5k": {
        "backend": "omnisafe",
        "algorithm": "ppolag",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 5000,
        "save_video": True,
        "video_length": 1000,
    },

    "ppolag_15k": {
        "backend": "omnisafe",
        "algorithm": "ppolag",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 15_000,
        "save_video": True,
        "video_length": 1000,
    },

    "ppolag_20k": {
        "backend": "omnisafe",
        "algorithm": "ppolag",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 20000,
        "save_video": True,
        "video_length": 1000,
    },

    "ppolag_100k": {
        "backend": "omnisafe",
        "algorithm": "ppolag",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 100_000,
        "save_video": True,
        "video_length": 1000,
    },

    "ppolag_500k": {
        "backend": "omnisafe",
        "algorithm": "ppolag",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 500_000,
        "save_video": True,
        "video_length": 1000,
    },

    "ppolag_1m500k": {
        "backend": "omnisafe",
        "algorithm": "ppolag",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 1_500_000,
        "save_video": True,
        "video_length": 2000,
    },

    "ppolag_3m500k": {
        "backend": "omnisafe",
        "algorithm": "ppolag",
        "env_id": "SafetyPointGoal1-v0",
        "timesteps": 3_500_000,
        "save_video": True,
        "video_length": 2000,
    }
}