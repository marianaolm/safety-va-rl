class SACSpec:
    def __init__(self, env_id, timesteps, device, search_space):
        self.env_id = env_id
        self.timesteps = timesteps
        self.device = device
        self.search_space = search_space

    @classmethod
    def from_sweep_definition(cls, sweep):
        return cls(
            env_id=sweep["env_id"],
            timesteps=sweep["timesteps"],
            device=sweep.get("device", "cpu"),
            search_space=sweep["search_space"],
        )

    def sample_hyperparams(self, trial): 
        hp = {}
        for k, spec in self.search_space.items():
            kind, *vals = spec
            if kind == "loguniform":
                hp[k] = trial.suggest_float(k, vals[0], vals[1], log=True)
            elif kind == "uniform":
                hp[k] = trial.suggest_float(k, vals[0], vals[1])
            elif kind == "categorical":
                hp[k] = trial.suggest_categorical(k, vals[0])
        return hp

    def build_exp(self, hparams):
        return {
            "backend": "sb3",
            "algorithm": "sac",
            "env_id": self.env_id,
            "timesteps": self.timesteps,
            "device": self.device,
            **hparams,
        }

    def train(self, exp, run_dir):
        from src.trainers.sb3.sac import train
        train(exp, run_dir)
