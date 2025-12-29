from stable_baselines3.common.callbacks import BaseCallback


class SafetyLoggingCallback(BaseCallback):
    """
    Logs safety cost and success rate for Safety-Gymnasium environments.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_costs = []
        self.episode_successes = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]

        for i, info in enumerate(infos):
            if "cost" in info:
                self.episode_costs.append(info["cost"])

            if dones[i]:
                success = float(info.get("goal_met", False))
                self.episode_successes.append(success)

        return True

    def _on_rollout_end(self) -> None:
        if self.episode_costs:
            self.logger.record(
                "custom/episode_cost",
                sum(self.episode_costs) / len(self.episode_costs),
            )
            self.episode_costs.clear()

        if self.episode_successes:
            self.logger.record(
                "custom/success_rate",
                sum(self.episode_successes) / len(self.episode_successes),
            )
            self.episode_successes.clear()
