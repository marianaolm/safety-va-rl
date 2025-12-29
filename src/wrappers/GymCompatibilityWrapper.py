import gymnasium as gym


class GymCompatibilityWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, _ = out
            return obs
        return out

    def step(self, action):
        out = self.env.step(action)

        # Safety-Gymnasium API
        if len(out) == 6:
            obs, reward, cost, terminated, truncated, info = out
            done = terminated or truncated
            info = dict(info)
            info["cost"] = cost
            return obs, reward, done, info

        # Gymnasium API
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
            return obs, reward, done, info

        # Gym API
        obs, reward, done, info = out
        return obs, reward, done, info
