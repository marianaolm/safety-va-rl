import numpy as np
import pygame
import safety_gymnasium
from stable_baselines3 import SAC 


# ---------------------------
# Keyboard controller
# ---------------------------
class KeyboardController:
    def __init__(self):
        pygame.init()
        pygame.display.set_mode((200, 200))  # required to receive key events

    def get_action(self):
        keys = pygame.key.get_pressed()
        action = np.zeros(2, dtype=np.float32)

        if keys[pygame.K_w]:
            action[1] += 1.0
        if keys[pygame.K_s]:
            action[1] -= 1.0
        if keys[pygame.K_a]:
            action[0] -= 1.0
        if keys[pygame.K_d]:
            action[0] += 1.0

        return np.clip(action, -1.0, 1.0)


# ---------------------------
# Human override wrapper
# ---------------------------
class HumanOverrideWrapper:
    def __init__(self, env, human_controller):
        self.env = env
        self.human = human_controller

    def reset(self):
        return self.env.reset()

    def step(self, policy_action):
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        # Hold SPACE to override
        if keys[pygame.K_SPACE]:
            action = self.human.get_action()
            info_override = True
        else:
            action = policy_action
            info_override = False

        obs, reward, terminated, truncated, info = self.env.step(action)
        info["human_override"] = info_override
        return obs, reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


# ---------------------------
# Main loop
# ---------------------------
def main():
    ENV_ID = "SafetyPointGoal2-v0"
    MODEL_PATH = "experiments/sac/SafetyPointGoal2-v0/n_timesteps_3500000/model/model.zip" 

    env = safety_gymnasium.make(
        ENV_ID,
        render_mode="human",
    )

    human = KeyboardController()
    env = HumanOverrideWrapper(env, human)

    model = SAC.load(MODEL_PATH, device="cpu")

    obs, _ = env.reset()

    print(
        "\nControls:\n"
        "  W/A/S/D : move\n"
        "  SPACE   : hold for human override\n"
        "  ESC     : quit\n"
    )

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("human_override", False):
            print("Human override")

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
