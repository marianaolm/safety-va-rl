import numpy as np
import pygame
import safety_gymnasium
from stable_baselines3 import SAC


class KeyboardController:
    def __init__(self, action_space, throttle=0.3, turn=1.0):
        self.low = np.array(action_space.low, dtype=np.float32)
        self.high = np.array(action_space.high, dtype=np.float32)
        self.throttle = float(throttle)
        self.turn = float(turn)

    def get_action(self):
        keys = pygame.key.get_pressed()
        action = np.zeros_like(self.low, dtype=np.float32)

        # arrows: up/down forward/reverse, left/right turn
        if keys[pygame.K_UP]:
            action[0] = +self.throttle
        elif keys[pygame.K_DOWN]:
            action[0] = -self.throttle

        if keys[pygame.K_LEFT]:
            action[1] = +self.turn
        elif keys[pygame.K_RIGHT]:
            action[1] = -self.turn

        return np.clip(action, self.low, self.high)


class HumanOverrideWrapper:
    def __init__(self, env, human_controller):
        self.env = env
        self.human = human_controller

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, policy_action):
        pygame.event.pump()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_SPACE]:
            action = self.human.get_action()
            human_override = True
        else:
            action = policy_action
            human_override = False

        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["human_override"] = human_override
        info["cost"] = cost
        return obs, reward, cost, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def frame_to_surface(frame_rgb: np.ndarray) -> pygame.Surface:
    # frame_rgb: (H, W, 3) uint8 -> pygame surface expects (W, H, 3)
    frame_rgb = np.ascontiguousarray(frame_rgb)
    return pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))


def draw_label(screen, font, text):
    txt = font.render(text, True, (255, 255, 255))
    bg = pygame.Surface((txt.get_width() + 16, txt.get_height() + 10))
    bg.set_alpha(160)
    bg.fill((0, 0, 0))
    screen.blit(bg, (10, 10))
    screen.blit(txt, (18, 15))


def main():
    ENV_ID = "SafetyPointGoal2-v0"
    MODEL_PATH = "experiments/sac/SafetyPointGoal2-v0/n_timesteps_3500000/model/model"

    W, H = 1200, 900

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Safety-Gymnasium | SPACE = HUMAN override | ESC quit")
    font = pygame.font.SysFont(None, 36)

    env_raw = safety_gymnasium.make(
        ENV_ID,
        render_mode="rgb_array",
        width=W,
        height=H,
        camera_name="fixedfar",
    )

    human = KeyboardController(env_raw.action_space, throttle=0.3, turn=1.0)
    env = HumanOverrideWrapper(env_raw, human)

    model = SAC.load(MODEL_PATH, device="cpu")
    obs, _ = env.reset()

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        policy_action, _ = model.predict(obs, deterministic=True)
        obs, reward, cost, terminated, truncated, info = env.step(policy_action)

        if terminated or truncated:
            obs, _ = env.reset()

        frame = env.render()
        if frame is not None:
            screen.blit(frame_to_surface(frame), (0, 0))
            draw_label(screen, font, "HUMAN" if info.get("human_override") else "POLICY")
            pygame.display.flip()

        clock.tick(30)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
