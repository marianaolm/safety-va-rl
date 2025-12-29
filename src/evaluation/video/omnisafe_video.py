import imageio
import safety_gymnasium
import torch

from src.wrappers.GymCompatibilityWrapper import GymCompatibilityWrapper


def save_video_from_agent(agent, exp: dict, run_dir, exp_name: str):
    env_id = exp["env_id"]
    video_length = exp.get("video_length", 1000)

    env = safety_gymnasium.make(
        env_id,
        render_mode="rgb_array",
        camera_name="fixedfar",
    )
    env = GymCompatibilityWrapper(env)

    obs = env.reset()
    frames = []

    actor_critic = agent.agent._actor_critic
    device = next(actor_critic.parameters()).device

    for step in range(video_length):
        obs_tensor = (
            torch.as_tensor(obs, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            dist = actor_critic.actor(obs_tensor)
            action_tensor = dist.mean

        action = action_tensor.squeeze(0).cpu().numpy()

        obs, _, done, _ = env.step(action)
        frames.append(env.render())

        if done and step < video_length - 1:
            obs = env.reset()

    env.close()

    video_dir = run_dir / "videos"
    video_dir.mkdir(exist_ok=True)

    video_path = video_dir / f"{exp_name}.mp4"
    imageio.mimsave(video_path, frames, fps=30)

    print(f"[OmniSafe] Video saved to {video_path}")