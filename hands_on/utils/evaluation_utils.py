import os
import random
from typing import Any, List

import gymnasium as gym
import imageio
import numpy as np
from tqdm import tqdm

from ..base import AgentBase


def evaluate_agent(
    env: gym.Env[Any, Any],
    policy: AgentBase,
    max_steps: int | None,
    episodes: int,
    seed: int | None,
    record_video: bool = False,
    video_dir: str = "./video",
    video_num: int | None = None,
) -> tuple[float, float]:
    """Evaluate the agent.

    Args:
        env (gym.Env): The environment.
        policy (AgentBase): The policy.
        max_steps (int): The maximum number of steps per episode.
        episodes (int): The number of episodes to evaluate.
        seed (Sequence[int]): The seed.
        record_video (bool): Whether to record videos during evaluation.
        video_dir (str): The directory to save the videos.

    Returns:
        tuple[float, float]: The average and std of the reward.
    """
    # Wrap the environment with RecordVideo if video recording is requested
    if record_video:
        # Create the video folder if it doesn't exist
        os.makedirs(video_dir, exist_ok=True)
        num = video_num if video_num is not None else 10
        trigger_step = episodes // num
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda x: x % trigger_step == 0,
            disable_logger=True,
        )

    rewards = []
    rnd = random.Random(seed)
    seeks = [rnd.randint(0, 1000) for _ in range(episodes)]
    for episode in tqdm(range(episodes), desc="Evaluating"):
        state, _ = env.reset(seed=seeks[episode])

        total_rewards_ep = 0.0

        def step() -> bool:
            nonlocal state, total_rewards_ep
            action = policy.action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_rewards_ep += float(reward)
            return terminated or truncated

        if max_steps is None:
            while not step():
                pass
        else:
            for _ in range(max_steps):
                if step():
                    break

        rewards.append(total_rewards_ep)

    return float(np.mean(rewards)), float(np.std(rewards))


def play_game_once(
    env: gym.Env[Any, Any],
    policy: AgentBase,
    save_video: bool = False,
    video_pathname: str = "",
    fps: int = 1,
    seed: int = 100,
) -> None:
    """Play the game once with the random seed.

    Args:
        env (gym.Env): The environment.
        policy (AgentBase): The policy.
        save_video (bool): Whether to save the video.
        video_pathname (str): The path and name of the video.
        fps (int): The fps of the video.
    """
    images: List[Any] = []
    state, _ = env.reset(seed=seed)
    img_raw: Any = env.render()
    assert img_raw is not None, "The image is None, please check the environment for rendering."
    if save_video:
        images.append(img_raw)

    terminated = truncated = False
    while not terminated and not truncated:
        # Take the action (index) that have the maximum expected future reward given that state
        action = policy.action(state)
        state, _, terminated, truncated, _ = env.step(action)
        if save_video:
            img_raw = env.render()
            assert img_raw is not None, (
                "The image is None, please check the environment for rendering."
            )
            images.append(img_raw)

    # Save video if requested
    if save_video and video_pathname:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(video_pathname), exist_ok=True)
        imageio.mimsave(video_pathname, [np.array(img) for img in images], fps=fps)
