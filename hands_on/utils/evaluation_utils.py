import os
import random
from typing import Any, List, Sequence

import gymnasium as gym
import imageio
import numpy as np
from tqdm import tqdm

from ..base import PolicyBase


def evaluate_agent(
    env: gym.Env[Any, Any],
    policy: PolicyBase,
    max_steps: int | None,
    episodes: int,
    seed: Sequence[int],
    stream_video: bool = False,
    video_folder: str = "./video",
) -> tuple[float, float]:
    """Evaluate the agent.

    Args:
        env (gym.Env): The environment.
        policy (PolicyBase): The policy.
        max_steps (int): The maximum number of steps per episode.
        episodes (int): The number of episodes to evaluate.
        seed (Sequence[int]): The seed.
        stream_video (bool): Whether to record videos during evaluation.
        video_folder (str): The folder to save the videos.

    Returns:
        tuple[float, float]: The average and std of the reward.
    """
    # Wrap the environment with RecordVideo if video recording is requested
    if stream_video:
        # Create the video folder if it doesn't exist
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(env, video_folder=video_folder)

    rewards = []
    policy.set_train_flag(train_flag=False)
    for episode in tqdm(range(episodes)):
        if seed:
            state, _ = env.reset(seed=seed[episode])
        else:
            state, _ = env.reset()

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
    policy: PolicyBase,
    save_video: bool = False,
    video_pathname: str = "",
    fps: int = 1,
    seed: int = random.randint(0, 500),
) -> None:
    """Play the game once with the random seed.

    Args:
        env (gym.Env): The environment.
        policy (PolicyBase): The policy.
        save_video (bool): Whether to save the video.
        video_pathname (str): The path and name of the video.
        fps (int): The fps of the video.
    """
    images: List[Any] = []
    policy.set_train_flag(train_flag=False)
    state, _ = env.reset(seed=seed)
    img_raw: Any = env.render()
    assert img_raw is not None, (
        "The image is None, please check the environment for rendering."
    )
    if save_video:
        images.append(img_raw)

    policy.set_train_flag(train_flag=False)
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
        imageio.mimsave(
            video_pathname, [np.array(img) for img in images], fps=fps
        )
