import os
import random
from typing import Any, List, Sequence

import gymnasium as gym
import imageio
import numpy as np
from tqdm import tqdm

from .base import PolicyBase


def evaluate_agent(
    env: gym.Env[Any, Any],
    policy: PolicyBase,
    max_steps: int,
    episodes: int,
    seed: Sequence[int],
    stream_video: bool = False,
    video_path: str = "",
) -> tuple[float, float]:
    """Evaluate the agent.

    Args:
        env (gym.Env): The environment.
        policy (PolicyBase): The policy.
        max_steps (int): The maximum number of steps per episode.
        episodes (int): The number of episodes to evaluate.
        seed (Sequence[int]): The seed.
        stream_video (bool): Whether to save the video on the file.
        video_path (str): The path to save the video.

    Returns:
        tuple[float, float]: The average and std of the reward.
    """
    if stream_video:
        assert video_path != "", (
            "video_path must be provided if stream_video is True"
        )
        if not video_path.endswith("/"):
            video_path = video_path + "/"

    rewards = []
    policy.set_train_flag(train_flag=False)
    for episode in tqdm(range(episodes)):
        if seed:
            state, _ = env.reset(seed=seed[episode])
        else:
            state, _ = env.reset()
        images: List[Any] = []
        if stream_video:
            img_raw: Any = env.render()
            assert img_raw is not None, (
                "The image is None, please check the environment for rendering."
            )
            images.append(img_raw)

        total_rewards_ep = 0.0
        for _ in range(max_steps):
            action = policy.action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_rewards_ep += float(reward)
            if stream_video:
                img_raw = env.render()
                assert img_raw is not None, (
                    "The image is None, please check the environment for rendering."
                )
                images.append(img_raw)
            if terminated or truncated:
                break

        if stream_video:
            imageio.mimsave(
                video_path + f"episode_{episode}.mp4",
                [np.array(img) for img in images],
                fps=1,
            )

        rewards.append(total_rewards_ep)
    return float(np.mean(rewards)), float(np.std(rewards))


def play_game_once(
    env: gym.Env[Any, Any],
    policy: PolicyBase,
    save_video: bool = False,
    video_pathname: str = "",
    fps: int = 1,
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
    state, _ = env.reset(seed=random.randint(0, 500))
    img_raw: Any = env.render()
    assert img_raw is not None, (
        "The image is None, please check the environment for rendering."
    )
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
        imageio.mimsave(
            video_pathname, [np.array(img) for img in images], fps=fps
        )
