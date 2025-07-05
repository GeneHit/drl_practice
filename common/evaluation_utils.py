import os
import random
from typing import Any, List, Sequence

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np

from .base import PolicyBase


def evaluate_agent(
    env: gym.Env[Any, Any],
    policy: PolicyBase,
    max_steps: int,
    episodes: int,
    seed: Sequence[int],
    stream_video: bool = False,
) -> tuple[float, float]:
    """Evaluate the agent.

    Args:
        env (gym.Env): The environment.
        policy (PolicyBase): The policy.
        max_steps (int): The maximum number of steps per episode.
        episodes (int): The number of episodes to evaluate.
        seed (Sequence[int]): The seed.
        stream_video (bool): Whether to show the env state (image) on the screen.

    Returns:
        tuple[float, float]: The average and std of the reward.
    """
    raise NotImplementedError("Not implemented")


def play_game_once(
    env: gym.Env[Any, Any],
    policy: PolicyBase,
    show_image: bool = False,
    save_video: bool = False,
    video_pathname: str = "",
    fps: int = 1,
) -> None:
    """Play the game once with the random seed.

    Args:
        env (gym.Env): The environment.
        policy (PolicyBase): The policy.
        show_image (bool): Whether to show the env state (image) on the screen in real time.
        save_video (bool): Whether to save the video.
        video_pathname (str): The path and name of the video.
        fps (int): The fps of the video.
    """
    images: List[Any] = []
    terminated = False
    truncated = False
    state, _ = env.reset(seed=random.randint(0, 500))
    img_raw: Any = env.render()
    assert img_raw is not None, (
        "The image is None, please check the environment for rendering."
    )
    img = img_raw

    # Initialize matplotlib figure if showing images
    if show_image:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        im = ax.imshow(img)
        ax.set_title("Game Environment")
        ax.axis("off")
        plt.show()

    if save_video:
        images.append(img)

    while not terminated and not truncated:
        # Take the action (index) that have the maximum expected future reward given that state
        action = policy.action(np.array(state))
        state, _, terminated, truncated, _ = env.step(action)
        img_raw = env.render()
        assert img_raw is not None, (
            "The image is None, please check the environment for rendering."
        )
        img = img_raw

        # Show image in real time if requested
        if show_image:
            im.set_array(img)
            plt.pause(0.01)  # Small pause to update the display

        # Collect images for video if requested
        if save_video:
            images.append(img)

    # Save video if requested
    if save_video and video_pathname:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(video_pathname), exist_ok=True)
        imageio.mimsave(
            video_pathname, [np.array(img) for img in images], fps=fps
        )

    # Close matplotlib figure if it was opened
    if show_image:
        plt.ioff()  # Turn off interactive mode
        plt.close(fig)
