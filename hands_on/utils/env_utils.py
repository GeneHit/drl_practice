from typing import Any

import gymnasium as gym
import numpy as np
import torch


def extract_episode_data_from_infos(infos: dict[str, Any]) -> tuple[list[float], list[int]]:
    """Extract episode rewards and lengths from infos dictionary.

    This function works with the RecordEpisodeStatistics wrapper which provides
    episode data in vectorized format as numpy arrays.

    Args:
        infos: Info dictionary from environment step containing episode statistics

    Returns:
        Tuple of (episode_rewards, episode_lengths) as lists of floats and ints
        Returns empty lists if no episodes completed in this step
    """
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    # Check if episode statistics are available
    if "episode" in infos:
        # _r marks which environments completed episodes
        if "_r" in infos["episode"]:
            completed_mask = infos["episode"]["_r"]
            if np.any(completed_mask):
                # Get rewards and lengths for completed episodes
                completed_rewards = infos["episode"]["r"][completed_mask]
                completed_lengths = infos["episode"]["l"][completed_mask]

                # Convert numpy arrays to Python lists
                episode_rewards.extend(completed_rewards.tolist())
                episode_lengths.extend(completed_lengths.tolist())

    return episode_rewards, episode_lengths


def describe_wrappers(env: gym.Env[Any, Any]) -> list[str]:
    stack = []
    while hasattr(env, "env"):
        stack.append(type(env).__name__)
        env = env.env
    stack.append(type(env).__name__)  # base env
    return list(reversed(stack))


def get_device() -> torch.device:
    """Get the device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    return device
