import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio
import numpy as np

from practice.base.chest import AgentBase
from practice.base.config import BaseConfig
from practice.base.env_typing import EnvType


def play_and_generate_video_generic(config: BaseConfig, env: EnvType) -> None:
    """Generic function to play game and generate video.

    Args:
        config: Configuration data
        env: Environment
    """
    # Load the trained model
    model = _load_model_from_config(config)
    artifact_config = config.artifact_config

    # Get video path
    video_path = Path(artifact_config.output_dir) / artifact_config.replay_video_filename

    # Play the game and save video
    play_game_once(
        env=env,
        policy=model,
        save_video=True,
        video_pathname=str(video_path),
        fps=artifact_config.fps,
        seed=artifact_config.seek_for_play,
    )
    print(f"Game replay saved to: {video_path}")


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
    images: list[Any] = []
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


def _load_model_from_config(cfg: BaseConfig) -> AgentBase:
    model_path = Path(cfg.artifact_config.output_dir) / cfg.artifact_config.model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return cfg.artifact_config.agent_type.load_from_checkpoint(str(model_path), device=cfg.device)
