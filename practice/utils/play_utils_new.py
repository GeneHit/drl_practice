import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import imageio
import numpy as np
from torch import nn

from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.exercise1_q.q_table_exercise import QTable
from practice.utils.eval_utils import get_action
from practice.utils_for_coding.network_utils import load_model


def play_and_generate_video_generic(config: BaseConfig, ctx: ContextBase) -> None:
    """Generic function to play game and generate video.

    Args:
        config: Configuration data
        ctx: ContextBase
    """
    # Load the trained model
    agent = _load_model_from_config(config, ctx)

    artifact_config = config.artifact_config

    # Get video path
    video_path = Path(artifact_config.output_dir) / artifact_config.replay_video_filename

    # Play the game and save video
    play_game_once(
        env=ctx.eval_env,
        agent=agent,
        save_video=True,
        video_pathname=str(video_path),
        fps=artifact_config.fps,
        seed=artifact_config.seed,
    )


def play_game_once(
    env: gym.Env[Any, Any],
    agent: nn.Module | QTable,
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

    reward_sum = 0.0
    while True:
        # the network has to decide the returned type of action.
        action = get_action(agent, state)
        state, reward, terminated, truncated, _ = env.step(action)
        reward_sum += float(reward)
        if save_video:
            img_raw = env.render()
            assert img_raw is not None, (
                "The image is None, please check the environment for rendering."
            )
            images.append(img_raw)
        if terminated or truncated:
            break

    # Save video if requested
    if save_video and video_pathname:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(video_pathname), exist_ok=True)
        imageio.mimsave(video_pathname, [np.array(img) for img in images], fps=fps)
    print(f"Play Result: reward {reward_sum:.2f}, video saved to {video_pathname}")


def _load_model_from_config(cfg: BaseConfig, ctx: ContextBase) -> nn.Module | QTable:
    model_path = Path(cfg.artifact_config.output_dir) / cfg.artifact_config.model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    path = str(model_path)
    if isinstance(ctx.trained_target, np.ndarray):
        return QTable.load_from_checkpoint(path, device=None)

    assert isinstance(ctx.trained_target, nn.Module)
    if cfg.artifact_config.play_full_model:
        return load_model(pathname=path, device=cfg.device, net=None)

    return load_model(pathname=path, device=cfg.device, net=ctx.network)
