import os
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import imageio
import numpy as np
import torch
from torch import nn

from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.exercise1_q.q_table_exercise import QTable
from practice.utils.eval_utils import get_action
from practice.utils_for_coding.network_utils import load_model


def play_and_generate_video_generic(
    config: BaseConfig,
    ctx: ContextBase,
    save_video: bool = True,
) -> None:
    """Generic function to play game and generate video.

    Args:
        config: Configuration data
        ctx: ContextBase
        save_video: Whether to save the video.
    """
    # Load the trained model
    agent = _load_model_from_config(config, ctx)

    artifact_config = config.artifact_config

    # Get video path
    video_path = Path(artifact_config.output_dir) / artifact_config.replay_video_filename

    # Play the game and save video
    _play_game_once(
        env=ctx.eval_env,
        agent=agent,
        save_video=save_video,
        video_pathname=str(video_path),
        fps=artifact_config.fps,
        fps_skip=artifact_config.fps_skip,
        seed=artifact_config.seek_for_play,
    )


def _play_game_once(
    env: gym.Env[Any, Any],
    agent: nn.Module | QTable,
    save_video: bool = False,
    video_pathname: str = "",
    fps: int = 1,
    fps_skip: int = 1,
    seed: int = 100,
) -> None:
    """Play the game once with the random seed.

    Args:
        env (gym.Env): The environment.
        policy (AgentBase): The policy.
        save_video (bool): Whether to save the video.
        video_pathname (str): The path and name of the video.
        fps (int): The fps of the video.
        fps_skip (int): The frame rate to skip.
        seed (int): The seed for the replay video.
    """
    images: list[Any] = []
    state, _ = env.reset(seed=seed)
    if save_video:
        img_raw: Any = env.render()
        assert img_raw is not None, "The image is None, please check the environment for rendering."
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
        imageio.mimsave(video_pathname, [np.array(img) for img in images[::fps_skip]], fps=fps)
    print(f"Play Result: reward {reward_sum:.2f}, video saved to {video_pathname}")


def _load_model_from_config(cfg: BaseConfig, ctx: ContextBase) -> nn.Module | QTable:
    """Load the model from the config."""
    if cfg.artifact_config.play_full_model:
        full_model_path = Path(cfg.artifact_config.output_dir) / cfg.artifact_config.model_filename
        if not full_model_path.exists():
            raise FileNotFoundError(f"Model file not found: {full_model_path}")

        if isinstance(ctx.trained_target, QTable):
            return cast(QTable, torch.load(str(full_model_path)))

        return load_model(pathname=str(full_model_path), device=cfg.device, net=None)

    state_dict_path = Path(cfg.artifact_config.output_dir) / cfg.artifact_config.state_dict_filename
    if not state_dict_path.exists():
        raise FileNotFoundError(f"State dict file not found: {state_dict_path}")

    if isinstance(ctx.trained_target, QTable):
        return QTable.load_from_checkpoint(str(state_dict_path), device=None)

    return load_model(pathname=str(state_dict_path), device=cfg.device, net=ctx.network)
