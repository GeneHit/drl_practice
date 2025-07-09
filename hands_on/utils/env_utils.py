from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Discrete
from numpy.typing import NDArray


def describe_wrappers(env: gym.Env[Any, Any]) -> list[str]:
    stack = []
    while hasattr(env, "env"):
        stack.append(type(env).__name__)
        env = env.env
    stack.append(type(env).__name__)  # base env
    return list(reversed(stack))


def make_image_env(
    env_id: str,
    render_mode: str,
    resize_shape: tuple[int, int],
    frame_stack_size: int,
) -> tuple[gym.Env[NDArray[Any], np.int64], dict[str, Any]]:
    """Make the 2D environment.

    env.action_space.n=np.int64(4)
    env.observation_space.shape=(4, 84, 84)
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.AddRenderObservation(env, render_only=True)
    env = gym.wrappers.ResizeObservation(env, shape=resize_shape)
    # -> [**shape, 3] -> [**shape, 1]
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
    # -> [**shape, 1] -> [4, **shape, 1]
    env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack_size)
    obs_shape = (frame_stack_size, *resize_shape)
    transposed_space = gym.spaces.Box(
        low=0, high=255, shape=obs_shape, dtype=np.uint8
    )
    # -> [num_stack, **shape, 1] -> [num_stack, **shape]
    env = gym.wrappers.TransformObservation(
        env,
        lambda obs: obs.squeeze(-1),
        observation_space=transposed_space,
    )
    # env = TransformObservation(env, lambda obs: np.transpose(obs, (2, 0, 1)))
    env = cast(gym.Env[NDArray[Any], np.int64], env)  # make mypy happy

    act_space = env.action_space
    assert isinstance(act_space, Discrete)  # make mypy happy
    env_info = {
        "wrappers": describe_wrappers(env),
        "observation_space.shape": env.observation_space.shape,
        "action_space": int(act_space.n),
        "observation_shape": obs_shape,
    }
    return env, env_info


def make_1d_env(
    env_id: str, render_mode: str | None = None
) -> tuple[gym.Env[NDArray[Any], np.int64], dict[str, Any]]:
    """Make the 1D environment.

    env.action_space.n=np.int64(4)
    env.observation_space.shape=(8,), np.int
    """
    env = gym.make(env_id, render_mode=render_mode)
    act_space = env.action_space
    assert isinstance(act_space, Discrete)  # make mypy happy
    env_info = {
        "observation_space.shape": env.observation_space.shape,
        "action_space": int(act_space.n),
    }
    return env, env_info


def make_discrete_env_with_kwargs(
    env_id: str, kwargs: dict[str, Any]
) -> tuple[gym.Env[np.int64, np.int64], dict[str, Any]]:
    """Make the environment based on configuration.

    observation:
        type: numpy.int64, obs.dtype: int64, obs.shape: (), obs_n: int
    action:
        type: numpy.int64, act.dtype: int64, act.shape: (), act_n: int
    """
    env = gym.make(id=env_id, **kwargs)

    act_space = env.action_space
    obs_space = env.observation_space
    assert isinstance(act_space, Discrete)  # make mypy happy
    assert isinstance(obs_space, Discrete)  # make mypy happy
    env_info = {
        "observation_space_n": int(obs_space.n),
        "action_space_n": int(act_space.n),
    }

    return env, env_info


def get_device() -> torch.device:
    """Get the device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    return device
