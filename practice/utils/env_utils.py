import inspect
import json
from typing import Any, Callable, TypeAlias, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from numpy.typing import NDArray

from practice.base.config import EnvConfig
from practice.base.env_typing import ActType, EnvsType, EnvsTypeC, EnvType, EnvTypeC

ObsInt: TypeAlias = np.uint8


def get_device(target: str | None = None) -> torch.device:
    """Get the device.

    Args:
        target: The target device.
            - "cpu": CPU
            - "cuda": GPU
            - "mps": Apple Silicon GPU
            - None: Auto-detect the best device

    Returns:
        The device.
    """
    if target is not None:
        return torch.device(target)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


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


def make_discrete_env_with_kwargs(
    env_id: str, kwargs: dict[str, Any], max_steps: int | None = None
) -> gym.Env[np.int64, np.int64]:
    """Make the environment based on configuration.

    observation:
        type: numpy.int64, obs.dtype: int64, obs.shape: (), obs_n: int
    action:
        type: numpy.int64, act.dtype: int64, act.shape: (), act_n: int
    """
    env = gym.make(id=env_id, **kwargs)
    # Add episode statistics tracking - tracks cumulative rewards and episode lengths
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if max_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

    return env


def make_env_with_kwargs(
    env_id: str,
    max_steps: int | None = None,
    kwargs: dict[str, Any] = {},
    normalize_obs: bool = False,
) -> EnvType:
    """Make the environment based on configuration.

    observation:
        type: numpy.float32, obs.dtype: float32, obs.shape: (n, )
    action:
        type: numpy.int64, act.dtype: int64, act.shape: (), act_n: int
    """
    env = gym.make(id=env_id, **kwargs)
    # Add episode statistics tracking - tracks cumulative rewards and episode lengths
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if max_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.DtypeObservation(env, dtype=np.float32)

    return env


def make_image_env_for_vectorized(
    env_id: str,
    render_mode: str,
    resize_shape: tuple[int, int],
    frame_stack_size: int,
    frame_skip: int = 1,
    max_steps: int | None = None,
    normalize_obs: bool = False,
) -> gym.Env[NDArray[ObsInt], ActType]:
    """Make the 2D environment.

    env.action_space.n=np.int64(4)
    env.observation_space.shape=(4, 84, 84)
    """
    env = gym.make(env_id, render_mode=render_mode)
    # Add episode statistics tracking - tracks cumulative rewards and episode lengths
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # Add auto-reset wrapper - provides terminal_observation for final observations
    env = gym.wrappers.Autoreset(env)
    env = gym.wrappers.AddRenderObservation(env, render_only=True)
    env = gym.wrappers.ResizeObservation(env, shape=resize_shape)
    # -> [**shape, 3] -> [**shape, 1]
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
    # -> [**shape, 1] -> [4, **shape, 1]
    env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack_size)
    obs_shape = (frame_stack_size, *resize_shape)
    transposed_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=ObsInt)
    # -> [num_stack, **shape, 1] -> [num_stack, **shape]
    env = gym.wrappers.TransformObservation(
        env,
        lambda obs: obs.squeeze(-1),
        observation_space=transposed_space,
    )
    if frame_skip > 1:
        env = gym.wrappers.MaxAndSkipObservation(env, skip=frame_skip)
    if max_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
    # env = TransformObservation(env, lambda obs: np.transpose(obs, (2, 0, 1)))
    return cast(gym.Env[NDArray[ObsInt], ActType], env)


def make_1d_env_for_vectorized(
    env_id: str,
    render_mode: str | None = None,
    max_steps: int | None = None,
    normalize_obs: bool = False,
) -> gym.Env[NDArray[np.float32], ActType]:
    """Make the 1D environment.

    Args:
        env_id: The environment ID to create
        render_mode: The render mode for the environment
        max_steps: Optional maximum steps per episode. If provided, wraps environment with TimeLimit

    Returns:
        Tuple of (environment, environment info dict)

    env.action_space.n=np.int64(4)
    env.observation_space.shape=(8,), np.float32
    """
    env = gym.make(env_id, render_mode=render_mode)
    # Add episode statistics tracking - tracks cumulative rewards and episode lengths
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # Add auto-reset wrapper - provides terminal_observation for final observations
    env = gym.wrappers.Autoreset(env)

    # Add time limit wrapper if max_steps is specified
    if max_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    if normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.DtypeObservation(env, dtype=np.float32)

    return cast(gym.Env[NDArray[np.float32], ActType], env)


def _make_vector_env(
    env_fn: Callable[[], EnvType],
    num_envs: int,
    use_multi_processing: bool = False,
) -> EnvsType:
    """Create a vector environment."""
    if use_multi_processing and num_envs > 1:
        return gym.vector.AsyncVectorEnv([env_fn for _ in range(num_envs)])

    return gym.vector.SyncVectorEnv([env_fn for _ in range(num_envs)])


def get_env_from_config(config: EnvConfig) -> tuple[EnvType | EnvsType, EnvType]:
    """Get the environment from the configuration.

    Returns:
        the training environments: EnvType | EnvsType
        the evaluation environment: EnvType.
            Most setup is same with training environments, but always use rgb_array
            render mode for evaluation:
            1. The evaluation episode is much smaller than training episodes.
            2. we always want a replay video for pushing to hub.
    """
    if config.vector_env_num is None:
        assert not config.use_image, "single environment is not supported for image now"
        train_env = make_env_with_kwargs(
            config.env_id,
            max_steps=config.max_steps,
            kwargs=config.env_kwargs,
            normalize_obs=config.normalize_obs,
        )
        # always use rgb_array and default max_steps for evaluation
        eval_env = make_env_with_kwargs(
            config.env_id,
            max_steps=config.max_steps,
            kwargs={**config.env_kwargs, "render_mode": "rgb_array"},
            normalize_obs=config.normalize_obs,
        )
        return train_env, eval_env

    def get_env_fn(
        render_mode: str | None,
        max_steps: int | None = None,
    ) -> Callable[[], EnvType]:
        def env_fn() -> EnvType:
            if config.use_image:
                # use cast to make mypy happy
                # For observation, returning float32, but EnvType requiring unint8 or float32.
                assert render_mode is not None
                assert config.image_shape is not None
                return cast(
                    EnvType,
                    make_image_env_for_vectorized(
                        env_id=config.env_id,
                        render_mode=render_mode,
                        resize_shape=config.image_shape,
                        frame_stack_size=config.frame_stack,
                        frame_skip=config.frame_skip,
                        max_steps=max_steps,
                        normalize_obs=config.normalize_obs,
                    ),
                )

            # use cast to make mypy happy
            # For observation, returning float32, but EnvType requiring unint8 or float32.
            return cast(
                EnvType,
                make_1d_env_for_vectorized(
                    env_id=config.env_id,
                    render_mode=render_mode,
                    max_steps=max_steps,
                    normalize_obs=config.normalize_obs,
                ),
            )

        return env_fn

    train_envs = _make_vector_env(
        get_env_fn(config.training_render_mode, config.max_steps),
        config.vector_env_num,
        config.use_multi_processing,
    )

    # always use rgb_array and default max_steps for evaluation
    eval_env = get_env_fn(render_mode="rgb_array", max_steps=None)()

    return train_envs, eval_env


def verify_vector_env_with_continuous_action(envs: EnvsTypeC) -> None:
    if not hasattr(envs, "num_envs"):
        raise TypeError("train_env is a single environment, use env property instead")
    assert isinstance(envs.single_action_space, Box), "Env must be continuous action space"
    assert len(envs.single_action_space.shape) == 1, "Env must be continuous action space"


def verify_env_with_continuous_action(env: EnvTypeC) -> None:
    if not hasattr(env, "action_space"):
        raise TypeError("train_env is a single environment, use env property instead")
    assert isinstance(env.action_space, Box), "Env must be continuous action space"
    assert env.action_space.shape is not None
    assert len(env.action_space.shape) == 1, "Env must be continuous action space"


def dump_env_wrappers(env: gym.Env[Any, Any]) -> dict[str, Any]:
    """Traverse the env wrapper stack and dump each wrapper's class and init-args to JSON."""
    wrappers = []
    curr = env
    while isinstance(curr, gym.Wrapper):
        cls = curr.__class__
        # try to extract the named parameters from the __init__ signature
        sig = inspect.signature(cls.__init__)
        params = {}
        for name, _ in sig.parameters.items():
            if name == "self" or name == "env":
                continue
            # if the wrapper object has the same name attribute, record it
            if hasattr(curr, name):
                val = getattr(curr, name)
                try:
                    json.dumps(val)  # only record JSON serializable
                    params[name] = val
                except TypeError:
                    params[name] = str(val)
        wrappers.append(
            {
                "class": f"{cls.__module__}.{cls.__name__}",
                "params": params,
            }
        )
        curr = curr.env

    # the bottom most is the original environment
    base = curr
    spec = getattr(base, "spec", None)
    base_info = {
        "class": f"{base.__class__.__module__}.{base.__class__.__name__}",
        "id": spec.id if spec else None,
    }
    return {"wrappers": wrappers, "base_env": base_info}
