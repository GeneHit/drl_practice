"""Train the DQN agent with multiple environments.

Reference:
Algorithm: https://huggingface.co/learn/deep-rl-course/unit3/deep-q-algorithm
Code:https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn_atari.py
"""

import argparse
import time
from datetime import datetime
from typing import Any, Callable, cast

import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium.spaces import Discrete

# from torch.utils.tensorboard import SummaryWriter
from hands_on.exercise2_dqn.config import DQNTrainConfig
from hands_on.exercise2_dqn.dqn_exercise import (
    DQNAgent,
    EnvsType,
    EnvType,
    QNet1D,
    QNet2D,
    dqn_train_loop,
)
from hands_on.utils.env_utils import make_1d_env, make_image_env
from hands_on.utils.evaluation_utils import evaluate_agent
from hands_on.utils.file_utils import (
    load_config_from_json,
    save_model_and_result,
)


def _make_vector_env(
    env_fn: Callable[[], EnvType],
    num_envs: int,
    use_multi_processing: bool = False,
) -> EnvsType:
    """Create a vector environment."""
    if use_multi_processing and num_envs > 1:
        return gym.vector.AsyncVectorEnv([env_fn for _ in range(num_envs)])

    return gym.vector.SyncVectorEnv([env_fn for _ in range(num_envs)])


def dqn_train_with_multi_envs(
    envs: EnvsType,
    env_fn: Callable[[], EnvType],
    cfg_data: dict[str, Any],
) -> None:
    """Main training function for multi-environment DQN."""
    # Get environment info from a single environment
    obs_shape = envs.single_observation_space.shape
    act_space = envs.single_action_space
    assert obs_shape is not None  # make mypy happy
    assert isinstance(act_space, Discrete)  # make mypy happy
    action_n: int = int(act_space.n)

    # Create Q-network
    if len(obs_shape) == 1:
        q_network: nn.Module = QNet1D(state_n=obs_shape[0], action_n=action_n)
    else:
        assert len(obs_shape) == 3, "The observation space must be 3D"
        q_network = QNet2D(in_shape=obs_shape, action_n=action_n)

    # Load checkpoint if exists
    checkpoint_pathname = cfg_data.get("checkpoint_pathname", None)
    if checkpoint_pathname:
        q_network.load_state_dict(torch.load(checkpoint_pathname))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    q_network = q_network.to(device)

    # Start training
    start_time = time.time()
    train_result = dqn_train_loop(
        envs=envs,
        q_network=q_network,
        device=device,
        config=DQNTrainConfig.from_dict(cfg_data["hyper_params"]),
    )
    envs.close()
    duration_min = (time.time() - start_time) / 60
    train_result["duration_min"] = duration_min

    # Evaluate the agent on a single environment
    dqn_agent = DQNAgent(q_network=q_network)
    eval_env = env_fn()
    try:
        mean_reward, std_reward = evaluate_agent(
            env=eval_env,
            policy=dqn_agent,
            max_steps=cfg_data["eval_params"].get("max_steps", None),
            episodes=int(cfg_data["eval_params"]["eval_episodes"]),
            seed=tuple(cfg_data["eval_params"]["eval_seed"]),
        )
    finally:
        eval_env.close()

    # Save model and result if requested
    save_result: bool = cfg_data["output_params"].get("save_result", False)
    if save_result:
        eval_result = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        cfg_data["env_params"].update({"device": str(device)})
        save_model_and_result(
            cfg_data, train_result, eval_result, agent=dqn_agent
        )


def main(cfg_data: dict[str, Any]) -> None:
    """Main function to setup environments and start training."""
    # Create environment factory function
    env_params = cfg_data["env_params"]
    use_image: bool = env_params.get("use_image", False)

    def env_fn() -> EnvType:
        if use_image:
            env, _ = make_image_env(
                env_id=env_params["env_id"],
                render_mode=env_params["render_mode"],
                resize_shape=tuple(env_params["resize_shape"]),
                frame_stack_size=env_params["frame_stack_size"],
            )
            return cast(EnvType, env)
        else:
            env, _ = make_1d_env(env_id=env_params["env_id"])
            return cast(EnvType, env)

    # Create vector environment
    num_envs: int = cfg_data["hyper_params"]["num_envs"]
    use_multi_processing: bool = cfg_data["hyper_params"][
        "use_multi_processing"
    ]
    envs = _make_vector_env(env_fn, num_envs, use_multi_processing)

    # Get environment info and update config
    temp_env, more_env_info = (
        make_image_env(
            env_id=env_params["env_id"],
            render_mode=env_params["render_mode"],
            resize_shape=tuple(env_params["resize_shape"]),
            frame_stack_size=env_params["frame_stack_size"],
        )
        if use_image
        else make_1d_env(env_id=env_params["env_id"])
    )
    temp_env.close()
    cfg_data["env_params"].update(more_env_info)

    # Start training
    try:
        dqn_train_with_multi_envs(envs=envs, env_fn=env_fn, cfg_data=cfg_data)
    finally:
        if not envs.closed:
            envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_data = load_config_from_json(args.config)
    main(cfg_data=cfg_data)
