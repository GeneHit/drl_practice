import time
from typing import Any, Callable, cast

import torch
from gymnasium.spaces import Discrete

from hands_on.exercise2_dqn.dqn_exercise import EnvsType, EnvType
from hands_on.exercise3_reinforce.config import ReinforceConfig
from hands_on.exercise3_reinforce.reinforce_exercise import (
    Reinforce1DNet,
    reinforce_train_loop,
)
from hands_on.utils.agent_utils import NNAgent
from hands_on.utils.env_utils import get_device
from hands_on.utils.evaluation_utils import evaluate_and_save_results


def reinforce_train_with_envs(
    envs: EnvsType,
    env_fn: Callable[[], EnvType],
    cfg_data: dict[str, Any],
) -> None:
    """Main training function for multi-environment REINFORCE."""
    # Get environment info from a single environment
    obs_shape = envs.single_observation_space.shape
    act_space = envs.single_action_space
    assert obs_shape is not None  # make mypy happy
    assert isinstance(act_space, Discrete)  # make mypy happy
    action_n: int = int(act_space.n)

    # Create policy network
    if len(obs_shape) != 1:
        raise NotImplementedError("2D observation space not implemented for REINFORCE")
    net = Reinforce1DNet(state_dim=obs_shape[0], action_dim=action_n, hidden_dim=128, layer_num=2)

    # Load checkpoint if exists
    checkpoint_pathname = cfg_data.get("checkpoint_pathname", None)
    if checkpoint_pathname:
        net.load_state_dict(torch.load(checkpoint_pathname))

    # Set device
    device = get_device()
    net = net.to(device)

    # Start training
    start_time = time.time()

    # Create config from hyper_params
    train_result = reinforce_train_loop(
        envs=envs,
        net=net,
        device=device,
        config=ReinforceConfig.from_dict(cfg_data["hyper_params"]),
    )
    envs.close()

    assert "episode_rewards" in train_result, "episode_rewards must be in train_result"
    train_result["duration_min"] = (time.time() - start_time) / 60

    # Create agent and evaluate/save results on a single environment
    agent = NNAgent(net=net)
    eval_env = env_fn()

    try:
        evaluate_and_save_results(
            env=eval_env, agent=agent, cfg_data=cfg_data, train_result=train_result
        )
    finally:
        eval_env.close()


def reinforce_main(cfg_data: dict[str, Any]) -> None:
    """Main function to setup environments and start training."""
    # Import here to avoid circular imports
    import gymnasium as gym

    from hands_on.utils.env_utils import make_1d_env

    def _make_vector_env(
        env_fn: Callable[[], EnvType],
        num_envs: int,
        use_multi_processing: bool = False,
    ) -> EnvsType:
        """Create a vector environment."""
        if use_multi_processing and num_envs > 1:
            return gym.vector.AsyncVectorEnv([env_fn for _ in range(num_envs)])
        return gym.vector.SyncVectorEnv([env_fn for _ in range(num_envs)])

    # Create environment factory function
    env_params = cfg_data["env_params"]

    def env_fn() -> EnvType:
        env, _ = make_1d_env(env_id=env_params["env_id"], max_steps=env_params.get("max_steps"))
        return cast(EnvType, env)

    # Create vector environment
    num_envs: int = cfg_data["hyper_params"]["num_envs"]
    use_multi_processing: bool = cfg_data["hyper_params"].get("use_multi_processing", False)
    envs = _make_vector_env(env_fn, num_envs, use_multi_processing)

    # Get environment info and update config
    temp_env, more_env_info = make_1d_env(
        env_id=env_params["env_id"], max_steps=env_params.get("max_steps")
    )
    temp_env.close()
    cfg_data["env_params"].update(more_env_info)

    # Start training
    try:
        reinforce_train_with_envs(envs=envs, env_fn=env_fn, cfg_data=cfg_data)
    finally:
        if not envs.closed:
            envs.close()
