import os
from typing import Any, Callable, cast

import gymnasium as gym
import torch
from gymnasium.spaces import Discrete

from hands_on.exercise2_dqn.dqn_exercise import EnvsType, EnvType
from hands_on.exercise3_reinforce.config import ReinforceConfig
from hands_on.exercise3_reinforce.reinforce_exercise import (
    Reinforce1DNet,
    reinforce_train_loop,
)
from hands_on.exercise4_curiosity.curiosity_exercise import RNDNetwork1D, RNDReward
from hands_on.utils.agent_utils import NNAgent
from hands_on.utils.env_utils import get_device, make_1d_env
from hands_on.utils.evaluation_utils import evaluate_and_save_results
from hands_on.utils_for_coding.scheduler_utils import ExponentialSchedule


def reinforce_train_with_envs(
    envs: EnvsType,
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
    device = get_device()

    intrinsic_rewarders = []
    if "model_params" in cfg_data and "intrinsic_rewarders" in cfg_data["model_params"]:
        for intrinsic_rewarder in cfg_data["model_params"]["intrinsic_rewarders"]:
            if intrinsic_rewarder["type"] == "RNDReward":
                init_predictor = RNDNetwork1D(
                    obs_dim=obs_shape[0],
                    output_dim=intrinsic_rewarder["params"]["output_dim"],
                ).to(device)
                target_network = RNDNetwork1D(
                    obs_dim=obs_shape[0],
                    output_dim=intrinsic_rewarder["params"]["output_dim"],
                ).to(device)
                intrinsic_rewarders.append(
                    RNDReward(
                        init_predictor=init_predictor,
                        target_network=target_network,
                        optimizer=torch.optim.Adam(
                            init_predictor.parameters(), lr=intrinsic_rewarder["params"]["lr"]
                        ),
                        device=device,
                        beta=ExponentialSchedule(
                            start_e=5.0,
                            end_e=intrinsic_rewarder["params"]["beta"],
                            decay_rate=0.0005,
                        ),
                    )
                )

    # Load checkpoint if exists
    checkpoint_pathname = cfg_data.get("checkpoint_pathname", None)
    if checkpoint_pathname:
        checkpoint = torch.load(checkpoint_pathname, weights_only=False)
        if isinstance(checkpoint, dict):
            # It's a state_dict
            net.load_state_dict(checkpoint)
        else:
            # It's a full model, extract state_dict
            net.load_state_dict(checkpoint.state_dict())

    # Set device
    net = net.to(device)

    # Create config from hyper_params
    reinforce_train_loop(
        envs=envs,
        net=net,
        device=device,
        config=ReinforceConfig.from_dict(cfg_data["hyper_params"]),
        log_dir=os.path.join(cfg_data["output_params"]["output_dir"], "runs"),
        rewarders=intrinsic_rewarders,
    )
    envs.close()

    # Create agent and evaluate/save results on a single environment
    agent = NNAgent(net=net)
    eval_env, _ = make_1d_env(
        env_id=cfg_data["env_params"]["env_id"],
        max_steps=cfg_data["env_params"].get("max_steps"),
        render_mode="rgb_array",
    )

    try:
        evaluate_and_save_results(env=eval_env, agent=agent, cfg_data=cfg_data)
    finally:
        eval_env.close()


def reinforce_main(cfg_data: dict[str, Any]) -> None:
    """Main function to setup environments and start training."""

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
    envs = _make_vector_env(
        env_fn=env_fn,
        num_envs=int(cfg_data["hyper_params"]["num_envs"]),
        use_multi_processing=cfg_data["hyper_params"].get("use_multi_processing", False),
    )

    # Get environment info and update config
    temp_env, more_env_info = make_1d_env(
        env_id=env_params["env_id"], max_steps=env_params.get("max_steps")
    )
    temp_env.close()
    cfg_data["env_params"].update(more_env_info)

    # Start training
    try:
        reinforce_train_with_envs(envs=envs, cfg_data=cfg_data)
    finally:
        if not envs.closed:
            envs.close()
