import os
from typing import Any

import torch
from gymnasium.spaces import Discrete

from hands_on.exercise2_dqn.dqn_exercise import EnvType
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


def reinforce_train(
    env: EnvType,
    cfg_data: dict[str, Any],
) -> None:
    """Main training function for single-environment REINFORCE."""
    # Get environment info
    obs_shape = env.observation_space.shape
    act_space = env.action_space
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
        env=env,
        net=net,
        device=device,
        config=ReinforceConfig.from_dict(cfg_data["hyper_params"]),
        log_dir=os.path.join(cfg_data["output_params"]["output_dir"], "runs"),
        rewarders=intrinsic_rewarders,
    )
    env.close()

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
    """Main function to setup environment and start training."""
    # Create environment
    env_params = cfg_data["env_params"]
    env, more_env_info = make_1d_env(
        env_id=env_params["env_id"], max_steps=env_params.get("max_steps")
    )
    cfg_data["env_params"].update(more_env_info)

    # Start training
    try:
        reinforce_train(env=env, cfg_data=cfg_data)
    finally:
        env.close()
