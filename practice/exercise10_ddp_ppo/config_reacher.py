"""PPO on Reacher-v5, for verifying the implementation with 1 process."""

from typing import cast

import torch.nn as nn
from torch.optim import Adam

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.base.env_typing import EnvsTypeC, EnvTypeC
from practice.exercise10_ddp_ppo.ppo_rnd_exercise import ContACNet, ContPPOConfig, ContPPOTrainer
from practice.utils.dist_utils import auto_init_distributed, get_device
from practice.utils.env_utils import (
    get_env_from_config,
    verify_env_with_continuous_action,
    verify_vector_env_with_continuous_action,
)
from practice.utils_for_coding.network_utils import load_checkpoint_if_exists
from practice.utils_for_coding.scheduler_utils import LinearSchedule


def get_app_config() -> ContPPOConfig:
    num_envs = 6
    rollout_len = 360
    # total steps = timesteps * num_envs * rollout_len
    timesteps = 800
    minibatch_size = 1024
    minibatch_num = (num_envs * rollout_len) // minibatch_size
    return ContPPOConfig(
        device=get_device("cpu"),
        timesteps=timesteps,
        rollout_len=rollout_len,
        learning_rate=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=LinearSchedule(start_e=0.02, end_e=0.005, duration=int(0.8 * timesteps)),
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        num_epochs=10,
        minibatch_num=minibatch_num,
        clip_coef=0.2,
        value_clip_range=1.0,
        hidden_sizes=(64, 64),
        use_layer_norm=True,
        action_scale=1.0,
        action_bias=0.0,
        log_std_min=-20,
        log_std_max=2,
        log_std_state_dependent=False,
        reward_configs=(),
        eval_episodes=20,
        eval_random_seed=42,
        eval_video_num=5,
        log_interval=1,
        env_config=EnvConfig(
            env_id="Reacher-v5",
            vector_env_num=num_envs,
            use_multi_processing=True,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=ContPPOTrainer,
            output_dir="results/exercise10_ddp_ppo/reacher/",
            save_result=True,
            repo_id="PPO-Reacher-v5",
            algorithm_name="PPO",
            extra_tags=("mujoco",),
        ),
    )


def generate_context(config: ContPPOConfig) -> ContextBase:
    """Generate the context for the training."""
    train_envs, eval_env = get_env_from_config(config.env_config)
    # use cast for type checking
    t_envs = cast(EnvsTypeC, train_envs)
    e_env = cast(EnvTypeC, eval_env)
    verify_vector_env_with_continuous_action(t_envs)
    verify_env_with_continuous_action(e_env)

    obs_shape = eval_env.observation_space.shape
    act_shape = eval_env.action_space.shape
    assert obs_shape is not None
    assert act_shape is not None

    actor_critic = ContACNet(
        obs_dim=obs_shape[0],
        act_dim=act_shape[0],
        hidden_sizes=config.hidden_sizes,
        action_scale=config.action_scale,
        action_bias=config.action_bias,
        use_layer_norm=config.use_layer_norm,
        log_std_min=config.log_std_min,
        log_std_max=config.log_std_max,
        log_std_state_dependent=config.log_std_state_dependent,
    )
    if config.checkpoint_pathname is not None:
        load_checkpoint_if_exists(actor_critic, config.checkpoint_pathname)
    ac = auto_init_distributed(config.device, actor_critic)

    def split_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        critic_params = []
        other_params = []
        for name, param in model.named_parameters():
            # value is critic head
            if "value" in name:
                critic_params.append(param)
            else:
                other_params.append(param)
        return other_params, critic_params

    other_params, critic_params = split_params(ac)
    assert len(other_params) > 0 and len(critic_params) > 0, "No parameters to optimize"
    optimizer = Adam(
        [
            {"params": other_params, "lr": config.learning_rate},
            {"params": critic_params, "lr": config.critic_lr},
        ]
    )

    return ContextBase(
        train_env=t_envs,
        eval_env=e_env,
        trained_target=ac,
        optimizer=optimizer,
        lr_schedulers=(),
        track_and_evaluate=True,
    )
