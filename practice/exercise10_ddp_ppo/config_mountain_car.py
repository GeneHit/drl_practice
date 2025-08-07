from typing import cast

import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.base.env_typing import EnvsTypeC, EnvTypeC
from practice.exercise4_curiosity.curiosity_exercise import RND1DNetworkConfig, RNDRewardConfig
from practice.exercise10_ddp_ppo.ppo_rnd_exercise import ContACNet, ContPPOConfig, ContPPOTrainer
from practice.utils.dist_utils import (
    auto_init_distributed,
    get_device,
    get_world_size,
    is_main_process,
)
from practice.utils.env_utils import (
    get_env_from_config,
    verify_env_with_continuous_action,
    verify_vector_env_with_continuous_action,
)
from practice.utils_for_coding.network_utils import load_checkpoint_if_exists
from practice.utils_for_coding.scheduler_utils import ConstantSchedule, LinearSchedule


def get_app_config() -> ContPPOConfig:
    """Get the application config."""
    device = get_device("cpu")
    total_steps = 300_000
    vector_env_num_per_process = 2
    rollout_len = 320
    minibatch_size = 64
    minibatch_num = rollout_len * vector_env_num_per_process // minibatch_size
    timesteps = total_steps // (get_world_size() * vector_env_num_per_process * rollout_len)
    return ContPPOConfig(
        device=device,
        timesteps=timesteps,
        rollout_len=rollout_len,
        learning_rate=1e-4,
        critic_lr=1e-4,
        gamma=0.995,
        gae_lambda=0.98,
        entropy_coef=ConstantSchedule(0.05),
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        num_epochs=4,
        minibatch_num=minibatch_num,
        clip_coef=0.2,
        value_clip_range=0.5,
        # reward_clip_range=ConstantSchedule(2.0),
        hidden_sizes=(32, 32),
        use_layer_norm=True,
        action_scale=1,
        action_bias=0,
        log_std_min=-10,
        log_std_max=2,
        log_std_state_dependent=False,
        reward_configs=(
            RND1DNetworkConfig(
                rnd_config=RNDRewardConfig(
                    beta=LinearSchedule(v0=5, v1=1, t1=int(timesteps * 0.8)),
                    device=device,
                    normalize=True,
                    max_reward=0.5,
                ),
                obs_dim=2,  # MountainCar observation dimension
                output_dim=32,
                hidden_sizes=(32, 32),
                learning_rate=5e-5,
            ),
        ),
        eval_episodes=100,
        eval_random_seed=42,
        eval_video_num=10,
        # the rollout number for logging
        log_interval=1,
        env_config=EnvConfig(
            env_id="MountainCarContinuous-v0",
            vector_env_num=vector_env_num_per_process,
            use_multi_processing=False,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=ContPPOTrainer,
            output_dir="results/exercise10_ddp_ppo/mountain_car/",
            save_result=True,
            repo_id="PPO-DDP-MountainCarContinuousV0",
            algorithm_name="PPO",
            extra_tags=("pytorch", "ddp", "rnd", "curiosity"),
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
        log_std_min=config.log_std_min,
        log_std_max=config.log_std_max,
        use_layer_norm=config.use_layer_norm,
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
        ],
        # weight_decay=1e-6,
    )
    # lr_schedulers = (
    #     # lr_scheduler.CosineAnnealingLR(optimizer, T_max=1400, eta_min=5e-5),
    #     lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.995),
    # )
    total_steps = config.timesteps * config.num_epochs * config.minibatch_num
    lr_schedulers = (
        lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[config.learning_rate * 10.0, config.critic_lr * 5.0],
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=50,
            anneal_strategy="cos",
        ),
        # lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=[
        #         config.learning_rate * 3,
        #         config.critic_lr * 3
        #     ],
        #     total_steps=total_steps,
        #     pct_start=0.2,
        #     div_factor=10.0,
        #     final_div_factor=10,
        #     anneal_strategy='cos'
        # ),
    )

    return ContextBase(
        train_env=t_envs,
        eval_env=e_env,
        trained_target=ac,
        optimizer=optimizer,
        lr_schedulers=lr_schedulers,
        track_and_evaluate=is_main_process(),
    )
