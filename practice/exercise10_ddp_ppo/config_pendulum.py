"""PPO on Pendulum-v1, for verification."""

from typing import cast

import torch.nn as nn
from torch.optim import Adam

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.base.env_typing import EnvsTypeC, EnvTypeC
from practice.exercise10_ddp_ppo.ppo_rnd_exercise import ContACNet, ContPPOConfig, ContPPOTrainer
from practice.utils.dist_utils import auto_init_distributed, get_device, is_main_process
from practice.utils.env_utils import (
    get_env_from_config,
    verify_env_with_continuous_action,
    verify_vector_env_with_continuous_action,
)
from practice.utils_for_coding.agent_utils import ContAgent
from practice.utils_for_coding.network_utils import load_checkpoint_if_exists
from practice.utils_for_coding.scheduler_utils import LinearSchedule


def get_app_config() -> ContPPOConfig:
    """Get the application config."""
    return ContPPOConfig(
        device=get_device("cpu"),
        timesteps=600,
        rollout_len=1024,
        learning_rate=1e-4,
        critic_lr=1e-4,
        gamma=0.99,
        gae_lambda=0.97,
        # pusher is simple, use a very small entropy coef
        entropy_coef=LinearSchedule(0.3, 0.005, 500),
        value_loss_coef=0.02,
        max_grad_norm=0.5,
        num_epochs=8,
        # better that rollout_len // minibatch_num >= 64
        minibatch_num=8,
        clip_coef=0.2,
        hidden_sizes=(128, 128),
        use_layer_norm=True,
        action_scale=2.0,
        action_bias=0,
        log_std_min=-20,
        log_std_max=2,
        reward_configs=(),
        eval_episodes=50,
        eval_random_seed=42,
        eval_video_num=10,
        # the rollout number for logging
        log_interval=1,
        env_config=EnvConfig(
            env_id="Pendulum-v1",
            vector_env_num=6,
            use_multi_processing=True,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=ContPPOTrainer,
            agent_type=ContAgent,
            output_dir="results/exercise10_ddp_ppo/pendulum/",
            save_result=True,
            repo_id="PPO-RND-PendulumV1",
            algorithm_name="PPO",
            extra_tags=("policy-gradient", "pytorch", "ddp", "rnd"),
        ),
    )


def get_env_for_play_and_hub(config: ContPPOConfig) -> EnvTypeC:
    """Get the environment for play and hub."""
    train_envs, eval_env = get_env_from_config(config.env_config)
    train_envs.close()
    # use cast for type checking
    verify_env_with_continuous_action(cast(EnvTypeC, eval_env))
    return cast(EnvTypeC, eval_env)


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
        ]
    )

    return ContextBase(
        train_env=t_envs,
        eval_env=e_env,
        trained_target=ac,
        optimizer=optimizer,
        lr_schedulers=(),
        track_and_evaluate=is_main_process(),
    )
