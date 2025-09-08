"""PPO on Reacher-v5, for verifying the implementation with 1 process."""

from typing import cast

from torch.optim import Adam

from practice.core.base.config import ArtifactConfig, EnvConfig
from practice.core.base.env_typing import EnvsTypeC, EnvTypeC
from practice.core.utils.dist_utils import get_device
from practice.core.utils.env_utils import (
    get_env_from_config,
    verify_env_with_continuous_action,
    verify_vector_env_with_continuous_action,
)
from practice.exercise9_sac.sac_exercise import SACActor
from practice.exercise12_mbpo.mbpo_exercise import (
    MBPOConfig,
    MBPOContext,
    MBPOTrainer,
    ModelRolloutConfig,
)
from practice.exercise12_mbpo.model_based_env import EnvModel, ModelBasedConfig, TrainConfig
from practice.utils.network_utils import DoubleQCritic
from practice.utils.scheduler_utils import LinearSchedule


def get_app_config() -> MBPOConfig:
    # timestep = total_steps // vector_env_num = 60000 // 6 = 10000
    total_steps = 60_000
    return MBPOConfig(
        device=get_device("cpu"),
        total_steps=total_steps,
        hidden_sizes=(64, 64),
        learning_rate=3e-4,
        critic_lr=3e-4,
        gamma=0.995,
        replay_buffer_capacity=int(total_steps * 0.2),
        batch_size=256,
        update_start_step=5000,
        max_action=1.0,
        tau=0.005,
        max_grad_norm=0.5,
        alpha=0.2,
        auto_tune_alpha=True,
        alpha_lr=3e-4,
        target_entropy=-2.0,  # = - action_dimension
        log_std_min=-7.0,
        log_std_max=2.0,
        use_layer_norm=False,
        sac_update_interval=1,
        update_num_per_epoch=1,
        use_model_based_env=True,
        model_update_interval=250,
        model_rollout_config=ModelRolloutConfig(
            rollout_num=10,
            rollout_len=LinearSchedule(v0=1, v1=4, t1=int(0.8 * total_steps)),
            replay_buffer_capacity=int(total_steps * 0.4),
            batch_rate_of_sample=LinearSchedule(v0=0.15, v1=0.3, t1=int(0.8 * total_steps)),
        ),
        model_based_config=ModelBasedConfig(
            num_models=3,
            model_hidden_sizes=(256, 256),
            done_threshold=0.5,
            log_std_bounds=(-5.0, 2.0),
            eps=1e-6,
            train=TrainConfig(
                epochs=20,
                batch_size=256,
                lr=1e-3,
                weight_decay=1e-6,
                loss_weight_delta=1.0,
                loss_weight_reward=1.0,
                loss_weight_done=1.0,
                buffer_ratio_for_val=0.1,
                early_stop_patience=6,
                bootstrap=True,
                dataloader_num_workers=0,
                dataloader_pin_memory=False,
            ),
        ),
        eval_episodes=50,
        eval_random_seed=42,
        eval_video_num=10,
        env_config=EnvConfig(
            env_id="Reacher-v5",
            vector_env_num=6,
            use_multi_processing=True,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=MBPOTrainer,
            output_dir="results/exercise12_mbpo/reacher_used/",
            save_result=True,
            repo_id="MBPO-ReacherV5",
            algorithm_name="MBPO",
            extra_tags=("model-based", "pytorch", "sac"),
        ),
    )


def generate_context(config: MBPOConfig) -> MBPOContext:
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

    actor = SACActor(
        state_dim=obs_shape[0],
        action_dim=act_shape[0],
        action_scale=config.max_action,
        action_bias=0.0,
        hidden_sizes=config.hidden_sizes,
        log_std_min=config.log_std_min,
        log_std_max=config.log_std_max,
        use_layer_norm=config.use_layer_norm,
    )
    critic = DoubleQCritic(
        state_dim=obs_shape[0],
        action_dim=act_shape[0],
        hidden_sizes=config.hidden_sizes,
        use_layer_norm=config.use_layer_norm,
    )
    actor.to(config.device)
    critic.to(config.device)

    actor_optimizer = Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = Adam(critic.parameters(), lr=config.critic_lr)

    env_model = EnvModel(
        state_dim=obs_shape[0],
        action_dim=act_shape[0],
        hidden_sizes=config.model_based_config.model_hidden_sizes,
    )
    env_model.to(config.device)

    return MBPOContext(
        train_env=t_envs,
        eval_env=e_env,
        trained_target=actor,
        optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        lr_schedulers=(),
        env_model=env_model,
    )
