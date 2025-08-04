from typing import cast

import torch.optim as optim

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.env_typing import EnvsTypeC, EnvTypeC
from practice.exercise9_sac.sac_exercise import (
    SACActor,
    SACConfig,
    SACTrainer,
)
from practice.utils.env_utils import (
    get_device,
    get_env_from_config,
    verify_env_with_continuous_action,
    verify_vector_env_with_continuous_action,
)
from practice.utils_for_coding.context_utils import ACContext
from practice.utils_for_coding.network_utils import DoubleQCritic


def get_app_config() -> SACConfig:
    """Get the application config."""
    # timestep = total_steps // vector_env_num = 120000 // 6 = 20000
    total_steps = 120000
    return SACConfig(
        device=get_device("cpu"),
        total_steps=total_steps,
        hidden_sizes=(128, 128),
        learning_rate=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        replay_buffer_capacity=int(total_steps * 0.8),
        batch_size=128,
        update_start_step=10000,
        max_action=2.0,
        tau=0.005,
        max_grad_norm=0.5,
        alpha=0.2,
        auto_tune_alpha=True,
        alpha_lr=3e-4,
        target_entropy=-1.0,  # -1 = - action_dimension
        log_std_min=-20,
        log_std_max=2,
        use_layer_norm=True,
        eval_episodes=50,
        eval_random_seed=42,
        eval_video_num=10,
        env_config=EnvConfig(
            env_id="Pendulum-v1",
            vector_env_num=6,
            use_multi_processing=True,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=SACTrainer,
            output_dir="results/exercise9_sac/pendulum/",
            save_result=True,
            repo_id="SAC-PendulumV1",
            algorithm_name="SAC",
            extra_tags=("policy-gradient", "pytorch", "ddpg"),
        ),
    )


def generate_context(config: SACConfig) -> ACContext:
    """Generate the context for the SAC algorithm."""
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

    actor_optimizer = optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=config.critic_lr)

    return ACContext(
        train_env=t_envs,
        eval_env=e_env,
        trained_target=actor,
        optimizer=actor_optimizer,
        critic=critic,
        critic_optimizer=critic_optimizer,
        lr_schedulers=(),
    )
