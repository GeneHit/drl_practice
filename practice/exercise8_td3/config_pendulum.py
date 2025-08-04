from typing import cast

import torch.optim as optim

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.env_typing import EnvsTypeC, EnvTypeC
from practice.exercise8_td3.td3_exercise import (
    TD3Actor,
    TD3Config,
    TD3Trainer,
)
from practice.utils.env_utils import (
    get_device,
    get_env_from_config,
    verify_env_with_continuous_action,
    verify_vector_env_with_continuous_action,
)
from practice.utils_for_coding.context_utils import ACContext
from practice.utils_for_coding.network_utils import DoubleQCritic
from practice.utils_for_coding.scheduler_utils import LinearSchedule


def get_app_config() -> TD3Config:
    """Get the application config."""
    # timestep = total_steps // vector_env_num = 240000 // 6 = 40000
    total_steps = 240000
    return TD3Config(
        device=get_device("cpu"),
        total_steps=total_steps,
        hidden_sizes=(256, 256),
        learning_rate=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        replay_buffer_capacity=int(total_steps * 1.0),
        batch_size=128,
        update_start_step=20000,
        policy_delay=2,
        policy_noise=0.2,
        noise_clip=0.5,
        exploration_noise=LinearSchedule(0.3, 0.001, 10000),
        max_action=2.0,
        tau=0.005,
        max_grad_norm=0.5,
        eval_episodes=50,
        eval_random_seed=42,
        eval_video_num=10,
        env_config=EnvConfig(
            env_id="Pendulum-v1",
            vector_env_num=6,
            use_multi_processing=True,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=TD3Trainer,
            output_dir="results/exercise8_td3/pendulum/",
            save_result=True,
            model_filename="td3_pendulum.pth",
            repo_id="TD3-PendulumV1",
            algorithm_name="TD3",
            extra_tags=("policy-gradient", "pytorch", "ddpg"),
        ),
    )


def generate_context(config: TD3Config) -> ACContext:
    """Generate the context for the TD3 algorithm."""
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

    actor = TD3Actor(
        state_dim=obs_shape[0],
        action_dim=act_shape[0],
        max_action=config.max_action,
        hidden_sizes=config.hidden_sizes,
    )
    critic = DoubleQCritic(
        state_dim=obs_shape[0],
        action_dim=act_shape[0],
        hidden_sizes=config.hidden_sizes,
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
