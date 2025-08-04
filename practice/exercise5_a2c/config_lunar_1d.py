import gymnasium as gym
from gymnasium.spaces import Discrete
from torch.optim import Adam

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.exercise5_a2c.a2c_gae_exercise import A2CConfig, A2CTrainer, ActorCritic
from practice.utils.env_utils import get_device, get_env_from_config
from practice.utils_for_coding.network_utils import load_checkpoint_if_exists
from practice.utils_for_coding.scheduler_utils import LinearSchedule


def get_app_config() -> A2CConfig:
    """Get the application config."""
    # rollout_num = 900000 / 128 / 6 = 1171
    return A2CConfig(
        # use CPU is faster since nn model is small
        device=get_device("cpu"),
        total_steps=900000,
        rollout_len=128,
        learning_rate=3e-4,
        critic_lr=3e-4,
        critic_lr_gamma=0.995,
        gamma=0.99,
        gae_lambda_or_n_step=0.97,
        entropy_coef=LinearSchedule(start_e=0.2, end_e=0.1, duration=600),
        value_loss_coef=0.5,
        value_clip_range=1.0,
        reward_clip=(-1, 1),
        max_grad_norm=0.5,
        hidden_sizes=(256, 256),
        log_interval=10,
        eval_episodes=50,
        eval_random_seed=42,
        eval_video_num=10,
        env_config=EnvConfig(
            env_id="LunarLander-v3",
            vector_env_num=6,
            use_multi_processing=True,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=A2CTrainer,
            output_dir="results/exercise5_a2c/lunar/",
            save_result=True,
            fps=30,
            fps_skip=2,
            repo_id="A2C-GAE-LunarLanderV3",
            algorithm_name="A2C-GAE",
            extra_tags=("policy-gradient", "pytorch", "a2c", "gae"),
        ),
    )


def generate_context(config: A2CConfig) -> ContextBase:
    """Generate the context for the training."""
    env, eval_env = get_env_from_config(config.env_config)
    # always use vectorized environment
    assert isinstance(env, gym.vector.VectorEnv)

    obs_shape = eval_env.observation_space.shape
    assert obs_shape is not None
    assert isinstance(eval_env.action_space, Discrete)
    action_n = int(eval_env.action_space.n)
    actor_critic = ActorCritic(
        obs_dim=obs_shape[0], n_actions=action_n, hidden_sizes=config.hidden_sizes
    )
    # Load checkpoint if exists
    load_checkpoint_if_exists(actor_critic, config.checkpoint_pathname)
    actor_critic.to(config.device)

    shared_and_policy_params = list(actor_critic.shared_layers.parameters()) + list(
        actor_critic.policy_logits.parameters()
    )
    optimizer = Adam(
        [
            {"params": shared_and_policy_params, "lr": config.learning_rate},
            {"params": actor_critic.value_head.parameters(), "lr": config.critic_lr},
        ]
    )

    return ContextBase(
        train_env=env,
        eval_env=eval_env,
        trained_target=actor_critic,
        optimizer=optimizer,
        lr_schedulers=(),
    )
