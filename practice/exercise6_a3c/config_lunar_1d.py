from gymnasium.spaces import Discrete
from torch.optim import Adam

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.exercise5_a2c.a2c_gae_exercise import A2CTrainer, ActorCritic
from practice.exercise6_a3c.a3c_exercise import A3CConfig
from practice.utils.env_utils import get_device, get_env_from_config
from practice.utils_for_coding.scheduler_utils import LinearSchedule


def get_app_config() -> A3CConfig:
    """Get the application config."""
    # rollout_num = 900000 / 128 / 3 / 2 = 1171
    global_step = 900000
    num_workers = 3
    step_per_worker = global_step // num_workers
    return A3CConfig(
        num_workers=num_workers,
        # have to use cpu for multiprocessing
        device=get_device("cpu"),
        total_steps=step_per_worker,
        rollout_len=128,
        learning_rate=1e-4,
        critic_lr=5e-5,
        critic_lr_gamma=0.995,
        gamma=0.99,
        gae_lambda_or_n_step=0.97,
        entropy_coef=LinearSchedule(start_e=0.2, end_e=0.1, duration=300),
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
            vector_env_num=2,
            use_multi_processing=False,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=A2CTrainer,  # unused
            output_dir="results/exercise6_a3c/lunar/",
            save_result=True,
            fps=30,
            fps_skip=2,
            repo_id="A3C-LunarLanderV3",
            algorithm_name="A3C",
            extra_tags=("policy-gradient", "pytorch", "a3c", "gae"),
        ),
    )


def generate_context(config: A3CConfig) -> ContextBase:
    """Generate the context for the A3C algorithm.

    Only for gameplay, check the `a3c_exercise.py` for the training process.
    """
    train_env, eval_env = get_env_from_config(config.env_config)

    # Initialize the global shared actor-critic network.
    obs_shape = eval_env.observation_space.shape
    assert obs_shape is not None
    assert isinstance(eval_env.action_space, Discrete)
    action_n = int(eval_env.action_space.n)
    actor_critic = ActorCritic(
        obs_dim=obs_shape[0],
        n_actions=action_n,
        hidden_sizes=config.hidden_sizes,
    )
    actor_critic.to(config.device)

    ctx = ContextBase(
        train_env=train_env,
        eval_env=eval_env,
        trained_target=actor_critic,
        optimizer=Adam(actor_critic.parameters(), lr=config.learning_rate),
    )
    return ctx
