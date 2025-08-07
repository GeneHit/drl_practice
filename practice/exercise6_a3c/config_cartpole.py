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
    # rollout_num = 250000 / 32 / 6 / 2 = 651
    global_step = 250000
    num_workers = 6
    step_per_worker = global_step // num_workers
    return A3CConfig(
        num_workers=num_workers,
        # have to use cpu for multiprocessing
        device=get_device("cpu"),
        total_steps=step_per_worker,
        rollout_len=32,
        learning_rate=1e-4,
        critic_lr=5e-5,
        critic_lr_gamma=0.95,
        gamma=0.99,
        gae_lambda_or_n_step=0.97,
        entropy_coef=LinearSchedule(v0=0.2, v1=0.1, t1=1000),
        # entropy_coef=ConstantSchedule(0.1),
        value_loss_coef=0.02,
        max_grad_norm=0.5,
        normalize_returns=True,
        hidden_sizes=(64, 64),
        log_interval=10,
        eval_episodes=50,
        eval_random_seed=42,
        eval_video_num=10,
        env_config=EnvConfig(
            env_id="CartPole-v1",
            vector_env_num=2,
            use_multi_processing=False,
            # max_steps=1000,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=A2CTrainer,
            output_dir="results/exercise6_a3c/cartpole/",
            save_result=True,
            fps_skip=2,
            repo_id="A3C-CartPoleV1",
            algorithm_name="A3C with GAE",
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
