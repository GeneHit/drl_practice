import gymnasium as gym
from gymnasium.spaces import Discrete
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.base.env_typing import EnvType
from practice.exercise7_ppo.ppo_exercise import ActorCritic, PPOConfig, PPOTrainer
from practice.utils.env_utils import get_device, get_env_from_config
from practice.utils_for_coding.agent_utils import ACAgent
from practice.utils_for_coding.network_utils import load_checkpoint_if_exists
from practice.utils_for_coding.scheduler_utils import LinearSchedule


def get_app_config() -> PPOConfig:
    """Get the application config."""
    return PPOConfig(
        # use CPU is faster since nn model is small
        device=get_device("cpu"),  # "cuda" or "mps" or "cpu"
        total_steps=300000,
        rollout_len=128,
        learning_rate=3e-4,
        critic_lr=1e-4,
        critic_lr_gamma=0.95,
        gamma=0.99,
        gae_lambda_or_n_step=0.97,
        entropy_coef=LinearSchedule(start_e=0.15, end_e=0.05, duration=100),
        # entropy_coef=ConstantSchedule(0.1),
        value_loss_coef=0.1,
        max_grad_norm=0.5,
        hidden_size=64,
        eval_episodes=50,
        eval_random_seed=42,
        eval_video_num=10,
        env_config=EnvConfig(
            env_id="CartPole-v1",
            vector_env_num=6,
            use_multi_processing=True,
            # max_steps=1000,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=PPOTrainer,
            agent_type=ACAgent,
            output_dir="results/exercise7_ppo/cartpole/",
            save_result=True,
            model_filename="ppo.pth",
            repo_id="PPO-CartPoleV1",
            algorithm_name="PPO",
            extra_tags=("PPO", "GAE"),
        ),
        num_epochs=8,
        minibatch_num=4,
        clip_coef=0.2,
    )


def get_env_for_play_and_hub(config: PPOConfig) -> EnvType:
    """Get the environment for play and hub."""
    _, eval_env = get_env_from_config(config.env_config)
    return eval_env


def generate_context(config: PPOConfig) -> ContextBase:
    """Generate the context for the training."""
    env, eval_env = get_env_from_config(config.env_config)
    # always use vectorized environment
    assert isinstance(env, gym.vector.VectorEnv)

    obs_shape = eval_env.observation_space.shape
    assert obs_shape is not None
    assert isinstance(eval_env.action_space, Discrete)
    action_n = int(eval_env.action_space.n)
    actor_critic = ActorCritic(
        obs_dim=obs_shape[0], n_actions=action_n, hidden_size=config.hidden_size
    )
    # Load checkpoint if exists
    load_checkpoint_if_exists(actor_critic, config.checkpoint_pathname)
    actor_critic.to(config.device)

    optimizer = Adam(
        [
            {"params": actor_critic.shared_layers.parameters(), "lr": config.learning_rate},
            {"params": actor_critic.policy_logits.parameters(), "lr": config.learning_rate},
            {"params": actor_critic.value_head.parameters(), "lr": config.critic_lr},
        ]
    )
    lr_schedulers: tuple[LRScheduler, ...] = ()
    critic_lr_gamma = config.critic_lr_gamma
    if critic_lr_gamma is not None:
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=[
                lambda epoch: 1.0,  # group 0: shared lr
                lambda epoch: 1.0,  # group 1: policy lr
                lambda epoch: critic_lr_gamma**epoch,  # group 2: critic lr
            ],
        )
        lr_schedulers = (scheduler,)

    return ContextBase(
        train_env=env,
        eval_env=eval_env,
        trained_target=actor_critic,
        optimizer=optimizer,
        lr_schedulers=lr_schedulers,
    )
