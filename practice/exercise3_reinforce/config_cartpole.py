import gymnasium as gym
from gymnasium.spaces import Discrete
from torch.optim import Adam

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.exercise3_reinforce.reinforce_exercise import (
    Reinforce1DNet,
    ReinforceConfig,
    ReinforceTrainer,
)
from practice.utils.env_utils import get_device, get_env_from_config
from practice.utils_for_coding.network_utils import load_checkpoint_if_exists
from practice.utils_for_coding.scheduler_utils import LinearSchedule


def get_app_config() -> ReinforceConfig:
    """Get the application config."""
    # cpu is faster since the training is not heavy.
    device = get_device("cpu")
    return ReinforceConfig(
        device=device,
        episode=2000,
        learning_rate=1e-4,
        gamma=0.99,
        entropy_coef=LinearSchedule(start_e=0.1, end_e=0.01, duration=2000),
        hidden_sizes=(128, 128),
        eval_episodes=20,
        eval_random_seed=42,
        eval_video_num=10,
        env_config=EnvConfig(
            env_id="CartPole-v1",
            # CartPole-v1 default 500, here set it to 1000 to see the performance of the agent.
            # max_steps=1000,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=ReinforceTrainer,
            output_dir="results/exercise3_reinforce/cartpole/",
            save_result=True,
            repo_id="Reinforce-CartPole",
            algorithm_name="Vanilla-Reinforce",
            extra_tags=("policy-gradient", "pytorch", "vanilla", "monte-carlo"),
        ),
    )


def generate_context(config: ReinforceConfig) -> ContextBase:
    """Generate the context for the training."""
    env, eval_env = get_env_from_config(config.env_config)
    assert isinstance(env, gym.Env)

    obs_shape = eval_env.observation_space.shape
    assert obs_shape is not None
    assert isinstance(eval_env.action_space, Discrete)
    action_n = int(eval_env.action_space.n)
    policy = Reinforce1DNet(
        state_dim=obs_shape[0], action_dim=action_n, hidden_sizes=config.hidden_sizes
    )
    load_checkpoint_if_exists(policy, config.checkpoint_pathname)
    policy.to(config.device)

    return ContextBase(
        train_env=env,
        eval_env=eval_env,
        trained_target=policy,
        optimizer=Adam(policy.parameters(), lr=config.learning_rate),
    )
