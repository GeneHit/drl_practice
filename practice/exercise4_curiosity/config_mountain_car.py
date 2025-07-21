import gymnasium as gym
from gymnasium.spaces import Discrete
from torch.optim import Adam

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.env_typing import EnvType
from practice.exercise3_reinforce.reinforce_exercise import Reinforce1DNet
from practice.exercise4_curiosity.curiosity_exercise import (
    RND1DNetworkConfig,
    RNDRewardConfig,
)
from practice.exercise4_curiosity.enhanced_reinforce import (
    EnhancedReinforceConfig,
    EnhancedReinforceTrainer,
    ReinforceContext,
)
from practice.utils.env_utils import get_device, get_env_from_config
from practice.utils_for_coding.agent_utils import NNAgent
from practice.utils_for_coding.baseline_utils import ConstantBaseline
from practice.utils_for_coding.network_utils import load_checkpoint_if_exists
from practice.utils_for_coding.reward_utils import XDirectionShapingRewardConfig
from practice.utils_for_coding.scheduler_utils import ExponentialSchedule


def get_app_config() -> EnhancedReinforceConfig:
    """Get the application config."""
    # get cuda or mps if available
    device = get_device()
    return EnhancedReinforceConfig(
        device=device,
        episode=1000,
        learning_rate=1e-3,
        gamma=0.999,
        grad_acc=1,
        baseline=ConstantBaseline(decay=0.99),
        entropy_coef=0.01,
        eval_episodes=20,
        eval_random_seed=42,
        eval_video_num=10,
        reward_configs=(
            RND1DNetworkConfig(
                rnd_config=RNDRewardConfig(
                    beta=ExponentialSchedule(start_e=5.0, end_e=1.0, decay_rate=-0.005),
                    device=device,
                    normalize=True,
                ),
                obs_dim=2,  # MountainCar observation dimension
                output_dim=32,
                learning_rate=1e-3,
            ),
            XDirectionShapingRewardConfig(
                beta=ExponentialSchedule(start_e=5.0, end_e=1.0, decay_rate=-0.005),
                goal_position=None,
            ),
        ),
        env_config=EnvConfig(
            env_id="MountainCar-v0",
        ),
        artifact_config=ArtifactConfig(
            trainer_type=EnhancedReinforceTrainer,
            agent_type=NNAgent,
            output_dir="results/exercise4_curiosity/mountain_car/",
            save_result=True,
            model_filename="curiosity.pth",
            repo_id="Reinforce-MountainCarV0",
            algorithm_name="Reinforce_RND",
            extra_tags=("curiosity", "reinforce", "rnd"),
        ),
    )


def get_env_for_play_and_hub(config: EnhancedReinforceConfig) -> EnvType:
    """Get the environment for play and hub."""
    _, eval_env = get_env_from_config(config.env_config)
    return eval_env


def generate_context(config: EnhancedReinforceConfig) -> ReinforceContext:
    """Generate the context for the training."""
    env, eval_env = get_env_from_config(config.env_config)
    assert isinstance(env, gym.Env)  # Changed from EnvType to gym.Env

    obs_shape = eval_env.observation_space.shape
    assert obs_shape is not None
    assert isinstance(eval_env.action_space, Discrete)
    action_n = int(eval_env.action_space.n)
    policy = Reinforce1DNet(state_dim=obs_shape[0], action_dim=action_n)
    load_checkpoint_if_exists(policy, config.checkpoint_pathname)
    policy.to(config.device)

    # Generate rewarders from reward_configs
    rewarders = tuple(reward_config.get_rewarder() for reward_config in config.reward_configs)

    return ReinforceContext(
        train_env=env,
        eval_env=eval_env,
        trained_target=policy,
        optimizer=Adam(policy.parameters(), lr=config.learning_rate),
        rewarders=rewarders,
    )
