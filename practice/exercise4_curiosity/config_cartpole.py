import gymnasium as gym
import torch
from gymnasium.spaces import Discrete
from torch.optim import Adam

from hands_on.utils.agent_utils import NNAgent
from hands_on.utils.env_utils import get_device
from hands_on.utils_for_coding.scheduler_utils import ExponentialSchedule
from practice.base.config import ArtifactConfig, EnvConfig
from practice.exercise4_curiosity.curiosity_exercise import (
    RND1DNetworkConfig,
    RNDRewardConfig,
)
from practice.exercise4_curiosity.enhanced_reinforce import (
    EnhancedReinforceConfig,
    EnhancedReinforceTrainer,
    Reinforce1DNet,
    ReinforceContext,
)
from practice.utils.env_utils import get_env_from_config


def get_app_config() -> EnhancedReinforceConfig:
    """Get the application config."""
    # get cuda or mps if available
    device = get_device()
    return EnhancedReinforceConfig(
        device=device,
        episode=10,
        learning_rate=1e-3,
        gamma=0.999,
        grad_acc=1,
        use_baseline=True,
        baseline_decay=0.99,
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
                obs_dim=4,  # CartPole observation dimension
                output_dim=32,
                learning_rate=1e-3,
            ),
        ),
        env_config=EnvConfig(
            env_id="CartPole-v1",
            record_eval_video=True,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=EnhancedReinforceTrainer,
            agent_type=NNAgent,
            output_dir="results/exercise4_curiosity/cartpole/",
            save_result=True,
            model_filename="curiosity.pth",
            repo_id="Reinforce-CartPole-v1",
        ),
    )


def generate_context(config: EnhancedReinforceConfig) -> ReinforceContext:
    """Generate the context for the application."""
    env, eval_env = get_env_from_config(config.env_config)
    assert isinstance(env, gym.Env)  # Changed from EnvType to gym.Env

    obs_shape = eval_env.observation_space.shape
    assert obs_shape is not None
    assert isinstance(eval_env.action_space, Discrete)
    action_n = int(eval_env.action_space.n)
    policy = Reinforce1DNet(state_dim=obs_shape[0], action_dim=action_n)
    # Load checkpoint if exists
    if config.checkpoint_pathname:
        checkpoint = torch.load(config.checkpoint_pathname, weights_only=False)
        if isinstance(checkpoint, dict):
            # It's a state_dict
            policy.load_state_dict(checkpoint)
        else:
            # It's a full model, extract state_dict
            policy.load_state_dict(checkpoint.state_dict())
    policy.to(config.device)

    # Generate rewarders from reward_configs
    rewarders = tuple(reward_config.get_rewarder() for reward_config in config.reward_configs)

    return ReinforceContext(
        env=env,
        eval_env=eval_env,
        trained_target=policy,
        optimizer=Adam(policy.parameters(), lr=config.learning_rate),
        rewarders=rewarders,
    )
