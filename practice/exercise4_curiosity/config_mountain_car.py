import gymnasium as gym
import torch.optim as optim
from gymnasium.spaces import Discrete

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.env_typing import EnvType
from practice.exercise3_reinforce.reinforce_exercise import Reinforce1DNet
from practice.exercise4_curiosity.curiosity_exercise import (
    RND1DNetworkConfig,
    RNDRewardConfig,
    XShapingRewardConfig,
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
from practice.utils_for_coding.scheduler_utils import LinearSchedule


def get_app_config() -> EnhancedReinforceConfig:
    """Get the application config."""
    device = get_device("cpu")
    total_steps = 200_000
    return EnhancedReinforceConfig(
        device=device,
        total_steps=total_steps,
        learning_rate=3e-3,
        lr_gamma=0.99,
        gamma=0.999,
        hidden_sizes=(32, 32),
        baseline=ConstantBaseline(decay=0.9),
        entropy_coef=LinearSchedule(start_e=0.01, end_e=0.001, duration=int(total_steps * 0.6)),
        max_grad_norm=0.5,
        log_interval=1,  # log every episode
        eval_episodes=100,
        eval_random_seed=42,
        eval_video_num=10,
        reward_configs=(
            # RND and shaping reward are both needed.
            RND1DNetworkConfig(
                rnd_config=RNDRewardConfig(
                    beta=LinearSchedule(
                        start_e=0.005, end_e=0.001, duration=int(total_steps * 0.8)
                    ),
                    device=device,
                    normalize=True,
                    max_reward=2,
                ),
                obs_dim=2,  # MountainCar observation dimension
                output_dim=32,
                hidden_sizes=(32, 32),
                learning_rate=5e-4,
            ),
            XShapingRewardConfig(
                beta=LinearSchedule(start_e=5.0, end_e=1.0, duration=int(total_steps * 0.8))
            ),
        ),
        env_config=EnvConfig(
            env_id="MountainCar-v0",
            normalize_obs=True,
            # max_steps=100,  # MountainCar-v0 default 200
        ),
        artifact_config=ArtifactConfig(
            trainer_type=EnhancedReinforceTrainer,
            agent_type=NNAgent,
            output_dir="results/exercise4_curiosity/mountain_car/",
            save_result=True,
            model_filename="model.pth",
            repo_id="Reinforce-MountainCarV0",
            algorithm_name="Reinforce_RND",
            extra_tags=("curiosity", "reinforce", "rnd", "policy-gradient", "pytorch"),
        ),
    )


def get_env_for_play_and_hub(config: EnhancedReinforceConfig) -> EnvType:
    """Get the environment for play and hub."""
    train_env, eval_env = get_env_from_config(config.env_config)
    train_env.close()
    return eval_env


def generate_context(config: EnhancedReinforceConfig) -> ReinforceContext:
    """Generate the context for the training."""
    env, eval_env = get_env_from_config(config.env_config)
    assert isinstance(env, gym.Env)  # Changed from EnvType to gym.Env

    obs_shape = eval_env.observation_space.shape
    assert obs_shape is not None
    assert isinstance(eval_env.action_space, Discrete)
    action_n = int(eval_env.action_space.n)
    policy = Reinforce1DNet(
        state_dim=obs_shape[0], action_dim=action_n, hidden_sizes=config.hidden_sizes
    )
    load_checkpoint_if_exists(policy, config.checkpoint_pathname)
    policy.to(config.device)

    # Generate rewarders from reward_configs
    rewarders = tuple(reward_config.get_rewarder() for reward_config in config.reward_configs)

    optimizer = optim.Adam(policy.parameters(), lr=config.learning_rate)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)

    return ReinforceContext(
        train_env=env,
        eval_env=eval_env,
        trained_target=policy,
        optimizer=optimizer,
        lr_schedulers=(lr_scheduler,),
        rewarders=rewarders,
    )
