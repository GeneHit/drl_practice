import gymnasium as gym
import torch
from gymnasium.spaces import Discrete
from torch.optim import Adam

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.base.env_typing import EnvType
from practice.exercise3_reinforce.reinforce_exercise import (
    Reinforce1DNet,
    ReinforceConfig,
    ReinforceTrainer,
)
from practice.utils.env_utils import get_device, get_env_from_config
from practice.utils_for_coding.agent_utils import NNAgent


def get_app_config() -> ReinforceConfig:
    """Get the application config."""
    # get cuda or mps if available
    device = get_device()
    return ReinforceConfig(
        device=device,
        episode=1000,
        learning_rate=1e-3,
        gamma=0.999,
        entropy_coef=0.01,
        eval_episodes=20,
        eval_random_seed=42,
        eval_video_num=10,
        env_config=EnvConfig(
            env_id="MountainCar-v0",
        ),
        artifact_config=ArtifactConfig(
            trainer_type=ReinforceTrainer,
            agent_type=NNAgent,
            output_dir="results/exercise3_reinforce/mountain_car/",
            save_result=True,
            model_filename="reinforce.pth",
            repo_id="Reinforce-MountainCarV0",
            algorithm_name="Reinforce",
            extra_tags=("reinforce", "policy-gradient"),
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

    return ContextBase(
        train_env=env,
        eval_env=eval_env,
        trained_target=policy,
        optimizer=Adam(policy.parameters(), lr=config.learning_rate),
    )


def get_env_for_play_and_hub(config: ReinforceConfig) -> EnvType:
    """Get the environment for play and hub."""
    _, eval_env = get_env_from_config(config.env_config)
    return eval_env
