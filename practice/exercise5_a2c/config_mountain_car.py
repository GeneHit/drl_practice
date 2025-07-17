import gymnasium as gym
import torch
from gymnasium.spaces import Discrete
from torch.optim import Adam

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.base.env_typing import EnvType
from practice.exercise5_a2c.a2c_gae_exercise import A2CConfig, A2CTrainer, ActorCritic
from practice.utils.env_utils import get_device, get_env_from_config
from practice.utils_for_coding.agent_utils import A2CAgent


def get_app_config() -> A2CConfig:
    """Get the application config."""
    # get cuda or mps if available
    device = get_device()
    return A2CConfig(
        device=device,
        episode=100,
        rollout_len=16,
        learning_rate=5e-4,
        gamma=0.99,
        gae_lambda_or_n_step=0.97,
        entropy_coef=0.01,
        grad_acc=1,
        eval_episodes=50,
        eval_random_seed=42,
        eval_video_num=10,
        env_config=EnvConfig(
            env_id="MountainCar-v0",
            vector_env_num=8,
            use_multi_processing=True,
            record_eval_video=True,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=A2CTrainer,
            agent_type=A2CAgent,
            output_dir="results/exercise5_a2c/mountain_car/",
            save_result=True,
            model_filename="a2c_gae.pth",
            repo_id="A2C-GAE-MountainCarV0",
            algorithm_name="A2C-GAE",
            extra_tags=("A2C", "GAE"),
        ),
    )


def get_env_for_play_and_hub(config: A2CConfig) -> EnvType:
    """Get the environment for play and hub."""
    _, eval_env = get_env_from_config(config.env_config)
    return eval_env


def generate_context(config: A2CConfig) -> ContextBase:
    """Generate the context for the training."""
    env, eval_env = get_env_from_config(config.env_config)
    # always use vectorized environment
    assert isinstance(env, gym.vector.VectorEnv)

    obs_shape = eval_env.observation_space.shape
    assert obs_shape is not None
    assert isinstance(eval_env.action_space, Discrete)
    action_n = int(eval_env.action_space.n)
    actor_critic = ActorCritic(obs_dim=obs_shape[0], n_actions=action_n, hidden_size=128)
    # Load checkpoint if exists
    if config.checkpoint_pathname:
        checkpoint = torch.load(config.checkpoint_pathname, weights_only=False)
        if isinstance(checkpoint, dict):
            # It's a state_dict
            actor_critic.load_state_dict(checkpoint)
        else:
            # It's a full model, extract state_dict
            actor_critic.load_state_dict(checkpoint.state_dict())
    actor_critic.to(config.device)

    return ContextBase(
        train_env=env,
        eval_env=eval_env,
        trained_target=actor_critic,
        optimizer=Adam(actor_critic.parameters(), lr=config.learning_rate),
    )
