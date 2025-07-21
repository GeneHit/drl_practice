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
from practice.utils_for_coding.scheduler_utils import LinearSchedule


def get_app_config() -> A2CConfig:
    """Get the application config."""
    # get cuda or mps if available
    device = get_device()
    total_step = 600000
    return A2CConfig(
        device=device,
        total_steps=total_step,
        rollout_len=32,
        learning_rate=1e-4,
        gamma=0.995,
        gae_lambda_or_n_step=0.97,
        entropy_coef=LinearSchedule(start_e=0.1, end_e=0.02, duration=total_step),
        # entropy_coef=ConstantSchedule(0.1),
        value_loss_coef=0.01,
        max_grad_norm=0.5,
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
            agent_type=A2CAgent,
            output_dir="results/exercise5_a2c/lunar_1d/",
            save_result=True,
            model_filename="a2c_gae.pth",
            repo_id="A2C-GAE-LunarLanderV3",
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
    actor_critic = ActorCritic(obs_dim=obs_shape[0], n_actions=action_n, hidden_size=256)
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
