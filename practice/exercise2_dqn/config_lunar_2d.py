from typing import Union

import torch
from gymnasium.spaces import Discrete
from torch.optim import Adam

from hands_on.exercise2_dqn.dqn_exercise import EnvType
from hands_on.utils.agent_utils import NNAgent
from hands_on.utils.env_utils import get_device
from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.exercise2_dqn.dqn_exercise import (
    DQNConfig,
    DQNTrainer,
    QNet1D,
    QNet2D,
)
from practice.utils.env_utils import get_env_from_config


def get_app_config() -> DQNConfig:
    """Get the application config."""
    # get cuda or mps if available
    device = get_device()
    return DQNConfig(
        device=device,
        timesteps=20000,
        learning_rate=1e-4,
        gamma=0.99,
        start_epsilon=1.0,
        end_epsilon=0.01,
        exploration_fraction=0.1,
        replay_buffer_capacity=40000,
        batch_size=64,
        train_interval=2,
        target_update_interval=1000,
        update_start_step=1000,
        eval_episodes=100,
        eval_random_seed=42,
        eval_video_num=10,
        env_config=EnvConfig(
            env_id="LunarLander-v3",
            vector_env_num=6,
            use_multi_processing=True,
            use_image=True,
            training_render_mode="rgb_array",
            image_shape=(84, 84),
            frame_stack=4,
            record_eval_video=True,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=DQNTrainer,
            agent_type=NNAgent,
            output_dir="results/exercise2_dqn/lunar_2d/",
            save_result=True,
            model_filename="dqn.pth",
            repo_id="dqn-2d-LunarLander-v3",
            algorithm_name="DQN",
            extra_tags=("deep-q-learning", "pytorch"),
        ),
    )


def generate_context(config: DQNConfig) -> ContextBase:
    """Generate the context for the training."""
    env, eval_env = get_env_from_config(config.env_config)
    # env can be either single env or vector env depending on vector_env_num
    # eval_env is always a single env

    obs_shape = eval_env.observation_space.shape
    assert obs_shape is not None
    assert isinstance(eval_env.action_space, Discrete)
    action_n = int(eval_env.action_space.n)

    # Create Q-network based on observation space
    q_network: Union[QNet1D, QNet2D]
    if len(obs_shape) == 1:
        q_network = QNet1D(state_n=obs_shape[0], action_n=action_n)
    elif len(obs_shape) == 3:
        q_network = QNet2D(in_shape=obs_shape, action_n=action_n)
    else:
        raise ValueError(f"Unsupported observation space shape: {obs_shape}")

    # Load checkpoint if exists
    if config.checkpoint_pathname:
        checkpoint = torch.load(config.checkpoint_pathname, weights_only=False)
        if isinstance(checkpoint, dict):
            # It's a state_dict
            q_network.load_state_dict(checkpoint)
        else:
            # It's a full model, extract state_dict
            q_network.load_state_dict(checkpoint.state_dict())
    q_network.to(config.device)

    return ContextBase(
        train_env=env,
        eval_env=eval_env,
        trained_target=q_network,
        optimizer=Adam(q_network.parameters(), lr=config.learning_rate),
    )


def get_env_for_play_and_hub(config: DQNConfig) -> EnvType:
    """Get the environment for play and hub."""
    _, eval_env = get_env_from_config(config.env_config)
    return eval_env
