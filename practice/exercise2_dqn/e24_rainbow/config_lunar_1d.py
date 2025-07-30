from gymnasium.spaces import Discrete
from torch.optim import Adam

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.base.env_typing import EnvType
from practice.exercise2_dqn.dqn_trainer import DQNTrainer
from practice.exercise2_dqn.e24_rainbow.rainbow_exercise import RainbowConfig, RainbowNet
from practice.utils.env_utils import get_device, get_env_from_config
from practice.utils_for_coding.agent_utils import NNAgent
from practice.utils_for_coding.network_utils import load_checkpoint_if_exists
from practice.utils_for_coding.scheduler_utils import LinearSchedule


def get_app_config() -> RainbowConfig:
    """Get the application config."""
    # get cuda or mps if available
    device = get_device("cpu")
    timesteps = 250_000
    return RainbowConfig(
        device=device,
        dqn_algorithm="rainbow",
        timesteps=timesteps,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_schedule=LinearSchedule(start_e=1.0, end_e=0.01, duration=int(0.6 * timesteps)),
        replay_buffer_capacity=120_000,
        batch_size=64,
        train_interval=1,
        target_update_interval=250,
        update_start_step=2000,
        n_step=3,
        alpha=0.6,
        beta=0.4,
        beta_increment=1e-6,
        noisy_std=0.5,
        eval_episodes=100,
        eval_random_seed=42,
        eval_video_num=10,
        log_interval=50,
        env_config=EnvConfig(
            env_id="LunarLander-v3",
            vector_env_num=6,
            use_multi_processing=True,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=DQNTrainer,
            agent_type=NNAgent,
            output_dir="results/exercise2_dqn/lunar_1d/",
            save_result=True,
            repo_id="DQN-1d-LunarLander-v3",
            algorithm_name="DQN",
            extra_tags=("deep-q-learning", "pytorch"),
            usage_instructions="Please check the necessary wrappers in the env setup.",
        ),
    )


def generate_context(config: RainbowConfig) -> ContextBase:
    """Generate the context for the training."""
    env, eval_env = get_env_from_config(config.env_config)
    # env can be either single env or vector env depending on vector_env_num
    # eval_env is always a single env

    obs_shape = eval_env.observation_space.shape
    assert obs_shape is not None
    assert isinstance(eval_env.action_space, Discrete)
    action_n = int(eval_env.action_space.n)

    # Create Q-network based on observation space
    assert len(obs_shape) == 1
    q_network = RainbowNet(
        state_n=obs_shape[0],
        action_n=action_n,
        hidden_sizes=(256, 256),
        noisy_std=config.noisy_std,
    )

    load_checkpoint_if_exists(q_network, config.checkpoint_pathname)
    q_network.to(config.device)

    return ContextBase(
        train_env=env,
        eval_env=eval_env,
        trained_target=q_network,
        optimizer=Adam(q_network.parameters(), lr=config.learning_rate),
    )


def get_env_for_play_and_hub(config: RainbowConfig) -> EnvType:
    """Get the environment for play and hub."""
    train_env, eval_env = get_env_from_config(config.env_config)
    train_env.close()
    return eval_env
