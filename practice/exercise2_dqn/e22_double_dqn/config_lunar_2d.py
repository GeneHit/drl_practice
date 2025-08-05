from gymnasium.spaces import Discrete
from torch.optim import Adam

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.exercise2_dqn.dqn_exercise import DQNConfig
from practice.exercise2_dqn.dqn_trainer import DQNTrainer
from practice.exercise2_dqn.e22_double_dqn.double_dqn_exercise import QNet2D
from practice.utils.env_utils import get_device, get_env_from_config
from practice.utils_for_coding.network_utils import load_checkpoint_if_exists
from practice.utils_for_coding.scheduler_utils import LinearSchedule


def get_app_config() -> DQNConfig:
    """Get the application config."""
    device = get_device("cpu")
    timesteps = 20000
    return DQNConfig(
        device=device,
        dqn_algorithm="double",
        timesteps=timesteps,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_schedule=LinearSchedule(start_e=1.0, end_e=0.01, duration=int(0.1 * timesteps)),
        replay_buffer_capacity=40000,
        batch_size=64,
        train_interval=2,
        target_update_interval=1000,
        update_start_step=1000,
        eval_episodes=100,
        eval_random_seed=42,
        eval_video_num=10,
        log_interval=50,
        env_config=EnvConfig(
            env_id="LunarLander-v3",
            vector_env_num=6,
            use_multi_processing=True,
            use_image=True,
            training_render_mode="rgb_array",
            image_shape=(84, 84),
            frame_stack=4,
            # frame_skip=4,
            # LunarLander-v3 default max_steps is 200
            # max_steps=100,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=DQNTrainer,
            output_dir="results/exercise2_dqn/double_dqn/lunar_2d/",
            save_result=True,
            repo_id="DoubleDQN-2d-LunarLander-v3",
            algorithm_name="Double-DQN",
            extra_tags=("deep-q-learning", "pytorch", "image"),
            usage_instructions="Don't forget to check the necessary wrappers in the env setup.",
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

    # Create Q-network
    assert len(obs_shape) == 3
    q_network = QNet2D(in_shape=obs_shape, action_n=action_n)

    load_checkpoint_if_exists(q_network, config.checkpoint_pathname)
    q_network.to(config.device)

    return ContextBase(
        train_env=env,
        eval_env=eval_env,
        trained_target=q_network,
        optimizer=Adam(q_network.parameters(), lr=config.learning_rate),
    )
