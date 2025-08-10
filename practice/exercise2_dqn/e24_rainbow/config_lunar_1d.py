from gymnasium.spaces import Discrete
from torch.optim import Adam

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.exercise2_dqn.dqn_trainer import DQNTrainer
from practice.exercise2_dqn.e24_rainbow.per_exercise import PERBufferConfig
from practice.exercise2_dqn.e24_rainbow.rainbow_exercise import RainbowConfig, RainbowNet
from practice.utils.env_utils import get_device, get_env_from_config
from practice.utils_for_coding.network_utils import load_checkpoint_if_exists
from practice.utils_for_coding.scheduler_utils import ConstantSchedule


def get_app_config() -> RainbowConfig:
    """Get the application config."""
    # get cuda or mps if available
    device = get_device("cpu")
    global_steps = 1500_000
    num_envs = 6
    # 1500_000 / 6 = 250_000
    timesteps = global_steps // num_envs
    return RainbowConfig(
        device=device,
        dqn_algorithm="rainbow",
        timesteps=timesteps // 2,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        train_interval=1,
        target_update_interval=250,
        update_start_step=2000,
        max_grad_norm=10.0,
        per_buffer_config=PERBufferConfig(
            capacity=int(global_steps * 0.1),
            n_step=3,
            gamma=0.99,
            use_uniform_sampling=True,
            alpha=0.6,
            beta=0.4,
            # ~= (1 - beta) / (timesteps * anneal_fraction) = 0.6 / (250_000 * 0.5)
            beta_increment=3e-6,
        ),
        noisy_std=0.5,
        v_min=-300.0,
        v_max=300.0,
        num_atoms=51,
        eval_episodes=100,
        eval_random_seed=42,
        eval_video_num=10,
        log_interval=100,
        env_config=EnvConfig(
            env_id="LunarLander-v3",
            vector_env_num=num_envs,
            use_multi_processing=True,
        ),
        artifact_config=ArtifactConfig(
            trainer_type=DQNTrainer,
            output_dir="results/exercise2_dqn/rainbow/lunar_1d/",
            save_result=True,
            repo_id="Rainbow-1d-LunarLander-v3",
            algorithm_name="Rainbow-DQN",
            extra_tags=("deep-q-learning", "pytorch", "rainbow", "dqn"),
        ),
        # unsed epsilon
        epsilon_schedule=ConstantSchedule(0.0),
        replay_buffer_capacity=0,  # unused, use per_buffer_config instead
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
        v_min=config.v_min,
        v_max=config.v_max,
        num_atoms=config.num_atoms,
    )

    load_checkpoint_if_exists(q_network, config.checkpoint_pathname)
    q_network.to(config.device)

    return ContextBase(
        train_env=env,
        eval_env=eval_env,
        trained_target=q_network,
        optimizer=Adam(q_network.parameters(), lr=config.learning_rate),
    )
