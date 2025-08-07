import numpy as np
import torch
from gymnasium.spaces import Discrete

from practice.base.config import ArtifactConfig, EnvConfig
from practice.base.context import ContextBase
from practice.exercise1_q.q_table_exercise import QTable
from practice.exercise1_q.q_trainer_exercise import EnvType, QTableConfig, QTableTrainer
from practice.utils.env_utils import get_device, make_discrete_env_with_kwargs
from practice.utils_for_coding.scheduler_utils import ExponentialSchedule


def get_app_config() -> QTableConfig:
    """Get the application config."""
    # get cuda or mps if available (though Q-learning doesn't use it)
    device = get_device()
    return QTableConfig(
        device=device,
        episodes=25000,
        learning_rate=0.7,
        gamma=0.95,
        epsilon_schedule=ExponentialSchedule(v0=0.05, v1=0.01, decay_rate=-0.0005),
        eval_episodes=100,
        eval_random_seed=42,
        eval_video_num=10,
        env_config=EnvConfig(
            env_id="Taxi-v3",
            max_steps=99,
            env_kwargs={
                "render_mode": "rgb_array",
            },
        ),
        artifact_config=ArtifactConfig(
            trainer_type=QTableTrainer,
            output_dir="results/exercise1_q/taxi/",
            save_result=True,
            state_dict_filename="table_numpy.pkl",
            fps=2,
            repo_id="q-Taxi-v3",
            algorithm_name="Q-Learning",
            extra_tags=("tabular",),
        ),
    )


def generate_context(config: QTableConfig) -> ContextBase:
    """Generate the context for the training."""
    # Use specialized discrete environment creation for Q-learning
    train_env = make_discrete_env_with_kwargs(
        env_id=config.env_config.env_id,
        kwargs=config.env_config.env_kwargs or {},
        max_steps=config.env_config.max_steps,
    )

    # Create evaluation environment with rgb_array render mode
    eval_env = _get_eval_env(config)

    # Verify discrete spaces
    assert isinstance(train_env.observation_space, Discrete)
    assert isinstance(train_env.action_space, Discrete)
    assert isinstance(eval_env.observation_space, Discrete)
    assert isinstance(eval_env.action_space, Discrete)

    obs_n = int(train_env.observation_space.n)
    action_n = int(train_env.action_space.n)

    # Create Q-table
    if config.checkpoint_pathname:
        # Load from checkpoint
        q_table = QTable.load_from_checkpoint(config.checkpoint_pathname, config.device)
    else:
        # Initialize new Q-table
        q_table = QTable(table=np.zeros((obs_n, action_n), dtype=np.float32))

    # Create dummy optimizer (not used in Q-learning but required by ContextBase)
    dummy_param = torch.tensor(0.0, requires_grad=True)
    dummy_optimizer = torch.optim.Adam([dummy_param], lr=config.learning_rate)

    return ContextBase(
        train_env=train_env,  # type: ignore
        eval_env=eval_env,  # type: ignore
        trained_target=q_table,
        optimizer=dummy_optimizer,
    )


def _get_eval_env(config: QTableConfig) -> EnvType:
    eval_kwargs = (config.env_config.env_kwargs or {}).copy()
    # always use rgb_array for evaluation/play
    eval_kwargs["render_mode"] = "rgb_array"
    eval_env = make_discrete_env_with_kwargs(
        env_id=config.env_config.env_id,
        kwargs=eval_kwargs,
        max_steps=config.env_config.max_steps,
    )
    return eval_env
