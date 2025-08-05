from dataclasses import dataclass

import gymnasium as gym
from tqdm import tqdm

from practice.base.config import BaseConfig
from practice.base.context import ContextBase
from practice.base.env_typing import ActType
from practice.base.trainer import TrainerBase
from practice.exercise1_q.q_table_exercise import ObsType
from practice.utils_for_coding.scheduler_utils import ScheduleBase
from practice.utils_for_coding.writer_utils import CustomWriter

# Type alias for single environment
EnvType = gym.Env[ObsType, ActType]


@dataclass(kw_only=True, frozen=True)
class QTableConfig(BaseConfig):
    """Configuration for Q-table learning algorithm."""

    episodes: int
    min_epsilon: float = 0.05
    max_epsilon: float = 1.0
    epsilon_schedule: ScheduleBase


class QTableTrainer(TrainerBase):
    """A trainer for the Q-table algorithm."""

    def __init__(self, config: QTableConfig, ctx: ContextBase) -> None:
        super().__init__(config=config, ctx=ctx)
        self._config: QTableConfig = config
        self._ctx: ContextBase = ctx
        self._q_table = ctx.table

    def train(self) -> None:
        """Train the Q-table agent."""
        # Initialize tensorboard writer
        writer = CustomWriter(
            track=True, log_dir=self._config.artifact_config.get_tensorboard_dir()
        )

        # Use single environment from context
        env = self._ctx.env

        # Training loop
        global_steps = 0
        for episode in tqdm(range(self._config.episodes), desc="Training"):
            state_raw, _ = env.reset()
            # Convert state to int for Q-table indexing
            state = int(state_raw)
            episode_reward = 0.0
            episode_step = 0

            done = False
            while not done:
                episilon = self._config.epsilon_schedule(episode)
                action = self._q_table.sample(state=state, epsilon=episilon)
                next_state_raw, reward, terminated, truncated, _ = env.step(action)
                # Convert next_state to int for Q-table indexing
                next_state = int(next_state_raw)
                td_error = self._q_table.update(
                    state=state,
                    action=action,
                    reward=float(reward),
                    next_state=next_state,
                    lr=self._config.learning_rate,
                    gamma=self._config.gamma,
                )

                # Log training metrics

                writer.log_stats(
                    data={
                        "training/td_error": td_error,
                        "training/epsilon": episilon,
                    },
                    step=global_steps,
                    log_interval=self._config.log_interval,
                    blocked=True,
                )

                done = bool(terminated or truncated)
                episode_reward += float(reward)
                episode_step += 1
                state = next_state
                global_steps += 1

            writer.log_stats(
                data={
                    "episode/reward": episode_reward,
                    "episode/length": episode_step,
                },
                step=episode,
                log_interval=self._config.log_interval // 10,
                blocked=False,
            )

        # Cleanup
        writer.close()
