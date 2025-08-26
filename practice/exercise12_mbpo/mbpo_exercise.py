from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader
from tqdm import tqdm

from practice.base.context import ContextBase
from practice.base.env_typing import ActTypeC, ObsType
from practice.base.trainer import TrainerBase
from practice.exercise9_sac.sac_exercise import SACConfig
from practice.exercise12_mbpo.model_based_env import EnvModel, ModelBasedConfig
from practice.utils_for_coding.buffer import Experience, ReplayBuffer
from practice.utils_for_coding.scheduler_utils import ScheduleBase
from practice.utils_for_coding.writer_utils import CustomWriter


@dataclass(kw_only=True, frozen=True)
class MBPOConfig(SACConfig):
    """The configuration for the MBPO algorithm."""

    train_interval: int
    """The interval of training the env model and SAC."""

    rollout_num: int
    """The rollout number of the model-based environment every epoch."""

    rollout_len: ScheduleBase
    """The generated rollout length of the model-based environment every epoch."""

    update_num_per_epoch: int
    """The update number of the SAC every epoch."""

    batch_rate_of_model_sample: ScheduleBase
    """The batch rate of the model sample."""

    model_based_config: ModelBasedConfig
    """The configuration for the model-based environment."""

    model_replay_buffer_capacity: int
    """The capacity of the model replay buffer."""


@dataclass(kw_only=True, frozen=True)
class MBPOContext(ContextBase):
    """The context for the MBPO algorithm."""

    critic: torch.nn.Module
    """The critic network."""

    critic_optimizer: torch.optim.Optimizer
    """The optimizer for the critic."""

    env_model: EnvModel
    """The environment model."""


class MBPOTrainer(TrainerBase):
    """A trainer for the MBPO algorithm."""

    def __init__(self, config: MBPOConfig, ctx: MBPOContext) -> None:
        super().__init__(config=config, ctx=ctx)
        self._config: MBPOConfig = config
        self._ctx: MBPOContext = ctx

    def train(self) -> None:
        """Train the policy network with a vectorized environment.

        Steps:
        for epoch in 1...epochs:
            1. step the real environment for real_step_per_epoch steps, and buffer the replay
            2. train all env models with real replay buffer
            3. use random model to generate rollout
                - 3.1 get rollout_num samples from real replay
                - 3.2 run rollout_len(epoch) steps for each sample
                - 3.3 buffer the model replay
            4. train the SAC with mixed data in update_num_per_epoch steps
                - 4.1 sample from the model replay with changed batch_rate_of_model_sample
                - 4.2 sample from the real replay with (1 - batch_rate_of_model_sample)
                - 4.3 train the SAC with mixed data
        """
        # Initialize tensorboard writer
        writer = CustomWriter(
            track=self._ctx.track_and_evaluate,
            log_dir=self._config.artifact_config.get_tensorboard_dir(),
        )
        # Use environment from context - must be vector environment
        envs = self._ctx.continuous_envs

        # Create trainer pod
        pod = _MBPOPod(config=self._config, ctx=self._ctx, writer=writer)
        # Create replay buffer using observation shape from context
        obs_dtype = envs.single_observation_space.dtype
        assert obs_dtype in (np.float32, np.uint8)
        assert envs.single_action_space.dtype == np.float32
        env_buffer = ReplayBuffer(capacity=self._config.replay_buffer_capacity)

        # Initialize environments
        states, _ = envs.reset()
        assert isinstance(states, np.ndarray), "States must be numpy array"
        # Track previous step terminal status to avoid invalid transitions
        prev_dones: NDArray[np.bool_] = np.zeros(envs.num_envs, dtype=np.bool_)
        episode_steps = 0

        # loop
        timestep = self._config.total_steps // envs.num_envs
        start_step = self._config.update_start_step // envs.num_envs
        for step in tqdm(range(timestep), desc="Training"):
            # Get actions for all environments
            actions = pod.action(states, step)
            # Step the environment
            next_states, rewards, terminated, truncated, infos = envs.step(actions)

            # Cast rewards to numpy array for indexing
            rewards = np.asarray(rewards, dtype=np.float32)
            # Handle terminal observations and create proper training transitions
            dones = np.logical_or(terminated, truncated, dtype=np.bool_)

            # Only store transitions for states that were not terminal in the previous step
            # we use AutoReset wrapper, so the envs will be reset automatically when it's done
            # when any done in n step, the next_states of n+1 step is the first of the next episode
            pre_non_terminal_mask = ~prev_dones
            if np.any(pre_non_terminal_mask):
                # Create training transitions
                env_buffer.add_batch(
                    states=states[pre_non_terminal_mask],
                    actions=actions[pre_non_terminal_mask],
                    rewards=rewards[pre_non_terminal_mask],
                    next_states=next_states[pre_non_terminal_mask],
                    dones=dones[pre_non_terminal_mask],
                )

            states = next_states
            prev_dones = dones
            # Log episode metrics
            episode_steps += writer.log_episode_stats_if_has(infos, episode_steps)

            # Training updates
            if step >= start_step and step % self._config.train_interval == 0:
                if len(env_buffer) < self._config.batch_size:
                    continue

                # 2. train all env models
                pod.train_env_model(
                    dataloader=env_buffer.dataloader(
                        batch_size=self._config.model_based_config.train.batch_size,
                        shuffle=True,
                        num_workers=2,
                        pin_memory=True,
                    ),
                    step=step,
                )

                # 3. use random model to generate rollout
                pod.generate_rollout(
                    real_data=env_buffer.sample(self._config.rollout_num), step=step
                )

                # 4. train the SAC with mixed data
                pod.train_sac(real_data=env_buffer.sample(pod.num_for_real_data(step)), step=step)

        writer.close()


class _MBPOPod:
    """A pod for the MBPO algorithm."""

    def __init__(self, config: MBPOConfig, ctx: MBPOContext, writer: CustomWriter) -> None:
        self._config = config
        self._ctx = ctx
        self._writer = writer

        self._model_buffer = ReplayBuffer(capacity=config.model_replay_buffer_capacity)

    def num_for_real_data(self, step: int) -> int:
        """Get the number to get real data for training SAC."""
        return int(self._config.batch_size * (1 - self._config.batch_rate_of_model_sample(step)))

    def action(self, state: NDArray[ObsType], step: int) -> NDArray[ActTypeC]:
        """Get actions for all environments."""
        raise NotImplementedError("Not implemented")

    def train_env_model(self, dataloader: DataLoader[dict[str, torch.Tensor]], step: int) -> None:
        """Train the environment model with real data."""
        raise NotImplementedError("Not implemented")

    def generate_rollout(self, real_data: Experience, step: int) -> None:
        """Generate rollout and buffer it."""
        raise NotImplementedError("Not implemented")

    def train_sac(self, real_data: Experience, step: int) -> None:
        """Train the policy network with mixed data.

        Args:
            real_data: The experience from real data. It will be mixed with model data inside.
            step: The current step.
        """
        raise NotImplementedError("Not implemented")
