from dataclasses import dataclass

import torch

from practice.base.context import ContextBase
from practice.base.trainer import TrainerBase
from practice.exercise9_sac.sac_exercise import SACConfig
from practice.exercise12_mbpo.model_based_env import EnvModel, ModelBasedConfig
from practice.utils_for_coding.replay_buffer_utils import Experience
from practice.utils_for_coding.scheduler_utils import ScheduleBase
from practice.utils_for_coding.writer_utils import CustomWriter


@dataclass(kw_only=True, frozen=True)
class MBPOConfig(SACConfig):
    """The configuration for the MBPO algorithm."""

    epochs: int
    """The number of epochs to train the policy."""

    real_step_per_epoch: int
    """The number of real environment steps per epoch."""

    rollout_num: int
    """The rollout number of the model-based environment every epoch."""

    rollout_len: ScheduleBase
    """The rollout length of the model-based environment every epoch."""

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
            2. train all env models with above boostrap samples
            3. use random model to generate rollout
                - 3.1 get rollout_num samples from real replay
                - 3.2 run rollout_len(epoch) steps for each sample
                - 3.3 buffer the model replay
            4. train the SAC with mixed data in update_num_per_epoch steps
                - 4.1 sample from the model replay with changed batch_rate_of_model_sample
                - 4.2 sample from the real replay with (1 - batch_rate_of_model_sample)
                - 4.3 train the SAC with mixed data
        """
        pass


class _MBPOPod:
    """A pod for the MBPO algorithm."""

    def __init__(self, config: MBPOConfig, ctx: MBPOContext, writer: CustomWriter) -> None:
        self._config = config
        self._ctx = ctx
        self._writer = writer

    def train_env_model(self, experience: Experience, step: int) -> None:
        """Train the environment model."""
        pass

    def generate_rollout(self, experience: Experience, step: int) -> None:
        """Generate rollout."""
        pass

    def train_sac(self, experience: Experience, step: int) -> None:
        """Train the policy network with a vectorized environment."""
        pass
