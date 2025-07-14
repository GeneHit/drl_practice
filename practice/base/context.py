from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from hands_on.exercise2_dqn.dqn_exercise import EnvsType, EnvType

if TYPE_CHECKING:
    from practice.base.trainer import TrainerBase


@dataclass(kw_only=True, frozen=True)
class ContextBase:
    env: EnvType | EnvsType
    eval_env: EnvType
    network: nn.Module
    optimizer: torch.optim.Optimizer
    trainer_name: type[TrainerBase]

    @property
    def env_state_shape(self) -> tuple[int, ...]:
        obs_shape = self.eval_env.observation_space.shape
        assert obs_shape is not None
        return obs_shape
