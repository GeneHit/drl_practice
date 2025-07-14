from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from hands_on.exercise2_dqn.dqn_exercise import EnvsType, EnvType


@dataclass(kw_only=True, frozen=True)
class ContextBase:
    env: EnvType | EnvsType
    eval_env: EnvType
    trained_target: nn.Module | NDArray[np.float32]
    optimizer: torch.optim.Optimizer

    @property
    def network(self) -> nn.Module:
        assert isinstance(self.trained_target, nn.Module)
        return self.trained_target

    @property
    def table(self) -> NDArray[np.float32]:
        assert isinstance(self.trained_target, np.ndarray)
        return self.trained_target

    @property
    def env_state_shape(self) -> tuple[int, ...]:
        obs_shape = self.eval_env.observation_space.shape
        assert obs_shape is not None
        return obs_shape
