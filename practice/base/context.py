from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from practice.base.env_typing import EnvsType, EnvType


@dataclass(kw_only=True, frozen=True)
class ContextBase:
    train_env: EnvType | EnvsType
    """The environment used for training."""
    eval_env: EnvType
    """The environment used for evaluation."""
    trained_target: nn.Module | NDArray[np.float32]
    """The trained policy/q-value network, or q-table."""
    optimizer: torch.optim.Optimizer
    """The optimizer used for training."""

    @property
    def env(self) -> EnvType:
        # Check if it's a vector environment by checking for num_envs attribute
        if hasattr(self.train_env, "num_envs"):
            raise TypeError("train_env is a vector environment, use envs property instead")
        return cast(EnvType, self.train_env)

    @property
    def envs(self) -> EnvsType:
        # Check if it's a vector environment by checking for num_envs attribute
        if not hasattr(self.train_env, "num_envs"):
            raise TypeError("train_env is a single environment, use env property instead")
        return cast(EnvsType, self.train_env)

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
