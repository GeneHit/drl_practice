from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch.optim.lr_scheduler import LRScheduler

from practice.base.env_typing import EnvsType, EnvsTypeC, EnvType, EnvTypeC


@dataclass(kw_only=True, frozen=True)
class ContextBase:
    train_env: EnvType | EnvsType | EnvsTypeC
    """The environment used for training."""
    eval_env: EnvType | EnvTypeC
    """The environment used for evaluation."""
    trained_target: nn.Module | NDArray[np.float32]
    """The trained dqn/policy/actor network, or q-table."""
    optimizer: torch.optim.Optimizer
    """The optimizer used for training."""
    lr_schedulers: tuple[LRScheduler, ...] = ()
    """The learning rate schedulers used for training."""

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
        """The q-table."""
        assert isinstance(self.trained_target, np.ndarray)
        return self.trained_target

    @property
    def env_state_shape(self) -> tuple[int, ...]:
        obs_shape = self.eval_env.observation_space.shape
        assert obs_shape is not None
        return obs_shape

    def step_lr_schedulers(self) -> None:
        for scheduler in self.lr_schedulers:
            scheduler.step()
