from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
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
    track_and_evaluate: bool = True
    """Whether to track and evaluate the training.

    If False, will not record the training tensorboard and evaluate the trained model.
    It is useful when train the model with distributed training.
    """

    @property
    def env(self) -> EnvType:
        """The discrete-action environment used for training."""
        # Check if it's a vector environment by checking for num_envs attribute
        env = cast(EnvType, self.train_env)
        if hasattr(env, "num_envs"):
            raise TypeError("train_env is a vector environment, use envs property instead")
        assert isinstance(env.action_space, Discrete), "Env must be discrete action space"
        return env

    @property
    def envs(self) -> EnvsType:
        """The discrete vector environment used for training."""
        # Check if it's a vector environment by checking for num_envs attribute
        envs = cast(EnvsType, self.train_env)
        if not hasattr(envs, "num_envs"):
            raise TypeError("train_env is a single environment, use env property instead")
        assert isinstance(envs.single_action_space, Discrete), "Env must be discrete action space"
        return envs

    @property
    def continuous_envs(self) -> EnvsTypeC:
        """The vector environment (with continuous action space) used for training."""
        envs = cast(EnvsTypeC, self.train_env)
        if not hasattr(envs, "num_envs"):
            raise TypeError("train_env is a single environment, use env property instead")
        assert isinstance(envs.single_action_space, Box), "Env must be continuous action space"
        return envs

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
