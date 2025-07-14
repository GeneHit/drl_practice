from dataclasses import dataclass

import torch
import torch.nn as nn

from hands_on.exercise2_dqn.dqn_exercise import EnvsType, EnvType


@dataclass(kw_only=True, frozen=True)
class ContextBase:
    env: EnvType | EnvsType
    eval_env: EnvType
    network: nn.Module
    optimizer: torch.optim.Optimizer

    @property
    def env_state_shape(self) -> tuple[int, ...]:
        obs_shape = self.eval_env.observation_space.shape
        assert obs_shape is not None
        return obs_shape
