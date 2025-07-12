import copy
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from hands_on.base import RewardBase


class RNDNetwork1D(nn.Module):
    """RNDNetwork1D is a network that to get a intrinsic reward from the environment."""

    def __init__(self, obs_dim: int, output_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


class RNDReward(RewardBase):
    """RNDReward is a reward that is used to encourage the agent to explore the environment.

    It uses a network to get a intrinsic reward from the environment.
    Paper: https://arxiv.org/abs/1808.04355
        Distills a randomly initialized neural network (target) into a trained (predictor) network.
    """

    def __init__(self, network: nn.Module, device: torch.device, beta: float = 0.1) -> None:
        self._predictor = network
        self._predictor.eval()
        self._target = copy.deepcopy(self._predictor)
        self._target.eval()
        self._device = device
        self._predictor.to(self._device)
        self._target.to(self._device)
        self._beta = beta

    def get_reward(
        self, state: NDArray[Any], next_state: NDArray[Any]
    ) -> NDArray[np.floating[Any]]:
        state_tensor = torch.from_numpy(state).to(self._device)

        with torch.no_grad():
            target_output = self._target(state_tensor)
            pred_output = self._predictor(state_tensor)
            diff = target_output - pred_output

        # normalize the diff with L2 norm
        return cast(
            NDArray[np.floating[Any]], torch.norm(diff.cpu(), p=2, dim=1).numpy() * self._beta
        )
