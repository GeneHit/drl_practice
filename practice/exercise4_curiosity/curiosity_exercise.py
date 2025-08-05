from dataclasses import dataclass
from typing import Any, Sequence, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from numpy.typing import NDArray

from practice.base.rewarder import RewardBase, RewardConfig
from practice.utils_for_coding.network_utils import MLP
from practice.utils_for_coding.scheduler_utils import ScheduleBase


class RNDNetwork1D(nn.Module):
    """RNDNetwork1D is a network that to get a intrinsic reward from the environment."""

    def __init__(self, obs_dim: int, output_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=obs_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            activation=nn.ReLU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.net(x))


@dataclass(kw_only=True, frozen=True)
class RNDRewardConfig:
    """The configuration for the RNDReward."""

    beta: ScheduleBase
    """The beta for the RNDReward."""
    normalize: bool = True
    """Whether to normalize the reward."""
    device: torch.device
    """The device to run the neural network."""
    max_reward: float
    """The maximum reward for the RNDReward."""


class RNDReward(RewardBase):
    """RNDReward is a reward that is used to encourage the agent to explore the environment.

    It uses a network to get a intrinsic reward from the environment.
    Paper: https://arxiv.org/abs/1808.04355
        Distills a randomly initialized neural network (target) into a trained (predictor) network.
    """

    def __init__(
        self,
        config: RNDRewardConfig,
        predictor: nn.Module,
        target: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self._config = config
        self._predictor = predictor
        self._target = target
        self._optimizer = optimizer

        self._predictor.train()
        # freeze target network
        self._target.eval()
        for p in self._target.parameters():
            p.requires_grad = False

        self._running_mean = 0.0
        self._running_var = 0.0
        self._count = 1e-4

    def save(self, path: str) -> None:
        """Save for continuing training."""
        torch.save(
            {
                "predictor": self._predictor.state_dict(),
                "target": self._target.state_dict(),
            },
            path,
        )

    def get_reward(
        self, state: NDArray[Any], next_state: NDArray[Any], step: int
    ) -> NDArray[np.floating[Any]]:
        """Get the intrinsic reward for given states and update the predictor network."""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        state_tensor = torch.from_numpy(state).float().to(self._config.device)
        # fititng predictor network to the freezed target network
        with torch.no_grad():
            target_output = self._target(state_tensor)
        pred_output = self._predictor(state_tensor)
        diff = pred_output - target_output
        # or torch.nn.functional.mse_loss(pred_output, target_output)
        loss = torch.mean(diff.pow(2))
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        # use L2 norm of diff as intrinsic reward (per batch sample)
        intrinsic = torch.norm(diff.detach(), p=2, dim=1).cpu().numpy()
        if self._config.normalize:
            intrinsic = self._norm(intrinsic)
        intrinsic = np.clip(intrinsic * self._config.beta(step), 0, self._config.max_reward)
        return cast(NDArray[np.floating[Any]], intrinsic)

    def _norm(self, x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = len(x)

        total_count = self._count + batch_count

        self._running_mean = self._running_mean * self._count + batch_mean * batch_count
        self._running_mean /= total_count

        self._running_var = self._running_var * self._count + batch_var * batch_count
        self._running_var /= total_count

        self._count = total_count

        std = np.sqrt(self._running_var) + 1e-8
        return cast(NDArray[np.floating[Any]], (x - self._running_mean) / std)


@dataclass(kw_only=True, frozen=True)
class RND1DNetworkConfig(RewardConfig):
    """A configuration of a simple 1D-Envs RNDReward."""

    rnd_config: RNDRewardConfig
    """The device to run the neural network."""
    obs_dim: int
    """Observation dimension for creating networks."""
    output_dim: int = 32
    """Output dimension for the RND networks."""
    hidden_sizes: tuple[int, ...]
    """Hidden dimension for the RND networks."""
    learning_rate: float = 1e-3
    """Learning rate for the predictor optimizer."""

    def get_rewarder(self) -> RewardBase:
        """Create and return an RNDReward instance."""
        device = self.rnd_config.device
        predictor = RNDNetwork1D(
            obs_dim=self.obs_dim, output_dim=self.output_dim, hidden_sizes=self.hidden_sizes
        ).to(device)
        target = RNDNetwork1D(
            obs_dim=self.obs_dim, output_dim=self.output_dim, hidden_sizes=self.hidden_sizes
        ).to(device)
        optimizer = torch.optim.Adam(predictor.parameters(), lr=self.learning_rate)

        return RNDReward(
            config=self.rnd_config,
            predictor=predictor,
            target=target,
            optimizer=optimizer,
        )


@dataclass(kw_only=True, frozen=True)
class XShapingRewardConfig(RewardConfig):
    """The configuration for the XShapingReward."""

    beta: ScheduleBase
    """The beta for the XDirectionShapingReward."""
    goal_position: float | None = None
    """The goal position for the XDirectionShapingReward."""

    def get_rewarder(self) -> RewardBase:
        return XShapingReward(self)


class XShapingReward(RewardBase):
    """The shaping reward on the x direction of the MountainCar.

    If goal_position is specified, use potential shaping,
    otherwise use the x direction change controlled by beta.
    """

    def __init__(self, config: XShapingRewardConfig) -> None:
        self._goal_position = config.goal_position
        self._beta = config.beta

    def get_reward(
        self, state: NDArray[Any], next_state: NDArray[Any], step: int
    ) -> NDArray[np.floating[Any]]:
        # support batch and single
        pos_before = state[..., 0]
        pos_after = next_state[..., 0]

        if self._goal_position is not None:
            shaping = np.abs(self._goal_position - pos_before) - np.abs(
                self._goal_position - pos_after
            )
        else:
            shaping = pos_after - pos_before

        reward: NDArray[np.floating[Any]] = self._beta(step) * shaping
        return reward.astype(np.float32)
