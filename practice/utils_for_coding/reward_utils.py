from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from practice.base.chest import RewardBase, RewardConfig, ScheduleBase


@dataclass(kw_only=True, frozen=True)
class XDirectionShapingRewardConfig(RewardConfig):
    """The configuration for the XDirectionShapingReward."""

    beta: ScheduleBase
    """The beta for the XDirectionShapingReward."""
    goal_position: float | None = None
    """The goal position for the XDirectionShapingReward."""

    def get_rewarder(self) -> RewardBase:
        return XDirectionShapingReward(self)


class XDirectionShapingReward(RewardBase):
    """The shaping reward on the x direction of the MountainCar.

    If goal_position is specified, use potential shaping,
    otherwise use the x direction change controlled by beta.
    """

    def __init__(self, config: XDirectionShapingRewardConfig):
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
