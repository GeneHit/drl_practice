import abc
from typing import Any

import numpy as np
from numpy.typing import NDArray


class RewardBase(abc.ABC):
    """Base class for the personal rewards."""

    @abc.abstractmethod
    def get_reward(
        self, state: NDArray[Any], next_state: NDArray[Any], step: int
    ) -> NDArray[np.floating[Any]]:
        """Get the reward for the given state and action."""


class RewardConfig(abc.ABC):
    """Base class for the reward configuration."""

    @abc.abstractmethod
    def get_rewarder(self) -> RewardBase:
        """Get the rewarder."""
