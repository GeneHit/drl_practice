import abc
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ScheduleBase(abc.ABC):
    """Base class for all schedules."""

    @abc.abstractmethod
    def __call__(self, t: int) -> float:
        """Get the value for the given time.

        Usage:
        ```python
        schedule = LinearSchedule(min_e=0.01, max_e=1.0, duration=1000)
        epsilon = schedule(t=100)
        ```
        """


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
