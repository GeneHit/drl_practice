import abc

import numpy as np
from numpy.typing import NDArray


class PolicyBase(abc.ABC):
    """Base class for all policies."""

    @abc.abstractmethod
    def set_train_flag(self, train_flag: bool) -> None:
        """Set the train flag."""

    @abc.abstractmethod
    def action(
        self, state: int | NDArray[np.float32]
    ) -> int | NDArray[np.float32]:
        """Get the action for the given state."""

    @abc.abstractmethod
    def update(
        self, state: int, action: int, next_state: int, reward: float
    ) -> None:
        """Update the policy."""

    @abc.abstractmethod
    def save(self, pathname: str) -> None:
        """Save the policy."""

    @abc.abstractmethod
    @classmethod
    def load(cls, pathname: str) -> "PolicyBase":
        """Load the policy."""
