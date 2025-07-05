import abc
from typing import Any


class PolicyBase(abc.ABC):
    """Base class for all policies."""

    @abc.abstractmethod
    def set_train_flag(self, train_flag: bool) -> None:
        """Set the train flag."""

    @abc.abstractmethod
    def action(self, state: Any, epsilon: float | None = None) -> int:
        """Get the action for the given state.

        Parameters
        ----------
        state : int
            The state to get the action for.
        epsilon : float | None, optional
            The epsilon value to use for the action in training mode.

        Returns
        -------
        The action for the given state: int
        """

    @abc.abstractmethod
    def get_score(self, state: Any, action: Any | None = None) -> float:
        """Get the score for the given state and action.

        If action is None, the score will be the maximum score for the given state.

        Parameters
        ----------
        state : int
            The state to get the score for.
        action : int | None, optional
            The action to get the score for.

        Returns
        -------
        The score for the given state and action: float
        """

    @abc.abstractmethod
    def update(
        self, state: Any | None, action: Any | None, score: float
    ) -> None:
        """Update the policy."""

    @abc.abstractmethod
    def save(self, pathname: str) -> None:
        """Save the policy."""

    @abc.abstractmethod
    @classmethod
    def load(cls, pathname: str) -> "PolicyBase":
        """Load the policy."""
