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

        TODO: support batch processing.

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

        TODO: support batch processing.

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
        self, state: Any | None, action: Any | None, reward_target: Any
    ) -> None:
        """Update the policy.

        Parameters
        ----------
        state : Any | None
            The state to update the policy/model for.
        action : Any | None
            The action to update the policy/model for.
        reward_target : Any
            The reward target to update the policy/model for. Cases:
            1. for q_table, it is the float TD score.
            2. for dqn, it is the tensor TD target from the target network.
        """

    @abc.abstractmethod
    def save(self, pathname: str) -> None:
        """Save the policy."""

    @classmethod
    def load(cls, pathname: str) -> "PolicyBase":
        """Load the policy."""
        raise NotImplementedError(
            "This method should be implemented by the subclass."
        )


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
