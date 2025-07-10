import abc
from typing import Any, TypeAlias

import numpy as np
import torch

ActType: TypeAlias = np.int64


class AgentBase(abc.ABC):
    """Base class for all policies-based or value-based agents.

    Notice:
    - ****Only Be Used for Evaluation/Gameplay.****
    - Have to replace the Any type with the actual type in the subclass.
    - It's attribute will have some network when necessary.
    """

    @abc.abstractmethod
    def action(self, state: Any) -> Any:
        """Get the action for the given single state.

        # TODO: have a better type for action output.

        Parameters
        ----------
        state : int
            The state to get the action for.
        step : int
            The timestep of the training process.

        Returns
        -------
        The action for the given state: int
        """

    @abc.abstractmethod
    def only_save_model(self, pathname: str) -> None:
        """Save the policy."""

    @classmethod
    @abc.abstractmethod
    def load_from_checkpoint(
        cls, pathname: str, device: torch.device | None
    ) -> "AgentBase":
        """Load the policy from a checkpoint."""


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
