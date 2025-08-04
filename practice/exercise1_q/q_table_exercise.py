from typing import TypeAlias, Union

import numpy as np
import pickle5 as pickle
import torch
from numpy.typing import NDArray

from practice.base.env_typing import ActType

# Type alias for discrete observations
ObsType: TypeAlias = np.int64


class QTable:
    """Q-table agent."""

    def __init__(self, table: NDArray[np.float32]) -> None:
        self._table = table

    @property
    def array(self) -> NDArray[np.float32]:
        """Get the Q-table for get value and update outside."""
        return self._table

    def action(self, state: Union[ObsType, int, np.integer]) -> ActType:
        """Get the action for evaluation/gameplay with 1 environment."""
        # Always use greedy policy for evaluation
        return np.argmax(self._table[state])

    def sample(self, state: Union[ObsType, int, np.integer], epsilon: float) -> ActType:
        """Sample an action from the Q-table."""
        return _epsilon_greedy_policy(
            q_table=self._table,
            state=int(state),
            epsilon=epsilon,
        )

    def update(
        self, state: int, action: ActType, reward: float, next_state: int, lr: float, gamma: float
    ) -> float:
        """Update the Q-table.

        Parameters
        ----------
            state: The current state.
            action: The action to take.
            reward: The reward received.
            next_state: The next state.
            lr: The learning rate.
            gamma: The discount factor.

        Returns:
            The TD error.
        """
        old_score = self._table[state, action]
        next_max_q = float(max(self._table[next_state]))

        td_target = reward + gamma * next_max_q
        td_error = td_target - old_score

        new_score = old_score + lr * td_error
        self._table[state, action] = new_score

        return float(td_error)

    def only_save_model(self, pathname: str) -> None:
        """Save the Q-table to a file."""
        assert pathname.endswith(".pkl")
        with open(pathname, "wb") as f:
            pickle.dump(self._table, f)

    @classmethod
    def load_from_checkpoint(cls, pathname: str, device: torch.device | None) -> "QTable":
        """Load the Q-table from a file."""
        with open(pathname, "rb") as f:
            table = pickle.load(f)
        return cls(table=table)


def _epsilon_greedy_policy(q_table: NDArray[np.float32], state: int, epsilon: float) -> ActType:
    """Take an action with the epsilon-greedy strategy.

    2 strategies:
    - Exploration: take a random action with probability epsilon.
    - Exploitation: take the action with the highest state, action value.

    Args:
        q_table: The Q-table.
        state: The current state.
        epsilon: The exploration rate.

    Returns:
        The action to take.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(q_table.shape[1], dtype=ActType)
    else:
        return np.argmax(q_table[state])
