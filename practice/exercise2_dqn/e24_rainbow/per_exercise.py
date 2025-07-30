from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

from practice.utils_for_coding.replay_buffer_utils import Experience


class PERBuffer:
    """Prioritized Experience Replay Buffer with n-step return.

    Reference:
    - https://arxiv.org/abs/1511.05952

    Args:
        capacity: The capacity of the buffer.
        n_step: The n-step return.
        gamma: The discount factor.
        alpha: The alpha parameter for prioritized experience replay.
        beta: The beta parameter for prioritized experience replay.
        beta_increment: The increment of beta per update.
    """

    def __init__(
        self,
        capacity: int,
        n_step: int,
        gamma: float,
        alpha: float,
        beta: float,
        beta_increment: float,
    ) -> None:
        self._capacity = capacity
        self._n_step = n_step
        self._gamma = gamma
        self._alpha = alpha
        self._beta = beta
        self._beta_increment = beta_increment
        # TODO: define experience type
        self._n_step_buffer = deque(maxlen=n_step)  # type: ignore
        self._max_priority = 1.0
        self._sum_tree = _SumTree(capacity)

    def __len__(self) -> int:
        return len(self._sum_tree)

    def add(
        self,
        states: NDArray[Any],
        actions: NDArray[Any],
        rewards: NDArray[Any],
        next_states: NDArray[Any],
        dones: NDArray[np.bool_],
    ) -> None:
        raise NotImplementedError("Not implemented")

    def sample(
        self,
        batch_size: int,
    ) -> tuple[Experience, NDArray[np.float32], NDArray[np.int32]]:
        """Sample a batch of experiences.

        Returns:
            tuple[Experience, NDArray[np.float32], NDArray[np.int32]]:
                - Experience: The sampled experiences.
                - weights: The weights of the sampled experiences.
                - indices: The indices of the sampled experiences.
        """
        raise NotImplementedError("Not implemented")

    def update_priorities(
        self,
        indices: NDArray[np.int32],
        priorities: NDArray[np.float32],
    ) -> None:
        """Update the priorities of the sampled experiences.

        Args:
            indices: The indices of the sampled experiences.
            priorities: The priorities of the sampled experiences.
        """
        raise NotImplementedError("Not implemented")


class _SumTree:
    """SumTree for prioritized experience replay.

    SumTree is a binary tree where the parent node is the sum of the two child nodes. It makes
    sampling more efficient.
    - The leaf nodes are the priorities of the experiences.
    - The root node is the total priority of all experiences.

    Reference:
    - https://arxiv.org/abs/1511.05952
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self._data = [None] * capacity
        self._data_ptr = 0
        self._size = 0

    def add(self, priority: float, data: Any) -> None:
        self._size = min(self._size + 1, self._capacity)
        raise NotImplementedError("Not implemented")

    def update(self, idx: int, priority: float) -> None:
        raise NotImplementedError("Not implemented")

    def get_leaf(self, v: float) -> tuple[int, float, Any]:
        raise NotImplementedError("Not implemented")

    @property
    def total_priority(self) -> float:
        return float(self._tree[0])

    def __len__(self) -> int:
        """The number of data in the buffer."""
        return self._size
