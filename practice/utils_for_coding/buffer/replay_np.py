from typing import Any, Generator

import numpy as np
from numpy.typing import NDArray

from .buffer_utils import BufferNP
from .data_type import Experience


class ReplayBuffer:
    """A replay buffer implementation using BufferNP for numpy-based storage."""

    def __init__(self, capacity: int) -> None:
        """Initialize the replay buffer with a given capacity.

        Args:
            capacity: Maximum number of experiences the buffer can hold
        """
        self._buffer = BufferNP(capacity)

    def add_batch(
        self,
        states: NDArray[Any],
        actions: NDArray[Any],
        rewards: NDArray[Any],
        next_states: NDArray[Any],
        dones: NDArray[Any],
    ) -> NDArray[np.int64]:
        """Add a batch of experiences to the buffer.

        Args:
            states: Array of states with shape [batch_size, *state_shape]
            actions: Array of actions with shape [batch_size, *action_shape]
            rewards: Array of rewards with shape [batch_size]
            next_states: Array of next states with shape [batch_size, *state_shape]
            dones: Array of done flags with shape [batch_size]

        Returns:
            Indices where the data was written
        """
        return self._buffer.add_batch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )

    def sample(self, batch_size: int) -> Experience:
        """Sample a random batch of experiences from the buffer.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Experience object containing sampled data as torch tensors
        """
        batch_data = self._buffer.sample(batch_size)
        return Experience.from_numpy(**batch_data)

    def sample_by_idxs(self, idxs: NDArray[np.int64]) -> Experience:
        """Sample experiences by specific indices.

        Args:
            idxs: Array of indices to sample

        Returns:
            Experience object containing sampled data as torch tensors
        """
        batch_data = self._buffer.sample_by_idxs(idxs)
        return Experience.from_numpy(**batch_data)

    def dataloader(
        self, batch_size: int, shuffle: bool = True
    ) -> Generator[Experience, None, None]:
        """Yield all data in the buffer in batches.

        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle the data

        Yields:
            Experience objects containing batch data as torch tensors
        """
        for batch_data in self._buffer.dataloader(batch_size, shuffle):
            yield Experience.from_numpy(**batch_data)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        """Return the capacity of the buffer."""
        return self._buffer.capacity
