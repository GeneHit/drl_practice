from typing import Generator

from torch import Tensor

from .buffer_utils import BufferTorch
from .data_type import Experience


class ReplayBuffer:
    """A replay buffer implementation using BufferTorch for torch-based storage."""

    def __init__(self, capacity: int) -> None:
        """Initialize the replay buffer with a given capacity.

        Args:
            capacity: Maximum number of experiences the buffer can hold
        """
        self._buffer = BufferTorch(capacity)

    def add_batch(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
    ) -> Tensor:
        """Add a batch of experiences to the buffer.

        Args:
            states: Tensor of states with shape [batch_size, *state_shape]
            actions: Tensor of actions with shape [batch_size, *action_shape]
            rewards: Tensor of rewards with shape [batch_size]
            next_states: Tensor of next states with shape [batch_size, *state_shape]
            dones: Tensor of done flags with shape [batch_size]

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
        return Experience.from_dict(**batch_data)

    def sample_by_idxs(self, idxs: Tensor) -> Experience:
        """Sample experiences by specific indices.

        Args:
            idxs: Tensor of indices to sample

        Returns:
            Experience object containing sampled data as torch tensors
        """
        batch_data = self._buffer.sample_by_idxs(idxs)
        return Experience.from_dict(**batch_data)

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
            yield Experience.from_dict(**batch_data)

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
