from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader

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

    def add_experience(self, experience: Experience) -> Tensor:
        """Add an experience to the buffer.

        Args:
            experience: The experience to add

        Returns:
            Indices where the data was written
        """
        return self.add_batch(
            states=experience.states,
            actions=experience.actions,
            rewards=experience.rewards,
            next_states=experience.next_states,
            dones=experience.dones,
        )

    def add_batch(
        self,
        states: Tensor | NDArray[Any],
        actions: Tensor | NDArray[Any],
        rewards: Tensor | NDArray[Any],
        next_states: Tensor | NDArray[Any],
        dones: Tensor | NDArray[Any],
    ) -> Tensor:
        """Add a batch of experiences to the buffer.

        Args:
            states: states with shape [batch_size, *state_shape]
            actions: actions with shape [batch_size, *action_shape]
            rewards: rewards with shape [batch_size]
            next_states: next states with shape [batch_size, *state_shape]
            dones: done flags with shape [batch_size]

        Returns:
            Indices where the data was written
        """
        # Note: the from_numpy use the original numpy array, so the memory is not copied
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards)
        if isinstance(next_states, np.ndarray):
            next_states = torch.from_numpy(next_states)
        if isinstance(dones, np.ndarray):
            dones = torch.from_numpy(dones)

        return self._buffer.add_batch(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )

    def sample(self, batch_size: int, latest: bool = False) -> Experience:
        """Sample a random batch of experiences from the buffer.

        Args:
            batch_size: Number of experiences to sample
            latest: Whether to sample the latest experiences

        Returns:
            Experience object containing sampled data as torch tensors
        """
        batch_data = self._buffer.sample(batch_size, latest=latest)
        return Experience.from_kwargs(**batch_data)

    def sample_by_idxs(self, idxs: Tensor) -> Experience:
        """Sample experiences by specific indices.

        Args:
            idxs: Tensor of indices to sample

        Returns:
            Experience object containing sampled data as torch tensors
        """
        batch_data = self._buffer.sample_by_idxs(idxs)
        return Experience.from_kwargs(**batch_data)

    def dataloader(
        self,
        batch_size: int,
        *,
        ratio: float = 1.0,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> DataLoader[dict[str, Tensor]]:
        """Get a standard torch.utils.data.DataLoader of all data in the buffer.

        - If the buffer stores CPU tensors:
            set num_workers>0, pin_memory=True to get asynchronous H->D transfer.
        - If the buffer stores GPU tensors:
            recommend num_workers=0 (CUDA tensors cannot be pickled between processes)
            and pin_memory=True.

        Args:
            batch_size: Size of each batch
            ratio: Ratio of the buffer to sample from
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for data loading
            pin_memory: Whether to pin memory for data loading
            drop_last: Whether to drop the last batch if it's not full

        Returns:
            A standard torch.utils.data.DataLoader of all data in the buffer
        """
        return self._buffer.dataloader(
            batch_size=batch_size,
            ratio=ratio,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

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
