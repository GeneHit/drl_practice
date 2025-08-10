from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True)
class Experience:
    """
    The experience of the agent for one or a batch of steps.

    Attributes:
        states:      State at the start [batch, state_dim]
        actions:     Action taken        [batch, action_dim]
        rewards:     Reward received     [batch]
        next_states: Next state          [batch, state_dim]
        dones:       Done mask (0/1)     [batch]
    """

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor

    def to(self, device: torch.device, dtype: torch.dtype | None = None) -> "Experience":
        """Move all tensors to a device and/or dtype."""
        return Experience(
            states=self.states.to(device, dtype),
            actions=self.actions.to(device, dtype),
            rewards=self.rewards.to(device, dtype),
            next_states=self.next_states.to(device, dtype),
            dones=self.dones.to(device, dtype),
        )


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...],
        state_dtype: type[np.float32] | type[np.uint8] = np.float32,
        action_dtype: type[np.int64] | type[np.float32] = np.int64,
        action_shape: tuple[int, ...] | None = None,  # NEW
    ) -> None:
        self._capacity = capacity
        self._ptr = 0  # current write pointer
        self._size = 0  # current valid data size
        self._printed_warning = False
        self._state_type = state_dtype

        # pre-allocate continuous memory
        self._states = np.empty((capacity, *state_shape), dtype=state_dtype)
        # Support for continuous action spaces
        if action_dtype == np.float32 and action_shape is not None:
            self._actions = np.empty((capacity, *action_shape), dtype=action_dtype)
        else:
            self._actions = np.empty(capacity, dtype=action_dtype)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._next_states = np.empty((capacity, *state_shape), dtype=state_dtype)
        self._dones = np.empty(capacity, dtype=np.bool_)

    def add_batch(
        self,
        states: NDArray[Any],
        actions: NDArray[Any],
        rewards: NDArray[Any],
        next_states: NDArray[Any],
        dones: NDArray[np.bool_],
    ) -> None:
        self._warn_if_necessary(states)
        batch_size = len(states)
        if self._ptr + batch_size <= self._capacity:
            # no wrap-around writing
            indices: Union[slice, list[slice]] = slice(self._ptr, self._ptr + batch_size)
        else:
            # segment writing (wrap-around case)
            head_size = self._capacity - self._ptr
            indices = [slice(self._ptr, None), slice(0, batch_size - head_size)]

        # batch writing
        attr_names = ["_states", "_actions", "_rewards", "_next_states", "_dones"]
        for i, data in enumerate([states, actions, rewards, next_states, dones]):
            arr = getattr(self, attr_names[i])
            # If arr is 2D and data is 2D, allow assignment
            if isinstance(indices, list):
                arr[indices[0]] = data[:head_size]
                arr[indices[1]] = data[head_size:]
            else:
                arr[indices] = data

        # update pointer and count
        self._ptr = (self._ptr + batch_size) % self._capacity
        self._size = min(self._size + batch_size, self._capacity)

    def sample(self, batch_size: int) -> Experience:
        # ensure not sampling empty data
        assert self._size > batch_size, "the buffer size should > batch_size"

        # generate unique random indices
        indices = np.random.choice(self._size, batch_size, replace=False)

        # batch extract data (avoid loop)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]

        # convert to tensor (keep data sharing without copying)
        return Experience(
            states=torch.as_tensor(states),
            actions=torch.as_tensor(actions),
            rewards=torch.as_tensor(rewards),
            next_states=torch.as_tensor(next_states),
            dones=torch.as_tensor(dones),
        )

    def __len__(self) -> int:
        return self._size

    def _warn_if_necessary(self, state: NDArray[Any]) -> None:
        if state.dtype != self._state_type and not self._printed_warning:
            print(f"Warning: state dtype mismatch, expected {self._state_type}, got {state.dtype}")
            self._printed_warning = True


class Buffer:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity

        self._ptr = 0  # current write pointer
        self._size = 0  # current valid data size
        self._data: dict[str, NDArray[Any]] = {}  # dynamic storage for all fields
        self._initialized = False

    def _init_memory(self, **batch_data: NDArray[Any]) -> None:
        if self._initialized:
            return

        # pre-allocate memory for each field
        for key, data in batch_data.items():
            shape = (self._capacity, *data.shape[1:])
            self._data[key] = np.empty(shape, dtype=data.dtype)

        self._initialized = True

    def add_batch(self, **batch_data: NDArray[Any]) -> NDArray[np.int64]:
        batch_size = next(iter(batch_data.values())).shape[0]
        if batch_size == 0:
            return np.array([], dtype=np.int64)

        # initialize storage when first called
        if not self._initialized:
            self._init_memory(**batch_data)

        # batch writing: no wrap-around case and wrap-around case
        ptr_end = self._ptr + batch_size
        if ptr_end <= self._capacity:
            # no wrap-around writing
            indices = slice(self._ptr, ptr_end)
            written_idxs: NDArray[np.int64] = np.arange(self._ptr, ptr_end, dtype=np.int64)
            # write data to each field
            for key, data in batch_data.items():
                self._data[key][indices] = data
        else:
            # segment writing (wrap-around case)
            head_size = self._capacity - self._ptr
            indices_list = [slice(self._ptr, None), slice(0, batch_size - head_size)]
            written_idxs = np.concatenate(
                [
                    np.arange(self._ptr, self._capacity, dtype=np.int64),
                    np.arange(0, batch_size - head_size, dtype=np.int64),
                ]
            )
            # write data to each field
            for key, data in batch_data.items():
                self._data[key][indices_list[0]] = data[:head_size]
                self._data[key][indices_list[1]] = data[head_size:]

        # update pointer and size
        self._ptr = ptr_end % self._capacity
        self._size = min(self._size + batch_size, self._capacity)
        return written_idxs

    def sample(self, batch_size: int) -> dict[str, NDArray[Any]]:
        idxs: NDArray[np.int64] = np.random.randint(0, self._size, batch_size, dtype=np.int64)
        return self.sample_by_idxs(idxs)

    def sample_by_idxs(self, idxs: NDArray[np.int64]) -> dict[str, NDArray[Any]]:
        assert self._initialized, "Memory not initialized. Call add_batch first."
        batch_size = len(idxs)
        assert self._size >= batch_size > 0, f"buffer size {self._size}, but required {batch_size}"

        # Check if all indices are within valid range
        if np.any(idxs >= self._size) or np.any(idxs < 0):
            raise ValueError(
                f"Invalid indices: indices must be in range [0, {self._size}), got {idxs}"
            )

        return {key: arr[idxs] for key, arr in self._data.items()}

    def __len__(self) -> int:
        return self._size
