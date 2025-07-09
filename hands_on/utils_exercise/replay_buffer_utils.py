from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True)
class Experience:
    """The experience of the agent.

    Usually be used for the network input, where the input is a batch of experiences.
    """

    states: torch.Tensor
    """The state of the environment at the start of the experience."""
    actions: torch.Tensor
    """The action taken by the agent."""
    rewards: torch.Tensor
    """The reward received by the agent."""
    next_states: torch.Tensor
    """The state of the environment at the end of the experience."""
    dones: torch.Tensor
    """Whether the experience is terminal or truncated.

    2 Cases:
    >> 1. done = terminated or truncated
    >> 2. done = terminaled.
    It depends on the outside caller, where how it consider the truncation.
    """


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...],
        state_dtype: type[np.floating[Any]] = np.float32,
        action_dtype: type[np.integer[Any]] = np.int64,
    ) -> None:
        self._capacity = capacity
        self._ptr = 0  # current write pointer
        self._size = 0  # current valid data size
        self._printed_warning = False
        self._state_type = state_dtype

        # pre-allocate continuous memory
        self._states = np.empty((capacity, *state_shape), dtype=state_dtype)
        self._actions = np.empty(capacity, dtype=action_dtype)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._next_states = np.empty(
            (capacity, *state_shape), dtype=state_dtype
        )
        self._dones = np.empty(capacity, dtype=np.bool_)

    def add_one(
        self,
        state: NDArray[Any],
        action: int,
        reward: float,
        next_state: NDArray[Any],
        done: bool,
    ) -> None:
        self._warn_if_necessary(state)

        # overwrite old data
        self._states[self._ptr] = state
        self._actions[self._ptr] = action
        self._rewards[self._ptr] = reward
        self._next_states[self._ptr] = next_state
        self._dones[self._ptr] = done

        # update pointer and count
        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

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
            indices: Union[slice, list[slice]] = slice(
                self._ptr, self._ptr + batch_size
            )
        else:
            # segment writing (wrap-around case)
            head_size = self._capacity - self._ptr
            indices = [slice(self._ptr, None), slice(0, batch_size - head_size)]

        # batch writing
        for i, data in enumerate(
            [states, actions, rewards, next_states, dones]
        ):
            if isinstance(indices, list):
                getattr(
                    self,
                    ["states", "actions", "rewards", "next_states", "dones"][i],
                )[indices[0]] = data[:head_size]
                getattr(
                    self,
                    ["states", "actions", "rewards", "next_states", "dones"][i],
                )[indices[1]] = data[head_size:]
            else:
                getattr(
                    self,
                    ["states", "actions", "rewards", "next_states", "dones"][i],
                )[indices] = data

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
            print(
                f"Warning: state dtype mismatch, expected {self._state_type}, "
                f"got {state.dtype}"
            )
            self._printed_warning = True
