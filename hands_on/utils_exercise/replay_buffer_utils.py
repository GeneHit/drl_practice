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
        action_dtype: type[np.signedinteger[Any]] = np.int64,
    ):
        self.capacity = capacity
        self.ptr = 0  # current write pointer
        self.size = 0  # current valid data size

        # pre-allocate continuous memory
        self.states = np.empty((capacity, *state_shape), dtype=state_dtype)
        self.actions = np.empty(capacity, dtype=action_dtype)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, *state_shape), dtype=state_dtype)
        self.dones = np.empty(capacity, dtype=np.bool_)

    def add_one(
        self,
        state: NDArray[Any],
        action: int,
        reward: float,
        next_state: NDArray[Any],
        done: bool,
    ) -> None:
        # overwrite old data
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # update pointer and count
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        states: NDArray[Any],
        actions: NDArray[Any],
        rewards: NDArray[Any],
        next_states: NDArray[Any],
        dones: NDArray[Any],
    ) -> None:
        batch_size = len(states)
        if self.ptr + batch_size <= self.capacity:
            # no wrap-around writing
            indices: Union[slice, list[slice]] = slice(
                self.ptr, self.ptr + batch_size
            )
        else:
            # segment writing (wrap-around case)
            head_size = self.capacity - self.ptr
            indices = [slice(self.ptr, None), slice(0, batch_size - head_size)]

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
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> Experience:
        # ensure not sampling empty data
        assert self.size > batch_size, "the buffer size should > batch_size"

        # generate unique random indices
        indices = np.random.choice(self.size, batch_size, replace=False)

        # batch extract data (avoid loop)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]

        # convert to tensor (keep data sharing without copying)
        return Experience(
            states=torch.as_tensor(states),
            actions=torch.as_tensor(actions),
            rewards=torch.as_tensor(rewards),
            next_states=torch.as_tensor(next_states),
            dones=torch.as_tensor(dones),
        )

    def __len__(self) -> int:
        return self.size
