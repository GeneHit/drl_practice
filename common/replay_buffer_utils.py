import random
from collections import deque
from dataclasses import dataclass
from typing import Any

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
    def __init__(self, capacity: int = 10000) -> None:
        # the buffer srore the tuple (state, action, reward, next_state, done)
        self.buffer: deque[
            tuple[
                NDArray[Any],
                int | np.integer[Any],
                float | np.floating[Any],
                NDArray[Any],
                bool | np.bool_,
            ]
        ] = deque(maxlen=capacity)

    def add_one(
        self,
        state: NDArray[Any],
        action: int | np.integer[Any],
        reward: float | np.floating[Any],
        next_state: NDArray[Any],
        done: bool | np.bool_,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def add_batch(
        self,
        states: NDArray[Any],
        actions: NDArray[np.integer[Any]],
        rewards: NDArray[np.floating[Any]],
        next_states: NDArray[Any],
        dones: NDArray[np.bool_],
    ) -> None:
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.add_one(s, a, r, ns, d)

    def sample(self, batch_size: int) -> Experience:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return Experience(
            states=torch.tensor(states),
            actions=torch.tensor(actions),
            rewards=torch.tensor(rewards),
            next_states=torch.tensor(next_states),
            dones=torch.tensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)
