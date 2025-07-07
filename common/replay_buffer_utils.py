import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, kw_only=True)
class Experience:
    """The experience of the agent.

    Usually be used for the network input, where the input is a batch of experiences.
    """

    states: NDArray[np.floating[Any]]
    """The state of the environment at the start of the experience."""
    actions: NDArray[np.integer[Any]]
    """The action taken by the agent."""
    rewards: NDArray[np.floating[Any]]
    """The reward received by the agent."""
    next_states: NDArray[np.floating[Any]]
    """The state of the environment at the end of the experience."""
    dones: NDArray[np.bool_]
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
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            next_states=np.array(next_states),
            dones=np.array(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)
